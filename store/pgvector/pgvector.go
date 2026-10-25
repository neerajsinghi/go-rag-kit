// Package pgvector implements ragkit.VectorStore on PostgreSQL with the
// pgvector extension, using only database/sql.
//
// The package does not import a database driver, so the root module stays
// dependency-free. Register any Postgres driver in your main package, for
// example:
//
//	import _ "github.com/jackc/pgx/v5/stdlib"
//
//	db, err := sql.Open("pgx", os.Getenv("DATABASE_URL"))
//	store, err := pgvector.New(db, "rag_chunks", 1536)
//	err = store.Migrate(ctx) // CREATE EXTENSION / TABLE / HNSW INDEX IF NOT EXISTS
//
// Vectors are sent as pgvector text literals ("[0.1,0.2,...]") and cast with
// ::vector, which works with every driver. Search uses the cosine distance
// operator (<=>) and reports cosine similarity (1 - distance) as the score.
//
// Status: compiled and unit-tested against a fake database/sql driver; run
// your own integration test against a real Postgres before production use.
package pgvector

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

var identRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]{0,62}$`)

// Store is a pgvector-backed vector store. It is safe for concurrent use
// (as is *sql.DB).
type Store struct {
	db    *sql.DB
	table string
	dims  int
}

var _ ragkit.VectorStore = (*Store)(nil)

// New returns a Store using table, which must be a plain SQL identifier
// (letters, digits, underscore). dims is the embedding dimensionality.
func New(db *sql.DB, table string, dims int) (*Store, error) {
	if db == nil {
		return nil, fmt.Errorf("%w: nil *sql.DB", ragkit.ErrInvalidArgument)
	}
	if !identRe.MatchString(table) {
		return nil, fmt.Errorf("%w: invalid table name %q", ragkit.ErrInvalidArgument, table)
	}
	if dims <= 0 || dims > 16000 {
		return nil, fmt.Errorf("%w: dims must be in [1, 16000], got %d", ragkit.ErrInvalidArgument, dims)
	}
	return &Store{db: db, table: table, dims: dims}, nil
}

// MigrateSQL returns the idempotent DDL statements Migrate executes, for
// review or for use with an external migration tool.
func (s *Store) MigrateSQL() []string {
	return []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
	id          text PRIMARY KEY,
	doc_id      text NOT NULL,
	chunk_index integer NOT NULL,
	content     text NOT NULL,
	start_off   integer NOT NULL,
	end_off     integer NOT NULL,
	metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
	embedding   vector(%d) NOT NULL
)`, s.table, s.dims),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_embedding_hnsw ON %s USING hnsw (embedding vector_cosine_ops)`, s.table, s.table),
		fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_doc_id ON %s (doc_id)`, s.table, s.table),
	}
}

// Migrate creates the extension, table and indexes if they do not exist.
func (s *Store) Migrate(ctx context.Context) error {
	for _, stmt := range s.MigrateSQL() {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("pgvector: migrate: %w", err)
		}
	}
	return nil
}

// Upsert implements ragkit.VectorStore. All records are written in a single
// transaction.
func (s *Store) Upsert(ctx context.Context, records ...ragkit.Record) error {
	if len(records) == 0 {
		return nil
	}
	type row struct {
		r    ragkit.Record
		vec  string
		meta []byte
	}
	rows := make([]row, len(records))
	for i, r := range records {
		if r.Chunk.ID == "" {
			return fmt.Errorf("%w: record %d has an empty chunk ID", ragkit.ErrInvalidArgument, i)
		}
		vec, err := s.literal(r.Vector)
		if err != nil {
			return fmt.Errorf("record %q: %w", r.Chunk.ID, err)
		}
		meta := r.Chunk.Metadata
		if meta == nil {
			meta = map[string]string{}
		}
		mb, err := json.Marshal(meta)
		if err != nil {
			return fmt.Errorf("record %q: encode metadata: %w", r.Chunk.ID, err)
		}
		rows[i] = row{r, vec, mb}
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pgvector: begin: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	q := fmt.Sprintf(`INSERT INTO %s (id, doc_id, chunk_index, content, start_off, end_off, metadata, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::vector)
ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, chunk_index = EXCLUDED.chunk_index,
	content = EXCLUDED.content, start_off = EXCLUDED.start_off, end_off = EXCLUDED.end_off,
	metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding`, s.table)
	stmt, err := tx.PrepareContext(ctx, q)
	if err != nil {
		return fmt.Errorf("pgvector: prepare upsert: %w", err)
	}
	defer stmt.Close()
	for _, rw := range rows {
		c := rw.r.Chunk
		if _, err := stmt.ExecContext(ctx, c.ID, c.DocID, c.Index, c.Text, c.Start, c.End, string(rw.meta), rw.vec); err != nil {
			return fmt.Errorf("pgvector: upsert %q: %w", c.ID, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("pgvector: commit: %w", err)
	}
	return nil
}

// Search implements ragkit.VectorStore. Scores are cosine similarities.
func (s *Store) Search(ctx context.Context, query []float32, k int) ([]ragkit.Result, error) {
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be positive, got %d", ragkit.ErrInvalidArgument, k)
	}
	vec, err := s.literal(query)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	q := fmt.Sprintf(`SELECT id, doc_id, chunk_index, content, start_off, end_off, metadata, 1 - (embedding <=> $1::vector) AS score
FROM %s ORDER BY embedding <=> $1::vector LIMIT $2`, s.table)
	rows, err := s.db.QueryContext(ctx, q, vec, k)
	if err != nil {
		return nil, fmt.Errorf("pgvector: search: %w", err)
	}
	defer rows.Close()

	var out []ragkit.Result
	for rows.Next() {
		var (
			r    ragkit.Result
			meta []byte
		)
		c := &r.Chunk
		if err := rows.Scan(&c.ID, &c.DocID, &c.Index, &c.Text, &c.Start, &c.End, &meta, &r.Score); err != nil {
			return nil, fmt.Errorf("pgvector: scan: %w", err)
		}
		if len(meta) > 0 {
			if err := json.Unmarshal(meta, &c.Metadata); err != nil {
				return nil, fmt.Errorf("pgvector: decode metadata of %q: %w", c.ID, err)
			}
			if len(c.Metadata) == 0 {
				c.Metadata = nil
			}
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgvector: rows: %w", err)
	}
	return out, nil
}

// Delete implements ragkit.VectorStore.
func (s *Store) Delete(ctx context.Context, ids ...string) error {
	if len(ids) == 0 {
		return nil
	}
	placeholders := make([]string, len(ids))
	args := make([]any, len(ids))
	for i, id := range ids {
		placeholders[i] = "$" + strconv.Itoa(i+1)
		args[i] = id
	}
	q := fmt.Sprintf(`DELETE FROM %s WHERE id IN (%s)`, s.table, strings.Join(placeholders, ", "))
	if _, err := s.db.ExecContext(ctx, q, args...); err != nil {
		return fmt.Errorf("pgvector: delete: %w", err)
	}
	return nil
}

// literal validates v and encodes it as a pgvector text literal.
func (s *Store) literal(v []float32) (string, error) {
	if len(v) != s.dims {
		return "", fmt.Errorf("%w: vector has %d dims, store has %d", ragkit.ErrDimensionMismatch, len(v), s.dims)
	}
	return VectorLiteral(v)
}

// VectorLiteral encodes v in pgvector's text format, e.g. "[1,0.5,-2]".
// NaN and Inf are rejected because pgvector rejects them.
func VectorLiteral(v []float32) (string, error) {
	var b strings.Builder
	b.Grow(len(v)*10 + 2)
	b.WriteByte('[')
	for i, x := range v {
		f := float64(x)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return "", fmt.Errorf("%w: vector contains NaN or Inf at %d", ragkit.ErrInvalidArgument, i)
		}
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(f, 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String(), nil
}

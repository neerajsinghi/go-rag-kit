package pgvector

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// fakeDB is a minimal database/sql driver that records statements and
// returns canned rows for queries.
type fakeDB struct {
	mu        sync.Mutex
	execs     []call
	queries   []call
	rows      [][]driver.Value
	committed int
}

type call struct {
	query string
	args  []driver.Value
}

func (f *fakeDB) Connect(context.Context) (driver.Conn, error) { return &fakeConn{f}, nil }
func (f *fakeDB) Driver() driver.Driver                        { return fakeDriver{f} }

type fakeDriver struct{ f *fakeDB }

func (d fakeDriver) Open(string) (driver.Conn, error) { return &fakeConn{d.f}, nil }

type fakeConn struct{ f *fakeDB }

func (c *fakeConn) Prepare(q string) (driver.Stmt, error) { return &fakeStmt{c.f, q}, nil }
func (c *fakeConn) Close() error                          { return nil }
func (c *fakeConn) Begin() (driver.Tx, error)             { return &fakeTx{c.f}, nil }

type fakeTx struct{ f *fakeDB }

func (t *fakeTx) Commit() error {
	t.f.mu.Lock()
	defer t.f.mu.Unlock()
	t.f.committed++
	return nil
}
func (t *fakeTx) Rollback() error { return nil }

type fakeStmt struct {
	f *fakeDB
	q string
}

func (s *fakeStmt) Close() error  { return nil }
func (s *fakeStmt) NumInput() int { return -1 }
func (s *fakeStmt) Exec(args []driver.Value) (driver.Result, error) {
	s.f.mu.Lock()
	defer s.f.mu.Unlock()
	s.f.execs = append(s.f.execs, call{s.q, args})
	return driver.RowsAffected(1), nil
}
func (s *fakeStmt) Query(args []driver.Value) (driver.Rows, error) {
	s.f.mu.Lock()
	defer s.f.mu.Unlock()
	s.f.queries = append(s.f.queries, call{s.q, args})
	return &fakeRows{rows: s.f.rows}, nil
}

type fakeRows struct {
	rows [][]driver.Value
	i    int
}

func (r *fakeRows) Columns() []string {
	return []string{"id", "doc_id", "chunk_index", "content", "start_off", "end_off", "metadata", "score"}
}
func (r *fakeRows) Close() error { return nil }
func (r *fakeRows) Next(dest []driver.Value) error {
	if r.i >= len(r.rows) {
		return io.EOF
	}
	copy(dest, r.rows[r.i])
	r.i++
	return nil
}

func newStore(t *testing.T, f *fakeDB) *Store {
	t.Helper()
	db := sql.OpenDB(f)
	t.Cleanup(func() { _ = db.Close() })
	s, err := New(db, "rag_chunks", 3)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestNewValidation(t *testing.T) {
	db := sql.OpenDB(&fakeDB{})
	defer db.Close()
	tests := []struct {
		name  string
		db    *sql.DB
		table string
		dims  int
	}{
		{"nil db", nil, "t", 3},
		{"sql injection", db, "t; DROP TABLE x", 3},
		{"quoted", db, `"t"`, 3},
		{"empty", db, "", 3},
		{"zero dims", db, "t", 0},
		{"huge dims", db, "t", 20000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.db, tt.table, tt.dims); !errors.Is(err, ragkit.ErrInvalidArgument) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

func TestVectorLiteral(t *testing.T) {
	got, err := VectorLiteral([]float32{1, 0.5, -2, 1e-7})
	if err != nil || got != "[1,0.5,-2,1e-07]" {
		t.Fatalf("got %q, %v", got, err)
	}
	if got, _ := VectorLiteral(nil); got != "[]" {
		t.Fatalf("empty = %q", got)
	}
	nan := float32(0)
	nan /= nan
	if _, err := VectorLiteral([]float32{nan}); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatalf("NaN err = %v", err)
	}
}

func TestMigrate(t *testing.T) {
	f := &fakeDB{}
	s := newStore(t, f)
	if err := s.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(f.execs) != 4 || !strings.Contains(f.execs[1].query, "vector(3)") || !strings.Contains(f.execs[2].query, "hnsw") {
		t.Fatalf("unexpected DDL: %+v", f.execs)
	}
}

func TestUpsert(t *testing.T) {
	f := &fakeDB{}
	s := newStore(t, f)
	ctx := context.Background()
	err := s.Upsert(ctx,
		ragkit.Record{Chunk: ragkit.Chunk{ID: "d#0", DocID: "d", Text: "hello", End: 5, Metadata: map[string]string{"k": "v"}}, Vector: []float32{1, 2, 3}},
		ragkit.Record{Chunk: ragkit.Chunk{ID: "d#1", DocID: "d", Index: 1}, Vector: []float32{0, 0, 1}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(f.execs) != 2 || f.committed != 1 {
		t.Fatalf("execs=%d committed=%d", len(f.execs), f.committed)
	}
	a := f.execs[0].args
	if a[0] != "d#0" || a[6] != `{"k":"v"}` || a[7] != "[1,2,3]" {
		t.Fatalf("args = %v", a)
	}
	if f.execs[1].args[6] != `{}` {
		t.Fatalf("nil metadata should encode as {}: %v", f.execs[1].args[6])
	}
	if !strings.Contains(f.execs[0].query, "ON CONFLICT (id) DO UPDATE") {
		t.Fatal("upsert must be idempotent")
	}

	if err := s.Upsert(ctx, ragkit.Record{Chunk: ragkit.Chunk{ID: "x"}, Vector: []float32{1}}); !errors.Is(err, ragkit.ErrDimensionMismatch) {
		t.Fatalf("err = %v", err)
	}
	if err := s.Upsert(ctx, ragkit.Record{Vector: []float32{1, 2, 3}}); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatalf("err = %v", err)
	}
}

func TestSearch(t *testing.T) {
	f := &fakeDB{rows: [][]driver.Value{
		{"d#0", "d", int64(0), "hello", int64(0), int64(5), []byte(`{"path":"a.md"}`), 0.98},
		{"d#1", "d", int64(1), "world", int64(6), int64(11), []byte(`{}`), 0.5},
	}}
	s := newStore(t, f)
	got, err := s.Search(context.Background(), []float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Chunk.ID != "d#0" || got[0].Score != 0.98 || got[0].Chunk.Metadata["path"] != "a.md" || got[1].Chunk.End != 11 {
		t.Fatalf("got %+v", got)
	}
	if got[1].Chunk.Metadata != nil {
		t.Errorf("empty metadata should be nil")
	}
	q := f.queries[0]
	if !strings.Contains(q.query, "<=>") || q.args[0] != "[1,0,0]" || q.args[1] != int64(2) {
		t.Fatalf("query = %+v", q)
	}
	if _, err := s.Search(context.Background(), []float32{1, 0, 0}, 0); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatalf("err = %v", err)
	}
}

func TestDelete(t *testing.T) {
	f := &fakeDB{}
	s := newStore(t, f)
	if err := s.Delete(context.Background()); err != nil || len(f.execs) != 0 {
		t.Fatal("empty delete should be a no-op")
	}
	if err := s.Delete(context.Background(), "a", "b"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(f.execs[0].query, "IN ($1, $2)") || len(f.execs[0].args) != 2 {
		t.Fatalf("exec = %+v", f.execs[0])
	}
}

// Package mongovector implements ragkit.VectorStore on MongoDB Atlas Vector
// Search.
//
// To keep the root module free of the MongoDB driver, the store talks to a
// small Collection interface instead of *mongo.Collection. Adapting the
// official driver takes a dozen lines; see the README for a ready-to-copy
// wrapper for go.mongodb.org/mongo-driver/v2.
//
// The collection needs an Atlas Vector Search index on the embedding field,
// for example:
//
//	{
//	  "fields": [
//	    {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
//	    {"type": "filter", "path": "doc_id"}
//	  ]
//	}
//
// Atlas reports vectorSearchScore, which for cosine similarity is
// (1 + cos) / 2 in [0, 1]; the store returns it unchanged as Result.Score.
//
// Status: unit-tested against a fake Collection; validate against a real
// Atlas cluster before production use.
package mongovector

import (
	"context"
	"fmt"
	"reflect"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// Collection is the subset of a MongoDB collection the store needs.
// Documents are plain maps so any driver (or a test fake) can satisfy it.
type Collection interface {
	// Aggregate runs an aggregation pipeline and returns all result
	// documents.
	Aggregate(ctx context.Context, pipeline []map[string]any) ([]map[string]any, error)
	// ReplaceByID upserts doc as the document whose _id is id.
	ReplaceByID(ctx context.Context, id string, doc map[string]any) error
	// DeleteByIDs deletes all documents whose _id is in ids.
	DeleteByIDs(ctx context.Context, ids []string) error
}

// Default field and tuning values used by New.
const (
	DefaultEmbeddingField = "embedding"
	DefaultNumCandidates  = 10 // multiplier applied to k
	maxNumCandidates      = 10000
)

// Store is an Atlas Vector Search store.
type Store struct {
	coll  Collection
	index string
	// EmbeddingField is the document field holding the vector.
	EmbeddingField string
	// CandidateFactor sets numCandidates = CandidateFactor * k (Atlas
	// recommends 10-20x k for good recall).
	CandidateFactor int
}

var _ ragkit.VectorStore = (*Store)(nil)

// New returns a Store querying the named Atlas Vector Search index.
func New(coll Collection, index string) (*Store, error) {
	if coll == nil {
		return nil, fmt.Errorf("%w: nil collection", ragkit.ErrInvalidArgument)
	}
	if index == "" {
		return nil, fmt.Errorf("%w: empty vector search index name", ragkit.ErrInvalidArgument)
	}
	return &Store{coll: coll, index: index, EmbeddingField: DefaultEmbeddingField, CandidateFactor: DefaultNumCandidates}, nil
}

// Upsert implements ragkit.VectorStore. Each record is replaced
// individually; MongoDB offers no cross-document atomicity here.
func (s *Store) Upsert(ctx context.Context, records ...ragkit.Record) error {
	for i, r := range records {
		if r.Chunk.ID == "" {
			return fmt.Errorf("%w: record %d has an empty chunk ID", ragkit.ErrInvalidArgument, i)
		}
		if len(r.Vector) == 0 {
			return fmt.Errorf("%w: record %q has an empty vector", ragkit.ErrInvalidArgument, r.Chunk.ID)
		}
	}
	for _, r := range records {
		c := r.Chunk
		vec := make([]float64, len(r.Vector))
		for i, x := range r.Vector {
			vec[i] = float64(x)
		}
		meta := make(map[string]any, len(c.Metadata))
		for k, v := range c.Metadata {
			meta[k] = v
		}
		doc := map[string]any{
			"_id":            c.ID,
			"doc_id":         c.DocID,
			"chunk_index":    c.Index,
			"content":        c.Text,
			"start":          c.Start,
			"end":            c.End,
			"metadata":       meta,
			s.EmbeddingField: vec,
		}
		if err := s.coll.ReplaceByID(ctx, c.ID, doc); err != nil {
			return fmt.Errorf("mongovector: upsert %q: %w", c.ID, err)
		}
	}
	return nil
}

// Pipeline returns the aggregation pipeline Search runs.
func (s *Store) Pipeline(query []float32, k int) []map[string]any {
	vec := make([]float64, len(query))
	for i, x := range query {
		vec[i] = float64(x)
	}
	factor := s.CandidateFactor
	if factor < 1 {
		factor = 1
	}
	// Atlas caps numCandidates at 10000 and requires numCandidates >= limit.
	candidates := max(min(k*factor, maxNumCandidates), k)
	return []map[string]any{
		{"$vectorSearch": map[string]any{
			"index":         s.index,
			"path":          s.EmbeddingField,
			"queryVector":   vec,
			"numCandidates": candidates,
			"limit":         k,
		}},
		{"$project": map[string]any{
			"_id": 1, "doc_id": 1, "chunk_index": 1, "content": 1, "start": 1, "end": 1, "metadata": 1,
			"score": map[string]any{"$meta": "vectorSearchScore"},
		}},
	}
}

// Search implements ragkit.VectorStore.
func (s *Store) Search(ctx context.Context, query []float32, k int) ([]ragkit.Result, error) {
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be positive, got %d", ragkit.ErrInvalidArgument, k)
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("%w: empty query vector", ragkit.ErrInvalidArgument)
	}
	docs, err := s.coll.Aggregate(ctx, s.Pipeline(query, k))
	if err != nil {
		return nil, fmt.Errorf("mongovector: search: %w", err)
	}
	out := make([]ragkit.Result, 0, len(docs))
	for i, d := range docs {
		r, err := decode(d)
		if err != nil {
			return nil, fmt.Errorf("mongovector: result %d: %w", i, err)
		}
		out = append(out, r)
	}
	return out, nil
}

// Delete implements ragkit.VectorStore.
func (s *Store) Delete(ctx context.Context, ids ...string) error {
	if len(ids) == 0 {
		return nil
	}
	if err := s.coll.DeleteByIDs(ctx, ids); err != nil {
		return fmt.Errorf("mongovector: delete: %w", err)
	}
	return nil
}

func decode(d map[string]any) (ragkit.Result, error) {
	var r ragkit.Result
	id, ok := d["_id"].(string)
	if !ok || id == "" {
		return r, fmt.Errorf("missing or non-string _id: %v", d["_id"])
	}
	r.Chunk.ID = id
	r.Chunk.DocID, _ = d["doc_id"].(string)
	r.Chunk.Text, _ = d["content"].(string)
	r.Chunk.Index = int(number(d["chunk_index"]))
	r.Chunk.Start = int(number(d["start"]))
	r.Chunk.End = int(number(d["end"]))
	r.Score = number(d["score"])
	if m, ok := asMap(d["metadata"]); ok && len(m) > 0 {
		r.Chunk.Metadata = make(map[string]string, len(m))
		for k, v := range m {
			if s, ok := v.(string); ok {
				r.Chunk.Metadata[k] = s
			} else {
				r.Chunk.Metadata[k] = fmt.Sprint(v)
			}
		}
	}
	return r, nil
}

// asMap converts a decoded sub-document to map[string]any. It accepts any
// map with string keys (including named types such as bson.M) and ordered
// documents shaped like bson.D (a slice of structs with Key and Value
// fields), without importing a driver.
func asMap(v any) (map[string]any, bool) {
	if m, ok := v.(map[string]any); ok {
		return m, true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return nil, false
		}
		out := make(map[string]any, rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			out[iter.Key().String()] = iter.Value().Interface()
		}
		return out, true
	case reflect.Slice:
		et := rv.Type().Elem()
		if et.Kind() != reflect.Struct {
			return nil, false
		}
		kf, kok := et.FieldByName("Key")
		vf, vok := et.FieldByName("Value")
		if !kok || !vok || kf.Type.Kind() != reflect.String {
			return nil, false
		}
		out := make(map[string]any, rv.Len())
		for i := range rv.Len() {
			e := rv.Index(i)
			out[e.FieldByIndex(kf.Index).String()] = e.FieldByIndex(vf.Index).Interface()
		}
		return out, true
	}
	return nil, false
}

// number converts the numeric types a BSON decoder may produce to float64.
func number(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int32:
		return float64(n)
	case int64:
		return float64(n)
	}
	return 0
}

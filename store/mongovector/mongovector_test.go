package mongovector

import (
	"context"
	"errors"
	"testing"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

type fakeColl struct {
	docs     map[string]map[string]any
	pipeline []map[string]any
	results  []map[string]any
	deleted  []string
	err      error
}

func (f *fakeColl) Aggregate(_ context.Context, p []map[string]any) ([]map[string]any, error) {
	f.pipeline = p
	return f.results, f.err
}

func (f *fakeColl) ReplaceByID(_ context.Context, id string, doc map[string]any) error {
	if f.err != nil {
		return f.err
	}
	if f.docs == nil {
		f.docs = map[string]map[string]any{}
	}
	f.docs[id] = doc
	return nil
}

func (f *fakeColl) DeleteByIDs(_ context.Context, ids []string) error {
	f.deleted = append(f.deleted, ids...)
	return f.err
}

// bsonM and bsonD mimic the driver's named document types.
type (
	bsonM map[string]any
	bsonE struct {
		Key   string
		Value any
	}
	bsonD []bsonE
)

func TestNew(t *testing.T) {
	if _, err := New(nil, "idx"); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if _, err := New(&fakeColl{}, ""); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatal(err)
	}
}

func TestUpsert(t *testing.T) {
	f := &fakeColl{}
	s, _ := New(f, "vector_index")
	err := s.Upsert(context.Background(), ragkit.Record{
		Chunk:  ragkit.Chunk{ID: "d#0", DocID: "d", Text: "hi", End: 2, Metadata: map[string]string{"k": "v"}},
		Vector: []float32{0.5, 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	doc := f.docs["d#0"]
	if doc["_id"] != "d#0" || doc["content"] != "hi" || doc["metadata"].(map[string]any)["k"] != "v" {
		t.Fatalf("doc = %v", doc)
	}
	if v := doc["embedding"].([]float64); len(v) != 2 || v[0] != 0.5 {
		t.Fatalf("embedding = %v", v)
	}
	for _, bad := range []ragkit.Record{{Vector: []float32{1}}, {Chunk: ragkit.Chunk{ID: "x"}}} {
		if err := s.Upsert(context.Background(), bad); !errors.Is(err, ragkit.ErrInvalidArgument) {
			t.Errorf("err = %v", err)
		}
	}
}

func TestPipeline(t *testing.T) {
	s, _ := New(&fakeColl{}, "vector_index")
	vs := s.Pipeline([]float32{1, 2}, 5)[0]["$vectorSearch"].(map[string]any)
	if vs["index"] != "vector_index" || vs["path"] != "embedding" || vs["limit"] != 5 || vs["numCandidates"] != 50 {
		t.Fatalf("$vectorSearch = %v", vs)
	}
	vs = s.Pipeline([]float32{1}, 5000)[0]["$vectorSearch"].(map[string]any)
	if vs["numCandidates"] != 10000 {
		t.Fatalf("numCandidates should be capped: %v", vs["numCandidates"])
	}
}

func TestSearchDecodes(t *testing.T) {
	f := &fakeColl{results: []map[string]any{
		{"_id": "d#0", "doc_id": "d", "content": "hello", "chunk_index": int32(0), "start": int32(0), "end": int64(5), "score": 0.97, "metadata": bsonD{{"path", "a.md"}}},
		{"_id": "d#1", "doc_id": "d", "content": "world", "chunk_index": int32(1), "score": float32(0.5), "metadata": bsonM{"page": int32(2)}},
		{"_id": "d#2", "score": 0.1},
	}}
	s, _ := New(f, "idx")
	got, err := s.Search(context.Background(), []float32{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].Score != 0.97 || got[0].Chunk.End != 5 || got[0].Chunk.Metadata["path"] != "a.md" {
		t.Fatalf("got %+v", got[0])
	}
	if got[1].Chunk.Index != 1 || got[1].Chunk.Metadata["page"] != "2" || got[1].Score != 0.5 {
		t.Fatalf("got %+v", got[1])
	}
	if got[2].Chunk.Metadata != nil {
		t.Fatal("missing metadata should be nil")
	}
}

func TestSearchErrors(t *testing.T) {
	ctx := context.Background()
	s, _ := New(&fakeColl{}, "idx")
	if _, err := s.Search(ctx, []float32{1}, 0); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatal(err)
	}
	if _, err := s.Search(ctx, nil, 1); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatal(err)
	}
	boom := errors.New("boom")
	s, _ = New(&fakeColl{err: boom}, "idx")
	if _, err := s.Search(ctx, []float32{1}, 1); !errors.Is(err, boom) {
		t.Fatal(err)
	}
	s, _ = New(&fakeColl{results: []map[string]any{{"_id": 42}}}, "idx")
	if _, err := s.Search(ctx, []float32{1}, 1); err == nil {
		t.Fatal("expected error for non-string _id")
	}
}

func TestDelete(t *testing.T) {
	f := &fakeColl{}
	s, _ := New(f, "idx")
	if err := s.Delete(context.Background()); err != nil || f.deleted != nil {
		t.Fatal("empty delete should be a no-op")
	}
	if err := s.Delete(context.Background(), "a", "b"); err != nil || len(f.deleted) != 2 {
		t.Fatalf("deleted = %v, %v", f.deleted, err)
	}
}

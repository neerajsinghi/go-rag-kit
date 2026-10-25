package memory

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

func rec(id string, v ...float32) ragkit.Record {
	return ragkit.Record{Chunk: ragkit.Chunk{ID: id, DocID: "doc-" + id, Text: id}, Vector: v}
}

func ids(rs []ragkit.Result) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.Chunk.ID
	}
	return out
}

func TestSearchTopK(t *testing.T) {
	ctx := context.Background()
	s := New()
	err := s.Upsert(ctx,
		rec("east", 1, 0),
		rec("north", 0, 1),
		rec("northeast", 1, 1),
		rec("west", -1, 0),
		rec("east-long", 10, 0), // same direction as east: cosine ignores magnitude
	)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		query []float32
		k     int
		want  []string
	}{
		{"exact match with tie broken by id", []float32{1, 0}, 2, []string{"east", "east-long"}},
		{"k larger than store", []float32{0, 1}, 10, []string{"north", "northeast", "east", "east-long", "west"}},
		{"diagonal", []float32{1, 1}, 1, []string{"northeast"}},
		{"opposite", []float32{-1, 0}, 1, []string{"west"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := s.Search(ctx, tt.query, tt.k)
			if err != nil {
				t.Fatal(err)
			}
			if fmt.Sprint(ids(got)) != fmt.Sprint(tt.want) {
				t.Fatalf("got %v, want %v", ids(got), tt.want)
			}
			for i := 1; i < len(got); i++ {
				if got[i].Score > got[i-1].Score {
					t.Fatalf("results not sorted: %v", got)
				}
			}
		})
	}
	got, _ := s.Search(ctx, []float32{1, 0}, 1)
	if math.Abs(got[0].Score-1) > 1e-6 {
		t.Errorf("identical direction score = %f, want 1", got[0].Score)
	}
}

func TestUpsertReplacesAndDelete(t *testing.T) {
	ctx := context.Background()
	s := New()
	if err := s.Upsert(ctx, rec("a", 1, 0), rec("b", 0, 1)); err != nil {
		t.Fatal(err)
	}
	if err := s.Upsert(ctx, rec("a", 0, 1)); err != nil {
		t.Fatal(err)
	}
	if s.Len() != 2 {
		t.Fatalf("Len = %d, want 2", s.Len())
	}
	got, _ := s.Search(ctx, []float32{0, 1}, 2)
	if got[0].Score < 0.999 || got[1].Score < 0.999 {
		t.Fatalf("upsert did not replace vector: %v", got)
	}
	if err := s.Delete(ctx, "a", "missing"); err != nil {
		t.Fatal(err)
	}
	if s.Len() != 1 {
		t.Fatalf("Len after delete = %d", s.Len())
	}
	_ = s.Delete(ctx, "b")
	// Empty store resets dimensionality.
	if err := s.Upsert(ctx, rec("c", 1, 2, 3)); err != nil {
		t.Fatalf("upsert after reset: %v", err)
	}
}

func TestInvalidInput(t *testing.T) {
	ctx := context.Background()
	s := New()
	if err := s.Upsert(ctx, rec("a", 1, 0)); err != nil {
		t.Fatal(err)
	}
	upserts := []struct {
		name string
		r    ragkit.Record
		want error
	}{
		{"empty id", rec("", 1, 0), ragkit.ErrInvalidArgument},
		{"empty vector", rec("x"), ragkit.ErrInvalidArgument},
		{"zero vector", rec("x", 0, 0), ragkit.ErrInvalidArgument},
		{"nan", rec("x", float32(math.NaN()), 1), ragkit.ErrInvalidArgument},
		{"inf", rec("x", float32(math.Inf(1)), 1), ragkit.ErrInvalidArgument},
		{"wrong dims", rec("x", 1, 0, 0), ragkit.ErrDimensionMismatch},
	}
	for _, tt := range upserts {
		t.Run("upsert "+tt.name, func(t *testing.T) {
			if err := s.Upsert(ctx, rec("ok", 0, 1), tt.r); !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
			if s.Len() != 1 {
				t.Fatal("failed batch was partially applied")
			}
		})
	}
	searches := []struct {
		name string
		q    []float32
		k    int
		want error
	}{
		{"zero k", []float32{1, 0}, 0, ragkit.ErrInvalidArgument},
		{"zero query", []float32{0, 0}, 1, ragkit.ErrInvalidArgument},
		{"empty query", nil, 1, ragkit.ErrInvalidArgument},
		{"wrong dims", []float32{1, 0, 0}, 1, ragkit.ErrDimensionMismatch},
	}
	for _, tt := range searches {
		t.Run("search "+tt.name, func(t *testing.T) {
			if _, err := s.Search(ctx, tt.q, tt.k); !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestEmptyStoreSearch(t *testing.T) {
	got, err := New().Search(context.Background(), []float32{1}, 3)
	if err != nil || len(got) != 0 {
		t.Fatalf("got %v, %v", got, err)
	}
}

func TestNoAliasing(t *testing.T) {
	ctx := context.Background()
	s := New()
	v := []float32{1, 0}
	r := rec("a", v...)
	r.Chunk.Metadata = map[string]string{"k": "v"}
	_ = s.Upsert(ctx, r)
	v[0], r.Chunk.Metadata["k"] = 0, "changed"
	got, _ := s.Search(ctx, []float32{1, 0}, 1)
	if got[0].Score < 0.999 || got[0].Chunk.Metadata["k"] != "v" {
		t.Fatal("store aliases caller memory")
	}
	got[0].Chunk.Metadata["k"] = "x"
	again, _ := s.Search(ctx, []float32{1, 0}, 1)
	if again[0].Chunk.Metadata["k"] != "v" {
		t.Fatal("search results alias store memory")
	}
}

func TestConcurrentAccess(t *testing.T) {
	ctx := context.Background()
	s := New()
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for j := range 50 {
				_ = s.Upsert(ctx, rec(fmt.Sprintf("%d-%d", i, j), float32(i+1), float32(j)))
			}
		}()
		go func() {
			defer wg.Done()
			for range 50 {
				_, _ = s.Search(ctx, []float32{1, 1}, 5)
			}
		}()
	}
	wg.Wait()
	if s.Len() != 400 {
		t.Fatalf("Len = %d, want 400", s.Len())
	}
}

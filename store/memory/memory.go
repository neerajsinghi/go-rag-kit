// Package memory provides an in-process ragkit.VectorStore that ranks by
// exact cosine similarity.
//
// It performs a brute-force scan with a bounded heap, which is exact and
// fast enough for tens of thousands of chunks: ideal for tests, CLIs,
// evaluation and small corpora. Use store/pgvector or store/mongovector for
// larger or persistent workloads.
package memory

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// Store is an in-memory cosine-similarity vector store. The zero value is
// ready to use. It is safe for concurrent use.
//
// The dimensionality is fixed by the first upserted vector and reset when the
// store becomes empty.
type Store struct {
	mu      sync.RWMutex
	dims    int
	records map[string]entry
}

type entry struct {
	chunk ragkit.Chunk
	vec   []float32 // unit length
}

var _ ragkit.VectorStore = (*Store)(nil)

// New returns an empty Store.
func New() *Store { return &Store{} }

// Len returns the number of stored records.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.records)
}

// Upsert implements ragkit.VectorStore. The batch is validated as a whole
// before anything is written, so a failed Upsert leaves the store unchanged.
// Vectors are copied and normalised; zero, NaN or infinite vectors are
// rejected.
func (s *Store) Upsert(ctx context.Context, records ...ragkit.Record) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	dims := s.dims
	prepared := make([]entry, len(records))
	for i, r := range records {
		if r.Chunk.ID == "" {
			return fmt.Errorf("%w: record %d has an empty chunk ID", ragkit.ErrInvalidArgument, i)
		}
		if len(r.Vector) == 0 {
			return fmt.Errorf("%w: record %q has an empty vector", ragkit.ErrInvalidArgument, r.Chunk.ID)
		}
		if dims == 0 {
			dims = len(r.Vector)
		}
		if len(r.Vector) != dims {
			return fmt.Errorf("%w: record %q has %d dims, store has %d", ragkit.ErrDimensionMismatch, r.Chunk.ID, len(r.Vector), dims)
		}
		vec, err := normalize(r.Vector)
		if err != nil {
			return fmt.Errorf("record %q: %w", r.Chunk.ID, err)
		}
		c := r.Chunk
		c.Metadata = ragkit.CopyMetadata(c.Metadata)
		prepared[i] = entry{chunk: c, vec: vec}
	}

	if s.records == nil {
		s.records = make(map[string]entry, len(prepared))
	}
	for _, e := range prepared {
		s.records[e.chunk.ID] = e
	}
	if len(s.records) > 0 {
		s.dims = dims
	}
	return nil
}

// Search implements ragkit.VectorStore. Scores are cosine similarities in
// [-1, 1]; ties are broken by chunk ID for deterministic output.
func (s *Store) Search(ctx context.Context, query []float32, k int) ([]ragkit.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be positive, got %d", ragkit.ErrInvalidArgument, k)
	}
	q, err := normalize(query)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.records) == 0 {
		return nil, nil
	}
	if len(q) != s.dims {
		return nil, fmt.Errorf("%w: query has %d dims, store has %d", ragkit.ErrDimensionMismatch, len(q), s.dims)
	}

	h := make(minHeap, 0, min(k, len(s.records)))
	for _, e := range s.records {
		r := ragkit.Result{Chunk: e.chunk, Score: dot(q, e.vec)}
		if len(h) < k {
			heap.Push(&h, r)
		} else if worse(h[0], r) {
			h[0] = r
			heap.Fix(&h, 0)
		}
	}
	out := []ragkit.Result(h)
	sort.Slice(out, func(i, j int) bool { return worse(out[j], out[i]) })
	for i := range out {
		out[i].Chunk.Metadata = ragkit.CopyMetadata(out[i].Chunk.Metadata)
	}
	return out, nil
}

// Delete implements ragkit.VectorStore.
func (s *Store) Delete(ctx context.Context, ids ...string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		delete(s.records, id)
	}
	if len(s.records) == 0 {
		s.dims = 0
	}
	return nil
}

// worse reports whether a ranks below b: lower score, or equal score and a
// larger chunk ID.
func worse(a, b ragkit.Result) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.Chunk.ID > b.Chunk.ID
}

// minHeap keeps the worst of the current top-k at the root.
type minHeap []ragkit.Result

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return worse(h[i], h[j]) }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(ragkit.Result)) }
func (h *minHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

func normalize(v []float32) ([]float32, error) {
	if len(v) == 0 {
		return nil, fmt.Errorf("%w: empty vector", ragkit.ErrInvalidArgument)
	}
	var sum float64
	for _, x := range v {
		f := float64(x)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, fmt.Errorf("%w: vector contains NaN or Inf", ragkit.ErrInvalidArgument)
		}
		sum += f * f
	}
	if sum == 0 {
		return nil, fmt.Errorf("%w: zero vector", ragkit.ErrInvalidArgument)
	}
	n := math.Sqrt(sum)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / n)
	}
	return out, nil
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

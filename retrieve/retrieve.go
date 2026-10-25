// Package retrieve implements ragkit.Retriever strategies: dense vector
// retrieval, BM25 keyword retrieval, and hybrid retrieval that fuses any
// number of retrievers with Reciprocal Rank Fusion (RRF).
//
// All of it is pure Go with no dependencies beyond the standard library.
package retrieve

import (
	"context"
	"fmt"
	"sort"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// Vector retrieves by embedding the query and searching a VectorStore.
type Vector struct {
	Embedder ragkit.Embedder
	Store    ragkit.VectorStore
}

var _ ragkit.Retriever = (*Vector)(nil)

// NewVector returns a Vector retriever.
func NewVector(e ragkit.Embedder, s ragkit.VectorStore) *Vector {
	return &Vector{Embedder: e, Store: s}
}

// Retrieve implements ragkit.Retriever.
func (v *Vector) Retrieve(ctx context.Context, query string, k int) ([]ragkit.Result, error) {
	if v.Embedder == nil || v.Store == nil {
		return nil, fmt.Errorf("%w: vector retriever needs an embedder and a store", ragkit.ErrInvalidArgument)
	}
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be positive, got %d", ragkit.ErrInvalidArgument, k)
	}
	vecs, err := v.Embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("embedder returned %d vectors for 1 query", len(vecs))
	}
	return v.Store.Search(ctx, vecs[0], k)
}

// sortResults orders results by descending score, breaking ties by chunk ID
// so output is deterministic.
func sortResults(rs []ragkit.Result) {
	sort.Slice(rs, func(i, j int) bool {
		if rs[i].Score != rs[j].Score {
			return rs[i].Score > rs[j].Score
		}
		return rs[i].Chunk.ID < rs[j].Chunk.ID
	})
}

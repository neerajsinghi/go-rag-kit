// Package fake provides a deterministic, offline ragkit.Embedder for tests,
// examples and evaluation baselines.
//
// It uses the hashing trick: every term is hashed into one of Dims buckets
// with a pseudo-random sign, and the resulting vector is L2-normalised. Texts
// sharing vocabulary therefore get high cosine similarity, which is enough to
// exercise vector retrieval end-to-end without network access or API keys.
// It has no semantic understanding: synonyms are unrelated.
package fake

import (
	"context"
	"hash/fnv"
	"math"
	"strings"

	"github.com/neerajsinghi/go-rag-kit/internal/tokenize"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// DefaultDims is the vector size used when Embedder.Dims is zero or negative.
const DefaultDims = 256

// Embedder is a deterministic hashing embedder. The zero value is ready to
// use and produces DefaultDims-dimensional vectors. It is safe for concurrent
// use.
type Embedder struct {
	Dims int
}

var _ ragkit.Embedder = (*Embedder)(nil)

// New returns an Embedder producing dims-dimensional vectors.
func New(dims int) *Embedder { return &Embedder{Dims: dims} }

// Embed implements ragkit.Embedder. It never fails except on a cancelled
// context.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = e.Vector(t)
	}
	return out, nil
}

// Vector embeds a single text. The result is always non-zero and unit length.
func (e *Embedder) Vector(text string) []float32 {
	dims := e.Dims
	if dims <= 0 {
		dims = DefaultDims
	}
	acc := make([]float64, dims)
	terms := tokenize.Terms(text)
	if len(terms) == 0 {
		// Keep the vector non-zero so stores never see a zero vector.
		terms = []string{strings.ToLower(strings.TrimSpace(text))}
	}
	for _, term := range terms {
		h := fnv.New64a()
		_, _ = h.Write([]byte(term))
		sum := h.Sum64()
		sign := 1.0
		if sum>>63 == 1 {
			sign = -1.0
		}
		acc[sum%uint64(dims)] += sign
	}
	var norm float64
	for _, v := range acc {
		norm += v * v
	}
	if norm == 0 {
		// Opposite-signed collisions cancelled out; fall back to a fixed axis.
		acc[0], norm = 1, 1
	}
	norm = math.Sqrt(norm)
	vec := make([]float32, dims)
	for i, v := range acc {
		vec[i] = float32(v / norm)
	}
	return vec
}

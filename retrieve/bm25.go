package retrieve

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/neerajsinghi/go-rag-kit/internal/tokenize"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// Default BM25 parameters (Robertson et al.; also the Lucene defaults).
const (
	DefaultK1 = 1.2
	DefaultB  = 0.75
)

// BM25 is an in-memory inverted index that ranks chunks with Okapi BM25:
//
//	score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
//	IDF(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
//
// Text is analysed by lower-casing, splitting on non-alphanumeric runes and
// dropping a small English stopword list. BM25 is safe for concurrent use and
// implements both ragkit.Retriever and ragkit.ChunkSink.
type BM25 struct {
	k1, b float64

	mu       sync.RWMutex
	docs     map[string]*bm25Doc
	postings map[string]map[string]int // term -> chunk ID -> term frequency
	totalLen int
}

type bm25Doc struct {
	chunk  ragkit.Chunk
	length int
	terms  []string // unique terms, for removal
}

var (
	_ ragkit.Retriever = (*BM25)(nil)
	_ ragkit.ChunkSink = (*BM25)(nil)
)

// BM25Option configures a BM25 index.
type BM25Option func(*BM25)

// WithK1 sets the term-frequency saturation parameter (default 1.2).
// Negative values are ignored.
func WithK1(k1 float64) BM25Option {
	return func(b *BM25) {
		if k1 >= 0 {
			b.k1 = k1
		}
	}
}

// WithB sets the length-normalisation parameter in [0, 1] (default 0.75).
// Out-of-range values are ignored.
func WithB(v float64) BM25Option {
	return func(b *BM25) {
		if v >= 0 && v <= 1 {
			b.b = v
		}
	}
}

// NewBM25 returns an empty BM25 index.
func NewBM25(opts ...BM25Option) *BM25 {
	idx := &BM25{
		k1:       DefaultK1,
		b:        DefaultB,
		docs:     make(map[string]*bm25Doc),
		postings: make(map[string]map[string]int),
	}
	for _, o := range opts {
		o(idx)
	}
	return idx
}

// Len returns the number of indexed chunks.
func (idx *BM25) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.docs)
}

// Add indexes chunks, replacing any previously indexed chunk with the same
// ID. Chunks with an empty ID are ignored.
func (idx *BM25) Add(chunks ...ragkit.Chunk) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, c := range chunks {
		if c.ID == "" {
			continue
		}
		idx.removeLocked(c.ID)
		terms := tokenize.Terms(c.Text)
		tf := make(map[string]int, len(terms))
		for _, t := range terms {
			tf[t]++
		}
		d := &bm25Doc{length: len(terms), terms: make([]string, 0, len(tf))}
		d.chunk = c
		d.chunk.Metadata = ragkit.CopyMetadata(c.Metadata)
		for t, n := range tf {
			p := idx.postings[t]
			if p == nil {
				p = make(map[string]int)
				idx.postings[t] = p
			}
			p[c.ID] = n
			d.terms = append(d.terms, t)
		}
		idx.docs[c.ID] = d
		idx.totalLen += d.length
	}
}

// Remove deletes chunks by ID. Unknown IDs are ignored.
func (idx *BM25) Remove(ids ...string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, id := range ids {
		idx.removeLocked(id)
	}
}

func (idx *BM25) removeLocked(id string) {
	d, ok := idx.docs[id]
	if !ok {
		return
	}
	for _, t := range d.terms {
		p := idx.postings[t]
		delete(p, id)
		if len(p) == 0 {
			delete(idx.postings, t)
		}
	}
	idx.totalLen -= d.length
	delete(idx.docs, id)
}

// Retrieve implements ragkit.Retriever. Only chunks sharing at least one
// query term are returned, so fewer than k results is normal. A query with
// no indexable terms returns no results.
func (idx *BM25) Retrieve(ctx context.Context, query string, k int) ([]ragkit.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be positive, got %d", ragkit.ErrInvalidArgument, k)
	}
	terms := tokenize.Terms(query)

	idx.mu.RLock()
	defer idx.mu.RUnlock()
	n := len(idx.docs)
	if n == 0 || len(terms) == 0 {
		return nil, nil
	}
	avgdl := float64(idx.totalLen) / float64(n)
	if avgdl == 0 {
		avgdl = 1
	}

	scores := make(map[string]float64)
	seen := make(map[string]struct{}, len(terms))
	for _, t := range terms {
		if _, dup := seen[t]; dup {
			continue
		}
		seen[t] = struct{}{}
		p := idx.postings[t]
		if len(p) == 0 {
			continue
		}
		df := float64(len(p))
		idf := math.Log(1 + (float64(n)-df+0.5)/(df+0.5))
		for id, tf := range p {
			f := float64(tf)
			dl := float64(idx.docs[id].length)
			scores[id] += idf * f * (idx.k1 + 1) / (f + idx.k1*(1-idx.b+idx.b*dl/avgdl))
		}
	}

	out := make([]ragkit.Result, 0, len(scores))
	for id, s := range scores {
		out = append(out, ragkit.Result{Chunk: idx.docs[id].chunk, Score: s})
	}
	sortResults(out)
	if len(out) > k {
		out = out[:k]
	}
	for i := range out {
		out[i].Chunk.Metadata = ragkit.CopyMetadata(out[i].Chunk.Metadata)
	}
	return out, nil
}

// Package ragkit is a small, dependency-free toolkit for building
// retrieval-augmented generation (RAG) pipelines in Go.
//
// The root package defines the core data types (Document, Chunk, Record,
// Result) and the four interfaces every component plugs into:
//
//   - Chunker splits a Document into Chunks.
//   - Embedder turns text into vectors.
//   - VectorStore stores vectors and answers nearest-neighbour queries.
//   - Retriever answers a text query with ranked Results.
//
// Concrete implementations live in sub-packages: chunk, embed/openai,
// embed/fake, store/memory, store/pgvector, store/mongovector, retrieve and
// eval. Sub-packages import the root package, never the other way round.
package ragkit

import (
	"context"
	"errors"
	"strconv"
)

// Common errors returned by ragkit components. Implementations wrap them with
// fmt.Errorf("%w: ...") so callers can test with errors.Is.
var (
	// ErrInvalidArgument reports a bad parameter such as an empty ID,
	// a non-positive k or an inconsistent configuration.
	ErrInvalidArgument = errors.New("ragkit: invalid argument")
	// ErrDimensionMismatch reports a vector whose length does not match the
	// dimensionality expected by a store.
	ErrDimensionMismatch = errors.New("ragkit: vector dimension mismatch")
)

// Document is a unit of source content, such as a file or a web page.
type Document struct {
	// ID uniquely identifies the document. It is required.
	ID string `json:"id"`
	// Text is the raw document content.
	Text string `json:"text"`
	// Metadata is arbitrary source metadata (path, URL, title, ...). It is
	// copied onto every chunk produced from the document.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Chunk is a contiguous slice of a Document that is embedded and retrieved
// as a unit.
type Chunk struct {
	// ID uniquely identifies the chunk; see ChunkID.
	ID string `json:"id"`
	// DocID is the ID of the source Document.
	DocID string `json:"doc_id"`
	// Index is the zero-based position of the chunk within its document.
	Index int `json:"index"`
	// Text is the chunk content.
	Text string `json:"text"`
	// Start and End are byte offsets into the source Document's Text, so
	// that doc.Text[Start:End] == Text.
	Start int `json:"start"`
	End   int `json:"end"`
	// Metadata holds the document metadata plus any chunker-specific keys
	// (for example the markdown section path).
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Record pairs a Chunk with its embedding vector for storage.
type Record struct {
	Chunk  Chunk
	Vector []float32
}

// Result is a retrieved Chunk with a relevance score. Higher is better; the
// scale depends on the producer (cosine similarity, BM25, RRF, ...).
type Result struct {
	Chunk Chunk   `json:"chunk"`
	Score float64 `json:"score"`
}

// Chunker splits a Document into Chunks.
type Chunker interface {
	Chunk(doc Document) ([]Chunk, error)
}

// Embedder converts texts into embedding vectors. The returned slice must
// have exactly one vector per input text, in the same order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// VectorStore persists embedded chunks and answers top-k similarity queries.
type VectorStore interface {
	// Upsert inserts or replaces records keyed by Chunk.ID.
	Upsert(ctx context.Context, records ...Record) error
	// Search returns at most k results ordered by descending similarity.
	Search(ctx context.Context, query []float32, k int) ([]Result, error)
	// Delete removes the records with the given chunk IDs. Unknown IDs are
	// ignored.
	Delete(ctx context.Context, ids ...string) error
}

// Retriever answers a text query with at most k Results ordered best-first.
type Retriever interface {
	Retrieve(ctx context.Context, query string, k int) ([]Result, error)
}

// ChunkID returns the canonical ID for the chunk at index within docID.
func ChunkID(docID string, index int) string {
	return docID + "#" + strconv.Itoa(index)
}

// CopyMetadata returns a shallow copy of m. It returns nil for an empty map.
func CopyMetadata(m map[string]string) map[string]string {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

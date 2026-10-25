package ragkit

import (
	"context"
	"fmt"
)

// DefaultBatchSize is the number of chunks embedded per Embedder call when
// Indexer.BatchSize is zero.
const DefaultBatchSize = 64

// ChunkSink receives chunks during indexing. Keyword indexes such as
// retrieve.BM25 implement it so a single Indexer can feed both the vector
// store and the keyword index.
type ChunkSink interface {
	Add(chunks ...Chunk)
}

// Indexer wires a Chunker, an Embedder and a VectorStore together:
// documents are chunked, embedded in batches and upserted.
//
// Embedder and Store must either both be set or both be nil; leaving them nil
// gives keyword-only indexing into Sinks.
type Indexer struct {
	Chunker  Chunker
	Embedder Embedder
	Store    VectorStore
	// Sinks additionally receive every chunk after it has been stored.
	Sinks []ChunkSink
	// BatchSize bounds the number of texts per Embed call. Zero means
	// DefaultBatchSize.
	BatchSize int
}

// Index chunks, embeds and stores docs. It returns the number of chunks
// indexed. On error, chunks from earlier batches may already be stored;
// upserts are idempotent, so re-running Index is safe.
func (ix *Indexer) Index(ctx context.Context, docs ...Document) (int, error) {
	if ix.Chunker == nil {
		return 0, fmt.Errorf("%w: indexer has no chunker", ErrInvalidArgument)
	}
	if (ix.Embedder == nil) != (ix.Store == nil) {
		return 0, fmt.Errorf("%w: indexer needs both an embedder and a store, or neither", ErrInvalidArgument)
	}
	if ix.BatchSize < 0 {
		return 0, fmt.Errorf("%w: negative batch size %d", ErrInvalidArgument, ix.BatchSize)
	}
	batchSize := ix.BatchSize
	if batchSize == 0 {
		batchSize = DefaultBatchSize
	}

	var chunks []Chunk
	for _, doc := range docs {
		cs, err := ix.Chunker.Chunk(doc)
		if err != nil {
			return 0, fmt.Errorf("chunk document %q: %w", doc.ID, err)
		}
		chunks = append(chunks, cs...)
	}

	if ix.Store != nil {
		for start := 0; start < len(chunks); start += batchSize {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
			batch := chunks[start:min(start+batchSize, len(chunks))]
			texts := make([]string, len(batch))
			for i, c := range batch {
				texts[i] = c.Text
			}
			vecs, err := ix.Embedder.Embed(ctx, texts)
			if err != nil {
				return 0, fmt.Errorf("embed batch at chunk %d: %w", start, err)
			}
			if len(vecs) != len(batch) {
				return 0, fmt.Errorf("embedder returned %d vectors for %d texts", len(vecs), len(batch))
			}
			records := make([]Record, len(batch))
			for i, c := range batch {
				records[i] = Record{Chunk: c, Vector: vecs[i]}
			}
			if err := ix.Store.Upsert(ctx, records...); err != nil {
				return 0, fmt.Errorf("upsert batch at chunk %d: %w", start, err)
			}
		}
	}

	for _, s := range ix.Sinks {
		s.Add(chunks...)
	}
	return len(chunks), nil
}

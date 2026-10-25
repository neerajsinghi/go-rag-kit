package ragkit_test

import (
	"context"
	"errors"
	"testing"

	ragkit "github.com/neerajsinghi/go-rag-kit"
	"github.com/neerajsinghi/go-rag-kit/chunk"
	"github.com/neerajsinghi/go-rag-kit/embed/fake"
	"github.com/neerajsinghi/go-rag-kit/retrieve"
	"github.com/neerajsinghi/go-rag-kit/store/memory"
)

type countingEmbedder struct {
	inner ragkit.Embedder
	calls int
	short bool
}

func (c *countingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	c.calls++
	v, err := c.inner.Embed(ctx, texts)
	if c.short && len(v) > 0 {
		v = v[1:]
	}
	return v, err
}

func TestIndexer(t *testing.T) {
	ctx := context.Background()
	emb := &countingEmbedder{inner: fake.New(32)}
	store := memory.New()
	bm := retrieve.NewBM25()
	ix := &ragkit.Indexer{
		Chunker:   chunk.Fixed{Size: 10},
		Embedder:  emb,
		Store:     store,
		Sinks:     []ragkit.ChunkSink{bm},
		BatchSize: 2,
	}
	n, err := ix.Index(ctx,
		ragkit.Document{ID: "a", Text: "0123456789abcdefghij"},
		ragkit.Document{ID: "b", Text: "short"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 || store.Len() != 3 || bm.Len() != 3 {
		t.Fatalf("n=%d store=%d bm25=%d, want 3", n, store.Len(), bm.Len())
	}
	if emb.calls != 2 {
		t.Errorf("embed calls = %d, want 2 batches", emb.calls)
	}
}

func TestIndexerErrors(t *testing.T) {
	ctx := context.Background()
	doc := ragkit.Document{ID: "a", Text: "hello world"}
	tests := []struct {
		name string
		ix   ragkit.Indexer
	}{
		{"no chunker", ragkit.Indexer{}},
		{"embedder without store", ragkit.Indexer{Chunker: chunk.Fixed{Size: 4}, Embedder: fake.New(4)}},
		{"negative batch", ragkit.Indexer{Chunker: chunk.Fixed{Size: 4}, BatchSize: -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.ix.Index(ctx, doc); !errors.Is(err, ragkit.ErrInvalidArgument) {
				t.Fatalf("err = %v", err)
			}
		})
	}
	ix := ragkit.Indexer{Chunker: chunk.Fixed{Size: 4}, Embedder: &countingEmbedder{inner: fake.New(4), short: true}, Store: memory.New()}
	if _, err := ix.Index(ctx, doc); err == nil {
		t.Fatal("expected error when embedder returns too few vectors")
	}
	ix = ragkit.Indexer{Chunker: chunk.Fixed{Size: 4}}
	if _, err := ix.Index(ctx, ragkit.Document{Text: "no id"}); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatalf("chunker error not propagated: %v", err)
	}
}

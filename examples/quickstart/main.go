// Command quickstart indexes a few markdown documents and answers queries
// with hybrid (BM25 + vector, fused with RRF) retrieval.
//
// It runs fully offline with the deterministic fake embedder. Set
// OPENAI_API_KEY to use real OpenAI embeddings instead:
//
//	go run ./examples/quickstart
//	OPENAI_API_KEY=sk-... go run ./examples/quickstart
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ragkit "github.com/neerajsinghi/go-rag-kit"
	"github.com/neerajsinghi/go-rag-kit/chunk"
	"github.com/neerajsinghi/go-rag-kit/embed/fake"
	"github.com/neerajsinghi/go-rag-kit/embed/openai"
	"github.com/neerajsinghi/go-rag-kit/retrieve"
	"github.com/neerajsinghi/go-rag-kit/store/memory"
)

var docs = []ragkit.Document{
	{
		ID:       "runbook",
		Metadata: map[string]string{"path": "docs/runbook.md"},
		Text: `# Payments API runbook

## Error PAY-4021

PAY-4021 means the card issuer declined the authorisation. Do not retry automatically; surface the decline to the customer.

## High latency

If p99 latency exceeds 800ms, check the connection pool size and the database's slow query log first.
`,
	},
	{
		ID:       "onboarding",
		Metadata: map[string]string{"path": "docs/onboarding.md"},
		Text: `# Onboarding

## Local setup

Install Go, run make dev, and export DATABASE_URL to point at the local Postgres container.

## Deploys

Deploys go out through the CI pipeline after review; never deploy from a laptop.
`,
	},
}

func main() {
	ctx := context.Background()

	var emb ragkit.Embedder = fake.New(256)
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		emb = openai.New(key)
	}
	store := memory.New()
	keywords := retrieve.NewBM25()

	ix := &ragkit.Indexer{
		Chunker:  chunk.Markdown{MaxSize: 800, Overlap: 80},
		Embedder: emb,
		Store:    store,
		Sinks:    []ragkit.ChunkSink{keywords},
	}
	n, err := ix.Index(ctx, docs...)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("indexed %d chunks from %d documents\n", n, len(docs))

	hybrid := retrieve.NewHybrid(retrieve.NewVector(emb, store), keywords)
	for _, q := range []string{"what does PAY-4021 mean?", "how do I set up postgres locally", "slow p99 latency"} {
		results, err := hybrid.Retrieve(ctx, q, 2)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("\nQ: %s\n", q)
		for i, r := range results {
			fmt.Printf("  %d. [%s] %s (score %.4f)\n", i+1, r.Chunk.Metadata["path"], r.Chunk.Metadata[chunk.MetaSection], r.Score)
		}
	}
}

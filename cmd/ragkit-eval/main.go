// Command ragkit-eval indexes a JSONL corpus and reports recall@k and MRR
// for BM25, vector and hybrid (RRF) retrieval over a labelled query set.
//
// Usage:
//
//	ragkit-eval -corpus corpus.jsonl -queries queries.jsonl [-k 5] [-mode all]
//
// Corpus lines are {"id": "...", "text": "...", "metadata": {...}}; query
// lines are {"query": "...", "relevant_ids": ["doc-id", ...]}. relevant_ids
// name documents, and retrieved chunks are de-duplicated by document.
//
// The default embedder is the offline hashing embedder (no API key needed).
// Pass -embedder openai to use the OpenAI embeddings API with
// OPENAI_API_KEY (and optionally OPENAI_BASE_URL).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"

	ragkit "github.com/neerajsinghi/go-rag-kit"
	"github.com/neerajsinghi/go-rag-kit/chunk"
	"github.com/neerajsinghi/go-rag-kit/embed/fake"
	"github.com/neerajsinghi/go-rag-kit/embed/openai"
	"github.com/neerajsinghi/go-rag-kit/eval"
	"github.com/neerajsinghi/go-rag-kit/retrieve"
	"github.com/neerajsinghi/go-rag-kit/store/memory"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	os.Exit(run(ctx, os.Args[1:], os.Stdout, os.Stderr, os.Getenv))
}

type config struct {
	corpus, queries string
	k               int
	mode            string
	embedder        string
	model           string
	dims            int
	chunker         string
	chunkSize       int
	overlap         int
	jsonOut         bool
	verbose         bool
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	var cfg config
	fs := flag.NewFlagSet("ragkit-eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.StringVar(&cfg.corpus, "corpus", "", "JSONL corpus of documents (required)")
	fs.StringVar(&cfg.queries, "queries", "", "JSONL query set with relevant_ids (required)")
	fs.IntVar(&cfg.k, "k", 5, "cut-off for recall@k and MRR@k")
	fs.StringVar(&cfg.mode, "mode", "all", "retrieval mode: all, bm25, vector or hybrid")
	fs.StringVar(&cfg.embedder, "embedder", "fake", "embedder: fake (offline hashing) or openai")
	fs.StringVar(&cfg.model, "model", openai.DefaultModel, "OpenAI embedding model")
	fs.IntVar(&cfg.dims, "dims", 0, "embedding size; 0 uses the embedder default (fake: 256, openai: model default)")
	fs.StringVar(&cfg.chunker, "chunker", "markdown", "chunker: markdown or fixed")
	fs.IntVar(&cfg.chunkSize, "chunk-size", 1000, "maximum chunk size in runes")
	fs.IntVar(&cfg.overlap, "overlap", 100, "chunk overlap in runes")
	fs.BoolVar(&cfg.jsonOut, "json", false, "print reports as JSON")
	fs.BoolVar(&cfg.verbose, "v", false, "print per-query results")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if cfg.corpus == "" || cfg.queries == "" {
		fmt.Fprintln(stderr, "ragkit-eval: -corpus and -queries are required")
		fs.Usage()
		return 2
	}
	if err := evaluate(ctx, cfg, stdout, getenv); err != nil {
		fmt.Fprintln(stderr, "ragkit-eval:", err)
		return 1
	}
	return 0
}

func evaluate(ctx context.Context, cfg config, stdout io.Writer, getenv func(string) string) error {
	docs, err := loadFile(cfg.corpus, eval.LoadDocuments)
	if err != nil {
		return fmt.Errorf("corpus: %w", err)
	}
	cases, err := loadFile(cfg.queries, eval.LoadCases)
	if err != nil {
		return fmt.Errorf("queries: %w", err)
	}

	var chunker ragkit.Chunker
	switch cfg.chunker {
	case "markdown":
		chunker = chunk.Markdown{MaxSize: cfg.chunkSize, Overlap: cfg.overlap}
	case "fixed":
		chunker = chunk.Fixed{Size: cfg.chunkSize, Overlap: cfg.overlap}
	default:
		return fmt.Errorf("unknown chunker %q", cfg.chunker)
	}

	modes := []string{cfg.mode}
	if cfg.mode == "all" {
		modes = []string{"bm25", "vector", "hybrid"}
	}
	needVectors := false
	for _, m := range modes {
		switch m {
		case "bm25":
		case "vector", "hybrid":
			needVectors = true
		default:
			return fmt.Errorf("unknown mode %q", m)
		}
	}

	bm := retrieve.NewBM25()
	ix := &ragkit.Indexer{Chunker: chunker, Sinks: []ragkit.ChunkSink{bm}}
	var vec *retrieve.Vector
	if needVectors {
		emb, err := newEmbedder(cfg, getenv)
		if err != nil {
			return err
		}
		store := memory.New()
		ix.Embedder, ix.Store = emb, store
		vec = retrieve.NewVector(emb, store)
	}
	n, err := ix.Index(ctx, docs...)
	if err != nil {
		return fmt.Errorf("index: %w", err)
	}

	retrievers := map[string]ragkit.Retriever{"bm25": bm}
	if vec != nil {
		retrievers["vector"] = vec
		retrievers["hybrid"] = retrieve.NewHybrid(vec, bm)
	}

	type namedReport struct {
		Mode string `json:"mode"`
		eval.Report
	}
	var reports []namedReport
	for _, m := range modes {
		rep, err := eval.Evaluate(ctx, retrievers[m], cases, eval.Options{K: cfg.k, ByDocument: true})
		if err != nil {
			return fmt.Errorf("%s: %w", m, err)
		}
		if !cfg.verbose {
			rep.Results = nil
		}
		reports = append(reports, namedReport{m, rep})
	}

	if cfg.jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(reports)
	}
	fmt.Fprintf(stdout, "corpus: %d documents, %d chunks; queries: %d; embedder: %s\n\n", len(docs), n, len(cases), embedderName(cfg, needVectors))
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "MODE\tRECALL@%d\tMRR@%d\n", cfg.k, cfg.k)
	for _, r := range reports {
		fmt.Fprintf(tw, "%s\t%.3f\t%.3f\n", r.Mode, r.Recall, r.MRR)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if cfg.verbose {
		for _, r := range reports {
			fmt.Fprintf(stdout, "\n[%s]\n", r.Mode)
			for _, q := range r.Results {
				fmt.Fprintf(stdout, "  recall=%.2f rr=%.2f  %q -> [%s] (want %s)\n", q.Recall, q.RR, q.Query, strings.Join(q.Retrieved, ", "), strings.Join(q.Relevant, ", "))
			}
		}
	}
	return nil
}

func newEmbedder(cfg config, getenv func(string) string) (ragkit.Embedder, error) {
	switch cfg.embedder {
	case "fake":
		if cfg.dims < 0 {
			return nil, fmt.Errorf("invalid -dims %d", cfg.dims)
		}
		return fake.New(cfg.dims), nil
	case "openai":
		key := getenv("OPENAI_API_KEY")
		if key == "" {
			return nil, errors.New("OPENAI_API_KEY is not set")
		}
		opts := []openai.Option{openai.WithModel(cfg.model)}
		if cfg.dims > 0 {
			opts = append(opts, openai.WithDimensions(cfg.dims))
		}
		if u := getenv("OPENAI_BASE_URL"); u != "" {
			opts = append(opts, openai.WithBaseURL(u))
		}
		return openai.New(key, opts...), nil
	}
	return nil, fmt.Errorf("unknown embedder %q", cfg.embedder)
}

func embedderName(cfg config, used bool) string {
	switch {
	case !used:
		return "none"
	case cfg.embedder == "openai":
		return "openai/" + cfg.model
	case cfg.dims > 0:
		return fmt.Sprintf("fake (%d dims)", cfg.dims)
	default:
		return fmt.Sprintf("fake (%d dims)", fake.DefaultDims)
	}
}

func loadFile[T any](path string, load func(io.Reader) ([]T, error)) ([]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return load(f)
}

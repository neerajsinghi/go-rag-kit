package retrieve

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/neerajsinghi/go-rag-kit/embed/fake"
	"github.com/neerajsinghi/go-rag-kit/store/memory"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

func chunk(id, text string) ragkit.Chunk {
	return ragkit.Chunk{ID: id, DocID: id, Text: text}
}

func ids(rs []ragkit.Result) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.Chunk.ID
	}
	return out
}

func list(ids ...string) []ragkit.Result {
	out := make([]ragkit.Result, len(ids))
	for i, id := range ids {
		out[i] = ragkit.Result{Chunk: chunk(id, ""), Score: float64(len(ids) - i)}
	}
	return out
}

var corpus = []ragkit.Chunk{
	chunk("goroutines", "Goroutines are lightweight threads managed by the Go runtime."),
	chunk("channels", "Channels let goroutines communicate. Send and receive on channels to synchronise goroutines."),
	chunk("pgvector", "pgvector adds a vector type and HNSW indexes to Postgres."),
	chunk("bread", "Sourdough bread needs flour, water, salt and patience."),
	chunk("errcode", "Error ERR_CONN_RESET means the peer reset the TCP connection."),
}

func TestBM25Ranking(t *testing.T) {
	idx := NewBM25()
	idx.Add(corpus...)
	ctx := context.Background()
	tests := []struct {
		query string
		top   string
	}{
		{"goroutines channels", "channels"},
		{"postgres vector index", "pgvector"},
		{"how do I bake bread", "bread"},
		{"ERR_CONN_RESET", "errcode"},
		{"Go runtime threads", "goroutines"},
	}
	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			got, err := idx.Retrieve(ctx, tt.query, 3)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) == 0 || got[0].Chunk.ID != tt.top {
				t.Fatalf("top = %v, want %q", ids(got), tt.top)
			}
			for i := 1; i < len(got); i++ {
				if got[i].Score > got[i-1].Score {
					t.Fatalf("not sorted: %v", got)
				}
			}
		})
	}
}

func TestBM25Properties(t *testing.T) {
	ctx := context.Background()

	t.Run("rare term outweighs common term", func(t *testing.T) {
		idx := NewBM25()
		idx.Add(chunk("a", "common common rare"), chunk("b", "common common common"), chunk("c", "common filler"))
		got, _ := idx.Retrieve(ctx, "common rare", 3)
		if got[0].Chunk.ID != "a" {
			t.Fatalf("got %v", ids(got))
		}
	})

	t.Run("shorter document wins on equal tf", func(t *testing.T) {
		idx := NewBM25()
		idx.Add(chunk("short", "kafka"), chunk("long", "kafka plus many other unrelated words here"), chunk("other", "nothing"))
		got, _ := idx.Retrieve(ctx, "kafka", 2)
		if fmt.Sprint(ids(got)) != "[short long]" {
			t.Fatalf("got %v", ids(got))
		}
	})

	t.Run("tf saturates", func(t *testing.T) {
		idx := NewBM25()
		idx.Add(chunk("x1", "go"), chunk("x5", "go go go go go"), chunk("x", "none"))
		got, _ := idx.Retrieve(ctx, "go", 2)
		ratio := got[0].Score / got[1].Score
		if ratio >= 5 || ratio <= 1 {
			t.Fatalf("score ratio %f should be sub-linear", ratio)
		}
	})

	t.Run("hand computed score", func(t *testing.T) {
		idx := NewBM25(WithK1(1.2), WithB(0))
		idx.Add(chunk("a", "apple"), chunk("b", "banana"))
		got, _ := idx.Retrieve(ctx, "apple", 1)
		// b=0: score = idf * tf*(k1+1)/(tf+k1) = idf * 2.2/2.2 = idf
		want := math.Log(1 + (2-1+0.5)/(1+0.5))
		if math.Abs(got[0].Score-want) > 1e-12 {
			t.Fatalf("score = %v, want %v", got[0].Score, want)
		}
	})

	t.Run("no matches and empty queries", func(t *testing.T) {
		idx := NewBM25()
		idx.Add(corpus...)
		for _, q := range []string{"", "   ", "the a of", "zzzz"} {
			got, err := idx.Retrieve(ctx, q, 5)
			if err != nil || len(got) != 0 {
				t.Errorf("query %q: got %v, %v", q, ids(got), err)
			}
		}
		if got, _ := NewBM25().Retrieve(ctx, "anything", 3); len(got) != 0 {
			t.Error("empty index should return nothing")
		}
	})

	t.Run("upsert and remove", func(t *testing.T) {
		idx := NewBM25()
		idx.Add(chunk("a", "alpha"), chunk("b", "beta"))
		idx.Add(chunk("a", "gamma"))
		if got, _ := idx.Retrieve(ctx, "alpha", 5); len(got) != 0 {
			t.Fatalf("stale postings after upsert: %v", ids(got))
		}
		if got, _ := idx.Retrieve(ctx, "gamma", 5); len(got) != 1 {
			t.Fatalf("upserted chunk not found: %v", ids(got))
		}
		idx.Remove("a", "missing")
		if idx.Len() != 1 {
			t.Fatalf("Len = %d", idx.Len())
		}
		if got, _ := idx.Retrieve(ctx, "gamma", 5); len(got) != 0 {
			t.Fatal("removed chunk still retrievable")
		}
	})

	t.Run("invalid k", func(t *testing.T) {
		if _, err := NewBM25().Retrieve(ctx, "x", 0); !errors.Is(err, ragkit.ErrInvalidArgument) {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestRRF(t *testing.T) {
	tests := []struct {
		name  string
		lists [][]ragkit.Result
		want  []string
	}{
		{"single list keeps order", [][]ragkit.Result{list("a", "b", "c")}, []string{"a", "b", "c"}},
		{"agreement wins", [][]ragkit.Result{list("a", "b", "c"), list("b", "a", "d")}, []string{"a", "b", "c", "d"}},
		{"consensus beats single top", [][]ragkit.Result{list("x", "b"), list("y", "b"), list("z", "b")}, []string{"b", "x", "y", "z"}},
		{"duplicates within list count once", [][]ragkit.Result{list("a", "a", "a"), list("b")}, []string{"a", "b"}},
		{"empty lists", [][]ragkit.Result{nil, {}}, []string{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RRF(0, tt.lists...)
			if fmt.Sprint(ids(got)) != fmt.Sprint(tt.want) {
				t.Fatalf("got %v, want %v", ids(got), tt.want)
			}
		})
	}

	t.Run("exact scores", func(t *testing.T) {
		got := RRF(60, list("a", "b"), list("b"))
		want := map[string]float64{"a": 1.0 / 61, "b": 1.0/62 + 1.0/61}
		for _, r := range got {
			if math.Abs(r.Score-want[r.Chunk.ID]) > 1e-12 {
				t.Errorf("%s score = %v, want %v", r.Chunk.ID, r.Score, want[r.Chunk.ID])
			}
		}
		if got[0].Chunk.ID != "b" {
			t.Errorf("order = %v", ids(got))
		}
	})

	t.Run("weights", func(t *testing.T) {
		got, err := WeightedRRF(60, []float64{1, 3}, list("a"), list("b"))
		if err != nil || got[0].Chunk.ID != "b" {
			t.Fatalf("got %v, %v", ids(got), err)
		}
		if _, err := WeightedRRF(60, []float64{1}, list("a"), list("b")); !errors.Is(err, ragkit.ErrInvalidArgument) {
			t.Fatalf("mismatched weights err = %v", err)
		}
		if _, err := WeightedRRF(60, []float64{-1}, list("a")); !errors.Is(err, ragkit.ErrInvalidArgument) {
			t.Fatalf("negative weight err = %v", err)
		}
	})
}

type staticRetriever struct {
	results []ragkit.Result
	err     error
	gotK    int
}

func (s *staticRetriever) Retrieve(ctx context.Context, _ string, k int) ([]ragkit.Result, error) {
	s.gotK = k
	if s.err != nil {
		return nil, s.err
	}
	return s.results, nil
}

func TestHybrid(t *testing.T) {
	ctx := context.Background()
	a := &staticRetriever{results: list("a", "b", "c")}
	b := &staticRetriever{results: list("b", "c", "d")}
	got, err := NewHybrid(a, b).Retrieve(ctx, "q", 2)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(ids(got)) != "[b c]" {
		t.Fatalf("got %v", ids(got))
	}
	if a.gotK != 2*DefaultCandidateFactor {
		t.Errorf("candidates = %d", a.gotK)
	}

	boom := errors.New("boom")
	if _, err := NewHybrid(a, &staticRetriever{err: boom}).Retrieve(ctx, "q", 2); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
	for name, h := range map[string]*Hybrid{
		"no retrievers": {},
		"nil retriever": {Retrievers: []ragkit.Retriever{nil}},
		"negative cand": {Retrievers: []ragkit.Retriever{a}, Candidates: -1},
	} {
		if _, err := h.Retrieve(ctx, "q", 1); !errors.Is(err, ragkit.ErrInvalidArgument) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

func TestHybridEndToEnd(t *testing.T) {
	ctx := context.Background()
	emb := fake.New(128)
	store := memory.New()
	bm := NewBM25()
	ix := &ragkit.Indexer{Chunker: identityChunker{}, Embedder: emb, Store: store, Sinks: []ragkit.ChunkSink{bm}}
	var docs []ragkit.Document
	for _, c := range corpus {
		docs = append(docs, ragkit.Document{ID: c.ID, Text: c.Text})
	}
	if _, err := ix.Index(ctx, docs...); err != nil {
		t.Fatal(err)
	}
	h := NewHybrid(NewVector(emb, store), bm)
	got, err := h.Retrieve(ctx, "postgres HNSW vector index", 3)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Chunk.ID != "pgvector" {
		t.Fatalf("top = %v", ids(got))
	}
}

func TestVectorInvalid(t *testing.T) {
	if _, err := (&Vector{}).Retrieve(context.Background(), "q", 1); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatalf("err = %v", err)
	}
	v := NewVector(fake.New(8), memory.New())
	if _, err := v.Retrieve(context.Background(), "q", 0); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatalf("err = %v", err)
	}
}

// identityChunker turns each document into a single chunk with the doc's ID.
type identityChunker struct{}

func (identityChunker) Chunk(d ragkit.Document) ([]ragkit.Chunk, error) {
	return []ragkit.Chunk{{ID: d.ID, DocID: d.ID, Text: d.Text, End: len(d.Text)}}, nil
}

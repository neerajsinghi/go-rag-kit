package eval

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

func TestRecallAtK(t *testing.T) {
	tests := []struct {
		name      string
		retrieved []string
		relevant  []string
		k         int
		want      float64
	}{
		{"all found", []string{"a", "b"}, []string{"a", "b"}, 2, 1},
		{"half found", []string{"a", "x", "y"}, []string{"a", "b"}, 3, 0.5},
		{"cut off by k", []string{"x", "y", "a"}, []string{"a"}, 2, 0},
		{"k beyond results", []string{"a"}, []string{"a", "b"}, 10, 0.5},
		{"duplicate retrieved counted once", []string{"a", "a"}, []string{"a", "b"}, 2, 0.5},
		{"duplicate relevant", []string{"a"}, []string{"a", "a"}, 1, 1},
		{"no relevant", []string{"a"}, nil, 1, 0},
		{"zero k", []string{"a"}, []string{"a"}, 0, 0},
		{"empty retrieved", nil, []string{"a"}, 3, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RecallAtK(tt.retrieved, tt.relevant, tt.k); math.Abs(got-tt.want) > 1e-12 {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestReciprocalRank(t *testing.T) {
	tests := []struct {
		name      string
		retrieved []string
		relevant  []string
		k         int
		want      float64
	}{
		{"first", []string{"a", "b"}, []string{"a"}, 5, 1},
		{"third", []string{"x", "y", "a"}, []string{"a"}, 5, 1.0 / 3},
		{"first relevant counts", []string{"x", "b", "a"}, []string{"a", "b"}, 5, 0.5},
		{"beyond k", []string{"x", "y", "a"}, []string{"a"}, 2, 0},
		{"none", []string{"x"}, []string{"a"}, 5, 0},
		{"negative k", []string{"a"}, []string{"a"}, -1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReciprocalRank(tt.retrieved, tt.relevant, tt.k); math.Abs(got-tt.want) > 1e-12 {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLoadCases(t *testing.T) {
	in := `{"query":"q1","relevant_ids":["a"]}

{"query":"q2","relevant_ids":["b","c"]}
`
	cases, err := LoadCases(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) != 2 || cases[1].RelevantIDs[1] != "c" {
		t.Fatalf("cases = %+v", cases)
	}
	bad := []struct{ name, in string }{
		{"malformed json", `{"query":`},
		{"empty query", `{"query":" ","relevant_ids":["a"]}`},
		{"no relevant", `{"query":"q"}`},
	}
	for _, b := range bad {
		t.Run(b.name, func(t *testing.T) {
			_, err := LoadCases(strings.NewReader("{\"query\":\"ok\",\"relevant_ids\":[\"a\"]}\n" + b.in))
			if err == nil || !strings.Contains(err.Error(), "line 2") {
				t.Fatalf("err = %v, want line 2 error", err)
			}
		})
	}
}

func TestLoadDocuments(t *testing.T) {
	docs, err := LoadDocuments(strings.NewReader(`{"id":"a","text":"x","metadata":{"k":"v"}}`))
	if err != nil || len(docs) != 1 || docs[0].Metadata["k"] != "v" {
		t.Fatalf("docs=%+v err=%v", docs, err)
	}
	if _, err := LoadDocuments(strings.NewReader(`{"text":"no id"}`)); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Fatalf("err = %v", err)
	}
}

// mapRetriever returns canned results per query.
type mapRetriever struct {
	results map[string][]ragkit.Result
	gotK    int
}

func (m *mapRetriever) Retrieve(_ context.Context, q string, k int) ([]ragkit.Result, error) {
	m.gotK = k
	if q == "fail" {
		return nil, errors.New("boom")
	}
	rs := m.results[q]
	return rs[:min(k, len(rs))], nil
}

func res(chunkID, docID string) ragkit.Result {
	return ragkit.Result{Chunk: ragkit.Chunk{ID: chunkID, DocID: docID}}
}

func TestEvaluate(t *testing.T) {
	r := &mapRetriever{results: map[string][]ragkit.Result{
		"q1": {res("a#0", "a"), res("a#1", "a"), res("b#0", "b")},
		"q2": {res("c#0", "c"), res("d#0", "d")},
	}}
	cases := []Case{
		{Query: "q1", RelevantIDs: []string{"b"}},
		{Query: "q2", RelevantIDs: []string{"x"}},
	}
	ctx := context.Background()

	rep, err := Evaluate(ctx, r, cases, Options{K: 2, ByDocument: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.gotK != 8 {
		t.Errorf("overfetch: retriever got k=%d, want 8", r.gotK)
	}
	// q1 docs dedupe to [a b]: recall 1, RR 0.5. q2: 0, 0.
	if rep.Queries != 2 || rep.Recall != 0.5 || rep.MRR != 0.25 {
		t.Fatalf("report = %+v", rep)
	}
	if got := rep.Results[0].Retrieved; len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("retrieved = %v", got)
	}

	// By chunk ID, b#0 is at rank 3, outside k=2.
	rep, err = Evaluate(ctx, r, []Case{{Query: "q1", RelevantIDs: []string{"b#0"}}}, Options{K: 2})
	if err != nil || rep.Recall != 0 || r.gotK != 2 {
		t.Fatalf("rep=%+v err=%v k=%d", rep, err, r.gotK)
	}

	if _, err := Evaluate(ctx, r, []Case{{Query: "fail", RelevantIDs: []string{"a"}}}, Options{K: 1}); err == nil {
		t.Fatal("expected retriever error")
	}
	for _, o := range []Options{{K: 0}, {K: 1, Overfetch: -1}} {
		if _, err := Evaluate(ctx, r, cases, o); !errors.Is(err, ragkit.ErrInvalidArgument) {
			t.Errorf("%+v: err = %v", o, err)
		}
	}
	if _, err := Evaluate(ctx, nil, cases, Options{K: 1}); !errors.Is(err, ragkit.ErrInvalidArgument) {
		t.Errorf("nil retriever err = %v", err)
	}
	if rep, err := Evaluate(ctx, r, nil, Options{K: 1}); err != nil || rep.Recall != 0 {
		t.Errorf("empty cases: %+v %v", rep, err)
	}
}

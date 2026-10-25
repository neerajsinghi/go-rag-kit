package fake

import (
	"context"
	"math"
	"testing"
)

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot // vectors are unit length
}

func TestEmbedderDeterministicAndNormalised(t *testing.T) {
	e := New(64)
	texts := []string{"golang concurrency", "golang concurrency", "", "   ", "the a an"}
	vecs, err := e.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != len(texts) {
		t.Fatalf("got %d vectors", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 64 {
			t.Fatalf("vector %d has %d dims", i, len(v))
		}
		var n float64
		for _, x := range v {
			n += float64(x) * float64(x)
		}
		if math.Abs(n-1) > 1e-5 {
			t.Errorf("vector %d has squared norm %f", i, n)
		}
	}
	if cosine(vecs[0], vecs[1]) < 0.9999 {
		t.Error("identical texts should embed identically")
	}
}

func TestEmbedderSimilarity(t *testing.T) {
	e := &Embedder{}
	q := e.Vector("postgres vector index")
	near := e.Vector("building a vector index in postgres with pgvector")
	far := e.Vector("baking sourdough bread at home")
	if cosine(q, near) <= cosine(q, far) {
		t.Fatalf("expected overlapping text to be closer: near=%f far=%f", cosine(q, near), cosine(q, far))
	}
}

func TestEmbedderCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := New(8).Embed(ctx, []string{"x"}); err == nil {
		t.Fatal("expected context error")
	}
}

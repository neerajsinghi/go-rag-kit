package retrieve

import (
	"context"
	"errors"
	"fmt"
	"sync"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// DefaultCandidateFactor controls how many candidates each underlying
// retriever returns when Hybrid.Candidates is zero: CandidateFactor * k.
const DefaultCandidateFactor = 4

// Hybrid runs several retrievers concurrently (typically a Vector and a
// BM25) and fuses their rankings with Reciprocal Rank Fusion.
//
// Combining dense and keyword retrieval covers each other's blind spots:
// embeddings match paraphrases, BM25 matches exact identifiers, error codes
// and rare names that embeddings tend to blur.
type Hybrid struct {
	// Retrievers are queried in parallel. At least one is required.
	Retrievers []ragkit.Retriever
	// Weights optionally weights each retriever's contribution; nil means
	// equal weights.
	Weights []float64
	// K is the RRF rank constant; zero means DefaultRRFK.
	K float64
	// Candidates is how many results to request from each retriever before
	// fusion. Zero means DefaultCandidateFactor * k.
	Candidates int
}

var _ ragkit.Retriever = (*Hybrid)(nil)

// NewHybrid returns a Hybrid with equal weights and default parameters.
func NewHybrid(retrievers ...ragkit.Retriever) *Hybrid {
	return &Hybrid{Retrievers: retrievers}
}

// Retrieve implements ragkit.Retriever. If any underlying retriever fails,
// the first error is returned. Result scores are RRF scores.
func (h *Hybrid) Retrieve(ctx context.Context, query string, k int) ([]ragkit.Result, error) {
	if len(h.Retrievers) == 0 {
		return nil, fmt.Errorf("%w: hybrid retriever has no retrievers", ragkit.ErrInvalidArgument)
	}
	if k <= 0 {
		return nil, fmt.Errorf("%w: k must be positive, got %d", ragkit.ErrInvalidArgument, k)
	}
	if h.Candidates < 0 {
		return nil, fmt.Errorf("%w: negative candidate count %d", ragkit.ErrInvalidArgument, h.Candidates)
	}
	candidates := h.Candidates
	if candidates == 0 {
		candidates = DefaultCandidateFactor * k
	}
	candidates = max(candidates, k)

	for i, r := range h.Retrievers {
		if r == nil {
			return nil, fmt.Errorf("%w: hybrid retriever %d is nil", ragkit.ErrInvalidArgument, i)
		}
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	lists := make([][]ragkit.Result, len(h.Retrievers))
	errs := make([]error, len(h.Retrievers))
	var wg sync.WaitGroup
	for i, r := range h.Retrievers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			lists[i], errs[i] = r.Retrieve(ctx, query, candidates)
			if errs[i] != nil {
				cancel()
			}
		}()
	}
	wg.Wait()
	if err := firstError(errs); err != nil {
		return nil, err
	}

	fused, err := WeightedRRF(h.K, h.Weights, lists...)
	if err != nil {
		return nil, err
	}
	if len(fused) > k {
		fused = fused[:k]
	}
	return fused, nil
}

// firstError returns the first root-cause error, preferring errors other than
// context.Canceled, which siblings report after one retriever fails.
func firstError(errs []error) error {
	var fallback error
	for i, err := range errs {
		if err == nil {
			continue
		}
		wrapped := fmt.Errorf("hybrid retriever %d: %w", i, err)
		if !errors.Is(err, context.Canceled) {
			return wrapped
		}
		if fallback == nil {
			fallback = wrapped
		}
	}
	return fallback
}

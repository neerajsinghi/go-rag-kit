package retrieve

import (
	"fmt"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// DefaultRRFK is the rank constant from Cormack, Clarke & Büttcher (2009),
// "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
// Methods".
const DefaultRRFK = 60

// RRF fuses ranked result lists with Reciprocal Rank Fusion:
//
//	score(d) = Σ_lists 1 / (k + rank_list(d))
//
// where rank is 1-based. Each list must already be ordered best-first; its
// own scores are ignored, which is what makes RRF robust when fusing
// incomparable scales such as cosine similarity and BM25. A chunk appearing
// more than once in a list counts only at its best rank. k <= 0 selects
// DefaultRRFK. The output is ordered by fused score with ties broken by
// chunk ID.
func RRF(k float64, lists ...[]ragkit.Result) []ragkit.Result {
	out, _ := WeightedRRF(k, nil, lists...)
	return out
}

// WeightedRRF is RRF with a per-list weight multiplying each list's
// contribution. A nil weights slice means every weight is 1; otherwise it
// must have one non-negative weight per list.
func WeightedRRF(k float64, weights []float64, lists ...[]ragkit.Result) ([]ragkit.Result, error) {
	if weights != nil && len(weights) != len(lists) {
		return nil, fmt.Errorf("%w: %d weights for %d lists", ragkit.ErrInvalidArgument, len(weights), len(lists))
	}
	for _, w := range weights {
		if w < 0 {
			return nil, fmt.Errorf("%w: negative RRF weight %v", ragkit.ErrInvalidArgument, w)
		}
	}
	if k <= 0 {
		k = DefaultRRFK
	}
	fused := make(map[string]*ragkit.Result)
	var order []string
	for li, list := range lists {
		w := 1.0
		if weights != nil {
			w = weights[li]
		}
		seen := make(map[string]struct{}, len(list))
		for rank, r := range list {
			id := r.Chunk.ID
			if _, dup := seen[id]; dup {
				continue
			}
			seen[id] = struct{}{}
			contrib := w / (k + float64(rank+1))
			if f, ok := fused[id]; ok {
				f.Score += contrib
				continue
			}
			fused[id] = &ragkit.Result{Chunk: r.Chunk, Score: contrib}
			order = append(order, id)
		}
	}
	out := make([]ragkit.Result, 0, len(order))
	for _, id := range order {
		out = append(out, *fused[id])
	}
	sortResults(out)
	return out, nil
}

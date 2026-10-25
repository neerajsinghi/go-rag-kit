// Package eval measures retrieval quality with recall@k and mean reciprocal
// rank (MRR) over a labelled query set.
//
// A query set is JSONL, one case per line:
//
//	{"query": "how do I create an HNSW index?", "relevant_ids": ["pgvector-guide"]}
//
// relevant_ids are document IDs by default (Options.ByDocument), or chunk
// IDs otherwise.
package eval

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// Case is one labelled query.
type Case struct {
	Query       string   `json:"query"`
	RelevantIDs []string `json:"relevant_ids"`
}

// maxLine bounds a single JSONL line (documents can be large).
const maxLine = 16 << 20

// LoadCases reads JSONL query cases from r. Blank lines are skipped. Every
// case needs a non-empty query and at least one relevant ID.
func LoadCases(r io.Reader) ([]Case, error) {
	var cases []Case
	err := readJSONL(r, func(raw []byte) error {
		var c Case
		if err := json.Unmarshal(raw, &c); err != nil {
			return err
		}
		if strings.TrimSpace(c.Query) == "" {
			return fmt.Errorf("%w: empty query", ragkit.ErrInvalidArgument)
		}
		if len(c.RelevantIDs) == 0 {
			return fmt.Errorf("%w: no relevant_ids", ragkit.ErrInvalidArgument)
		}
		cases = append(cases, c)
		return nil
	})
	return cases, err
}

// LoadDocuments reads JSONL documents ({"id", "text", "metadata"}) from r.
// Blank lines are skipped; every document needs an ID.
func LoadDocuments(r io.Reader) ([]ragkit.Document, error) {
	var docs []ragkit.Document
	err := readJSONL(r, func(raw []byte) error {
		var d ragkit.Document
		if err := json.Unmarshal(raw, &d); err != nil {
			return err
		}
		if d.ID == "" {
			return fmt.Errorf("%w: document without id", ragkit.ErrInvalidArgument)
		}
		docs = append(docs, d)
		return nil
	})
	return docs, err
}

func readJSONL(r io.Reader, fn func(raw []byte) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), maxLine)
	for line := 1; sc.Scan(); line++ {
		raw := []byte(strings.TrimSpace(sc.Text()))
		if len(raw) == 0 {
			continue
		}
		if err := fn(raw); err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("read jsonl: %w", err)
	}
	return nil
}

// RecallAtK is the fraction of distinct relevant IDs found among the first k
// retrieved IDs. It returns 0 when relevant is empty or k <= 0.
func RecallAtK(retrieved, relevant []string, k int) float64 {
	rel := toSet(relevant)
	if len(rel) == 0 || k <= 0 {
		return 0
	}
	found := make(map[string]struct{})
	for _, id := range retrieved[:min(k, len(retrieved))] {
		if _, ok := rel[id]; ok {
			found[id] = struct{}{}
		}
	}
	return float64(len(found)) / float64(len(rel))
}

// ReciprocalRank is 1/rank of the first relevant ID within the first k
// retrieved IDs (rank is 1-based), or 0 if none is found. Averaged over
// queries it gives MRR@k.
func ReciprocalRank(retrieved, relevant []string, k int) float64 {
	rel := toSet(relevant)
	for i, id := range retrieved[:min(max(k, 0), len(retrieved))] {
		if _, ok := rel[id]; ok {
			return 1 / float64(i+1)
		}
	}
	return 0
}

func toSet(ids []string) map[string]struct{} {
	s := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		s[id] = struct{}{}
	}
	return s
}

// Options configures Evaluate.
type Options struct {
	// K is the cut-off for recall@k and MRR@k. Required.
	K int
	// ByDocument maps retrieved chunks to their DocID and de-duplicates them
	// (keeping the best rank), so relevant_ids name documents. Otherwise
	// relevant_ids name chunk IDs.
	ByDocument bool
	// Overfetch multiplies K when ByDocument is set, so that K distinct
	// documents remain after de-duplication. Zero means 4.
	Overfetch int
}

// QueryResult is the per-query outcome.
type QueryResult struct {
	Query     string   `json:"query"`
	Relevant  []string `json:"relevant_ids"`
	Retrieved []string `json:"retrieved_ids"`
	Recall    float64  `json:"recall"`
	RR        float64  `json:"reciprocal_rank"`
}

// Report aggregates results over all cases.
type Report struct {
	K       int           `json:"k"`
	Queries int           `json:"queries"`
	Recall  float64       `json:"recall_at_k"`
	MRR     float64       `json:"mrr"`
	Results []QueryResult `json:"results"`
}

// Evaluate runs every case against r and aggregates recall@k and MRR@k.
func Evaluate(ctx context.Context, r ragkit.Retriever, cases []Case, opts Options) (Report, error) {
	if r == nil {
		return Report{}, fmt.Errorf("%w: nil retriever", ragkit.ErrInvalidArgument)
	}
	if opts.K <= 0 {
		return Report{}, fmt.Errorf("%w: K must be positive, got %d", ragkit.ErrInvalidArgument, opts.K)
	}
	if opts.Overfetch < 0 {
		return Report{}, fmt.Errorf("%w: negative overfetch %d", ragkit.ErrInvalidArgument, opts.Overfetch)
	}
	fetch := opts.K
	if opts.ByDocument {
		of := opts.Overfetch
		if of == 0 {
			of = 4
		}
		fetch = opts.K * of
	}

	rep := Report{K: opts.K, Queries: len(cases), Results: make([]QueryResult, 0, len(cases))}
	for i, c := range cases {
		results, err := r.Retrieve(ctx, c.Query, fetch)
		if err != nil {
			return Report{}, fmt.Errorf("case %d (%q): %w", i, c.Query, err)
		}
		ids := make([]string, 0, len(results))
		seen := make(map[string]struct{}, len(results))
		for _, res := range results {
			id := res.Chunk.ID
			if opts.ByDocument {
				id = res.Chunk.DocID
			}
			if _, dup := seen[id]; dup {
				continue
			}
			seen[id] = struct{}{}
			ids = append(ids, id)
		}
		ids = ids[:min(opts.K, len(ids))]
		qr := QueryResult{
			Query:     c.Query,
			Relevant:  c.RelevantIDs,
			Retrieved: ids,
			Recall:    RecallAtK(ids, c.RelevantIDs, opts.K),
			RR:        ReciprocalRank(ids, c.RelevantIDs, opts.K),
		}
		rep.Recall += qr.Recall
		rep.MRR += qr.RR
		rep.Results = append(rep.Results, qr)
	}
	if n := len(cases); n > 0 {
		rep.Recall /= float64(n)
		rep.MRR /= float64(n)
	}
	return rep, nil
}

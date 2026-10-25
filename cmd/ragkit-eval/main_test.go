package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	corpus  = "../../examples/eval/corpus.jsonl"
	queries = "../../examples/eval/queries.jsonl"
)

func noEnv(string) string { return "" }

func TestRunTable(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(context.Background(), []string{"-corpus", corpus, "-queries", queries, "-k", "3"}, &out, &errOut, noEnv)
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	for _, want := range []string{"RECALL@3", "bm25", "vector", "hybrid", "12 documents"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(context.Background(), []string{"-corpus", corpus, "-queries", queries, "-mode", "hybrid", "-json", "-v"}, &out, &errOut, noEnv)
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	var reports []struct {
		Mode    string            `json:"mode"`
		Recall  float64           `json:"recall_at_k"`
		MRR     float64           `json:"mrr"`
		Results []json.RawMessage `json:"results"`
	}
	if err := json.Unmarshal(out.Bytes(), &reports); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if len(reports) != 1 || reports[0].Mode != "hybrid" || len(reports[0].Results) != 12 {
		t.Fatalf("reports = %+v", reports)
	}
	if reports[0].Recall < 0.8 || reports[0].MRR < 0.8 {
		t.Errorf("hybrid quality regressed: recall=%.3f mrr=%.3f", reports[0].Recall, reports[0].MRR)
	}
}

func TestRunErrors(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.jsonl")
	if err := os.WriteFile(bad, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		args []string
		code int
	}{
		{"missing flags", nil, 2},
		{"unknown flag", []string{"-nope"}, 2},
		{"missing file", []string{"-corpus", filepath.Join(dir, "x"), "-queries", queries}, 1},
		{"bad corpus", []string{"-corpus", bad, "-queries", queries}, 1},
		{"bad mode", []string{"-corpus", corpus, "-queries", queries, "-mode", "magic"}, 1},
		{"bad chunker", []string{"-corpus", corpus, "-queries", queries, "-chunker", "magic"}, 1},
		{"openai without key", []string{"-corpus", corpus, "-queries", queries, "-embedder", "openai"}, 1},
		{"bad overlap", []string{"-corpus", corpus, "-queries", queries, "-chunk-size", "10", "-overlap", "10"}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, errOut bytes.Buffer
			if code := run(context.Background(), tt.args, &out, &errOut, noEnv); code != tt.code {
				t.Fatalf("exit %d, want %d; stderr: %s", code, tt.code, errOut.String())
			}
		})
	}
}

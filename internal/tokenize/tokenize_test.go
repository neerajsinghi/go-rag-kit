package tokenize

import (
	"fmt"
	"testing"
)

func TestTerms(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"", []string{}},
		{"The Quick, brown fox!", []string{"quick", "brown", "fox"}},
		{"ERR_CONN_RESET at line 42", []string{"err", "conn", "reset", "line", "42"}},
		{"Ünïcödé naïve café", []string{"ünïcödé", "naïve", "café"}},
		{"the a of", []string{}},
	}
	for _, tt := range tests {
		if got := Terms(tt.in); fmt.Sprint(got) != fmt.Sprint(tt.want) {
			t.Errorf("Terms(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

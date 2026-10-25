// Package tokenize provides the shared, deliberately simple text analyzer used
// by the BM25 index and the fake embedder.
package tokenize

import (
	"strings"
	"unicode"
)

// stopwords is a small English stopword list. It is intentionally short:
// BM25's IDF already discounts frequent terms, this only removes noise.
var stopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {},
	"by": {}, "for": {}, "from": {}, "how": {}, "i": {}, "in": {}, "is": {},
	"it": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {}, "this": {},
	"to": {}, "was": {}, "what": {}, "when": {}, "where": {}, "which": {},
	"who": {}, "why": {}, "will": {}, "with": {}, "do": {}, "does": {},
	"can": {}, "you": {}, "your": {}, "my": {}, "we": {}, "our": {},
}

// Terms lower-cases s, splits it on any rune that is not a letter or digit,
// and drops stopwords. It never returns empty strings.
func Terms(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := fields[:0]
	for _, f := range fields {
		if _, stop := stopwords[f]; stop {
			continue
		}
		out = append(out, f)
	}
	return out
}

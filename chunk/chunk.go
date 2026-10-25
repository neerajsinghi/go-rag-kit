// Package chunk provides ragkit.Chunker implementations: a fixed-size
// splitter with overlap and a markdown, heading-aware splitter.
//
// Sizes are measured in runes (Unicode code points), never bytes, so
// multi-byte text is never split mid-character. Chunk offsets are byte
// offsets into the source text, so doc.Text[c.Start:c.End] == c.Text.
package chunk

import (
	"fmt"
	"strings"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// Metadata keys added by the chunkers in this package, in addition to the
// source document's metadata.
const (
	// MetaSection is the heading path of a markdown chunk, e.g. "Guide > Install".
	MetaSection = "section"
	// MetaHeading is the innermost heading of a markdown chunk.
	MetaHeading = "heading"
)

// span is a half-open byte range [start, end).
type span struct{ start, end int }

func validateDoc(doc ragkit.Document) error {
	if doc.ID == "" {
		return fmt.Errorf("%w: document ID is empty", ragkit.ErrInvalidArgument)
	}
	return nil
}

func validateSize(size, overlap int) error {
	if size <= 0 {
		return fmt.Errorf("%w: chunk size must be positive, got %d", ragkit.ErrInvalidArgument, size)
	}
	if overlap < 0 || overlap >= size {
		return fmt.Errorf("%w: overlap must be in [0, size), got %d with size %d", ragkit.ErrInvalidArgument, overlap, size)
	}
	return nil
}

// splitRunes splits text[start:end] into windows of at most size runes where
// consecutive windows share overlap runes. size > overlap >= 0 is assumed.
func splitRunes(text string, start, end, size, overlap int) []span {
	offs := make([]int, 0, end-start+1)
	for i := range text[start:end] {
		offs = append(offs, start+i)
	}
	n := len(offs)
	if n == 0 {
		return nil
	}
	offs = append(offs, end) // sentinel so offs[n] is the end offset
	step := size - overlap
	var out []span
	for i := 0; ; i += step {
		j := min(i+size, n)
		out = append(out, span{offs[i], offs[j]})
		if j == n {
			return out
		}
	}
}

// build converts spans into chunks, dropping whitespace-only spans and
// numbering the remaining ones consecutively.
func build(doc ragkit.Document, spans []span, extra func(i int) map[string]string) []ragkit.Chunk {
	chunks := make([]ragkit.Chunk, 0, len(spans))
	for i, s := range spans {
		text := doc.Text[s.start:s.end]
		if strings.TrimSpace(text) == "" {
			continue
		}
		meta := ragkit.CopyMetadata(doc.Metadata)
		if extra != nil {
			for k, v := range extra(i) {
				if meta == nil {
					meta = make(map[string]string)
				}
				meta[k] = v
			}
		}
		idx := len(chunks)
		chunks = append(chunks, ragkit.Chunk{
			ID:       ragkit.ChunkID(doc.ID, idx),
			DocID:    doc.ID,
			Index:    idx,
			Text:     text,
			Start:    s.start,
			End:      s.end,
			Metadata: meta,
		})
	}
	return chunks
}

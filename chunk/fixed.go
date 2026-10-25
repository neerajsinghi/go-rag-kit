package chunk

import ragkit "github.com/neerajsinghi/go-rag-kit"

// Fixed splits text into windows of Size runes, where consecutive windows
// share Overlap runes. It requires Size > 0 and 0 <= Overlap < Size.
//
// Whitespace-only windows are dropped; an empty document yields no chunks.
type Fixed struct {
	Size    int
	Overlap int
}

var _ ragkit.Chunker = Fixed{}

// Chunk implements ragkit.Chunker.
func (f Fixed) Chunk(doc ragkit.Document) ([]ragkit.Chunk, error) {
	if err := validateDoc(doc); err != nil {
		return nil, err
	}
	if err := validateSize(f.Size, f.Overlap); err != nil {
		return nil, err
	}
	return build(doc, splitRunes(doc.Text, 0, len(doc.Text), f.Size, f.Overlap), nil), nil
}

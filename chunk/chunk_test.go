package chunk

import (
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// checkInvariants verifies properties every chunker output must satisfy.
func checkInvariants(t *testing.T, doc ragkit.Document, chunks []ragkit.Chunk, maxRunes int) {
	t.Helper()
	for i, c := range chunks {
		if c.Index != i {
			t.Errorf("chunk %d has Index %d", i, c.Index)
		}
		if c.ID != ragkit.ChunkID(doc.ID, i) {
			t.Errorf("chunk %d has ID %q", i, c.ID)
		}
		if c.DocID != doc.ID {
			t.Errorf("chunk %d has DocID %q", i, c.DocID)
		}
		if c.Start < 0 || c.End > len(doc.Text) || c.Start >= c.End {
			t.Fatalf("chunk %d has bad offsets [%d,%d)", i, c.Start, c.End)
		}
		if doc.Text[c.Start:c.End] != c.Text {
			t.Errorf("chunk %d text does not match offsets", i)
		}
		if !utf8.ValidString(c.Text) {
			t.Errorf("chunk %d is not valid UTF-8: %q", i, c.Text)
		}
		if n := utf8.RuneCountInString(c.Text); n > maxRunes {
			t.Errorf("chunk %d has %d runes, max %d", i, n, maxRunes)
		}
		for k, v := range doc.Metadata {
			if c.Metadata[k] != v {
				t.Errorf("chunk %d lost metadata %q", i, k)
			}
		}
	}
}

func TestFixed(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		size    int
		overlap int
		want    []string
	}{
		{"empty", "", 4, 1, nil},
		{"whitespace only", "   \n\t ", 4, 1, nil},
		{"shorter than size", "abc", 10, 2, []string{"abc"}},
		{"exact size", "abcd", 4, 0, []string{"abcd"}},
		{"no overlap", "abcdefghij", 4, 0, []string{"abcd", "efgh", "ij"}},
		{"with overlap", "abcdefghij", 4, 2, []string{"abcd", "cdef", "efgh", "ghij"}},
		{"overlap tail", "abcdefg", 4, 1, []string{"abcd", "defg"}},
		{"multibyte runes", "héllo wörld", 4, 1, []string{"héll", "lo w", "wörl", "ld"}},
		{"emoji", "😀😁😂🤣😃", 2, 0, []string{"😀😁", "😂🤣", "😃"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc := ragkit.Document{ID: "d", Text: tt.text, Metadata: map[string]string{"source": "test.txt"}}
			got, err := Fixed{Size: tt.size, Overlap: tt.overlap}.Chunk(doc)
			if err != nil {
				t.Fatalf("Chunk: %v", err)
			}
			checkInvariants(t, doc, got, tt.size)
			if len(got) != len(tt.want) {
				t.Fatalf("got %d chunks %v, want %d", len(got), texts(got), len(tt.want))
			}
			for i := range got {
				if got[i].Text != tt.want[i] {
					t.Errorf("chunk %d = %q, want %q", i, got[i].Text, tt.want[i])
				}
			}
		})
	}
}

func TestFixedInvalid(t *testing.T) {
	tests := []struct {
		name string
		f    Fixed
		doc  ragkit.Document
	}{
		{"zero size", Fixed{Size: 0}, ragkit.Document{ID: "d", Text: "x"}},
		{"negative size", Fixed{Size: -1}, ragkit.Document{ID: "d", Text: "x"}},
		{"overlap equals size", Fixed{Size: 3, Overlap: 3}, ragkit.Document{ID: "d", Text: "x"}},
		{"negative overlap", Fixed{Size: 3, Overlap: -1}, ragkit.Document{ID: "d", Text: "x"}},
		{"missing doc id", Fixed{Size: 3}, ragkit.Document{Text: "x"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.f.Chunk(tt.doc)
			if !errors.Is(err, ragkit.ErrInvalidArgument) {
				t.Fatalf("err = %v, want ErrInvalidArgument", err)
			}
		})
	}
}

func TestFixedMetadataIsCopied(t *testing.T) {
	doc := ragkit.Document{ID: "d", Text: "abcdefgh", Metadata: map[string]string{"k": "v"}}
	got, err := Fixed{Size: 4}.Chunk(doc)
	if err != nil {
		t.Fatal(err)
	}
	got[0].Metadata["k"] = "mutated"
	if doc.Metadata["k"] != "v" || got[1].Metadata["k"] != "v" {
		t.Fatal("chunk metadata aliases document metadata")
	}
}

const sampleMarkdown = `Intro paragraph before any heading.

# Guide

Welcome to the guide.

## Install

Run go get.

` + "```sh" + `
# this is a shell comment, not a heading

go get example.com/x
` + "```" + `

## Empty

### Deep

Deep content. ##

#hashtag is not a heading.
`

func TestMarkdownSections(t *testing.T) {
	doc := ragkit.Document{ID: "readme", Text: sampleMarkdown, Metadata: map[string]string{"path": "README.md"}}
	got, err := Markdown{MaxSize: 500}.Chunk(doc)
	if err != nil {
		t.Fatal(err)
	}
	checkInvariants(t, doc, got, 500)

	want := []struct {
		section, heading, prefix string
	}{
		{"", "", "Intro paragraph"},
		{"Guide", "Guide", "# Guide"},
		{"Guide > Install", "Install", "## Install"},
		{"Guide > Empty > Deep", "Deep", "### Deep"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d chunks %q, want %d", len(got), texts(got), len(want))
	}
	for i, w := range want {
		c := got[i]
		if c.Metadata[MetaSection] != w.section || c.Metadata[MetaHeading] != w.heading {
			t.Errorf("chunk %d section=%q heading=%q, want %q/%q", i, c.Metadata[MetaSection], c.Metadata[MetaHeading], w.section, w.heading)
		}
		if !strings.HasPrefix(c.Text, w.prefix) {
			t.Errorf("chunk %d = %q, want prefix %q", i, c.Text, w.prefix)
		}
	}
	if !strings.Contains(got[2].Text, "go get example.com/x") || !strings.Contains(got[2].Text, "# this is a shell comment") {
		t.Errorf("fenced code block not kept with its section: %q", got[2].Text)
	}
	if !strings.Contains(got[3].Text, "#hashtag") {
		t.Errorf("#hashtag line should be body text: %q", got[3].Text)
	}
}

func TestMarkdownPacksAndSplits(t *testing.T) {
	long := strings.Repeat("x", 25)
	text := "# T\n\npara one\n\npara two\n\n" + long + "\n\nafter\n"
	doc := ragkit.Document{ID: "d", Text: text}
	got, err := Markdown{MaxSize: 20, Overlap: 5}.Chunk(doc)
	if err != nil {
		t.Fatal(err)
	}
	checkInvariants(t, doc, got, 20)
	want := []string{"# T\n\npara one", "para two", strings.Repeat("x", 20), strings.Repeat("x", 10), "after"}
	if len(got) != len(want) {
		t.Fatalf("got %q, want %q", texts(got), want)
	}
	for i := range want {
		if got[i].Text != want[i] {
			t.Errorf("chunk %d = %q, want %q", i, got[i].Text, want[i])
		}
		if got[i].Metadata[MetaSection] != "T" {
			t.Errorf("chunk %d section = %q", i, got[i].Metadata[MetaSection])
		}
	}
}

func TestMarkdownEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		text string
		want int
	}{
		{"empty", "", 0},
		{"headings only", "# A\n## B\n### C\n", 0},
		{"crlf line endings", "# A\r\n\r\nbody\r\n", 1},
		{"no trailing newline", "# A\nbody", 1},
		{"unclosed fence", "# A\n```\n# not heading\n", 1},
		{"seven hashes is text", "####### seven\n", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc := ragkit.Document{ID: "d", Text: tt.text}
			got, err := Markdown{}.Chunk(doc)
			if err != nil {
				t.Fatal(err)
			}
			checkInvariants(t, doc, got, DefaultMarkdownMaxSize)
			if len(got) != tt.want {
				t.Fatalf("got %d chunks %q, want %d", len(got), texts(got), tt.want)
			}
		})
	}
}

func TestParseHeading(t *testing.T) {
	tests := []struct {
		line  string
		level int
		title string
		ok    bool
	}{
		{"# Title", 1, "Title", true},
		{"### Title ###", 3, "Title", true},
		{"## C#", 2, "C#", true},
		{"   # indented", 1, "indented", true},
		{"    # code", 0, "", false},
		{"#nospace", 0, "", false},
		{"#", 1, "", true},
		{"plain", 0, "", false},
	}
	for _, tt := range tests {
		level, title, ok := parseHeading(tt.line)
		if level != tt.level || title != tt.title || ok != tt.ok {
			t.Errorf("parseHeading(%q) = %d,%q,%v want %d,%q,%v", tt.line, level, title, ok, tt.level, tt.title, tt.ok)
		}
	}
}

func TestMarkdownInvalid(t *testing.T) {
	for _, m := range []Markdown{{MaxSize: -1}, {MaxSize: 10, Overlap: 10}, {Overlap: -1}} {
		if _, err := m.Chunk(ragkit.Document{ID: "d", Text: "x"}); !errors.Is(err, ragkit.ErrInvalidArgument) {
			t.Errorf("%+v: err = %v, want ErrInvalidArgument", m, err)
		}
	}
}

func texts(cs []ragkit.Chunk) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.Text
	}
	return out
}

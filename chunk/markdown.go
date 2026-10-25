package chunk

import (
	"strings"
	"unicode/utf8"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// DefaultMarkdownMaxSize is the chunk size, in runes, used by Markdown when
// MaxSize is zero.
const DefaultMarkdownMaxSize = 1000

// Markdown splits markdown documents along ATX headings ("#" to "######").
//
// Each section (a heading plus the content up to the next heading) is
// chunked independently, so a chunk never spans two sections. Within a
// section, paragraphs are packed greedily into chunks of at most MaxSize
// runes; a single paragraph longer than MaxSize falls back to fixed-size
// splitting with Overlap. Fenced code blocks (``` or ~~~) are kept intact
// where they fit and "#" lines inside them are not treated as headings.
//
// Every chunk carries the document metadata plus MetaSection (the heading
// path, e.g. "Guide > Install") and MetaHeading (the innermost heading).
// Sections consisting of only a heading are skipped.
type Markdown struct {
	// MaxSize is the maximum chunk size in runes. Zero means
	// DefaultMarkdownMaxSize.
	MaxSize int
	// Overlap is used only when an oversized paragraph is split.
	Overlap int
}

var _ ragkit.Chunker = Markdown{}

// Chunk implements ragkit.Chunker.
func (m Markdown) Chunk(doc ragkit.Document) ([]ragkit.Chunk, error) {
	if err := validateDoc(doc); err != nil {
		return nil, err
	}
	maxSize := m.MaxSize
	if maxSize == 0 {
		maxSize = DefaultMarkdownMaxSize
	}
	if err := validateSize(maxSize, m.Overlap); err != nil {
		return nil, err
	}

	var spans []span
	var paths [][]string
	for _, sec := range parseSections(doc.Text) {
		for _, s := range pack(doc.Text, sec.paras, maxSize, m.Overlap) {
			spans = append(spans, s)
			paths = append(paths, sec.path)
		}
	}
	return build(doc, spans, func(i int) map[string]string {
		p := paths[i]
		if len(p) == 0 {
			return nil
		}
		return map[string]string{
			MetaSection: strings.Join(p, " > "),
			MetaHeading: p[len(p)-1],
		}
	}), nil
}

type section struct {
	path       []string
	paras      []span
	hasHeading bool
}

func (s section) hasBody() bool {
	if s.hasHeading {
		return len(s.paras) > 1
	}
	return len(s.paras) > 0
}

type heading struct {
	level int
	title string
}

// parseSections scans text line by line and groups paragraphs into sections.
func parseSections(text string) []section {
	var (
		secs       []section
		cur        section
		stack      []heading
		inFence    bool
		fence      string
		paraStart  = -1
		paraEnd    int
		appendPara = func(start, end int) {
			if paraStart < 0 {
				paraStart = start
			}
			paraEnd = end
		}
		flushPara = func() {
			if paraStart >= 0 {
				cur.paras = append(cur.paras, span{paraStart, paraEnd})
				paraStart = -1
			}
		}
	)

	for pos := 0; pos < len(text); {
		lineEnd, next := len(text), len(text)
		if nl := strings.IndexByte(text[pos:], '\n'); nl >= 0 {
			lineEnd, next = pos+nl, pos+nl+1
		}
		line := strings.TrimRight(text[pos:lineEnd], "\r")
		contentEnd := pos + len(line)

		if inFence {
			appendPara(pos, contentEnd)
			if strings.HasPrefix(strings.TrimLeft(line, " "), fence) {
				inFence = false
			}
		} else if marker, ok := fenceOpen(line); ok {
			inFence, fence = true, marker
			appendPara(pos, contentEnd)
		} else if level, title, ok := parseHeading(line); ok {
			flushPara()
			secs = append(secs, cur)
			for len(stack) > 0 && stack[len(stack)-1].level >= level {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, heading{level, title})
			path := make([]string, len(stack))
			for i, h := range stack {
				path[i] = h.title
			}
			cur = section{path: path, hasHeading: true, paras: []span{{pos, contentEnd}}}
		} else if strings.TrimSpace(line) == "" {
			flushPara()
		} else {
			appendPara(pos, contentEnd)
		}
		pos = next
	}
	flushPara()
	secs = append(secs, cur)

	out := secs[:0]
	for _, s := range secs {
		if s.hasBody() {
			out = append(out, s)
		}
	}
	return out
}

// leadingSpaces reports whether line starts with at most three spaces of
// indentation and returns the line without them.
func leadingSpaces(line string) (string, bool) {
	trimmed := strings.TrimLeft(line, " ")
	return trimmed, len(line)-len(trimmed) <= 3
}

func fenceOpen(line string) (string, bool) {
	s, ok := leadingSpaces(line)
	if !ok {
		return "", false
	}
	for _, m := range []string{"```", "~~~"} {
		if strings.HasPrefix(s, m) {
			return m, true
		}
	}
	return "", false
}

// parseHeading recognises an ATX heading line and returns its level and
// title, with any closing "#" sequence removed.
func parseHeading(line string) (int, string, bool) {
	s, ok := leadingSpaces(line)
	if !ok {
		return 0, "", false
	}
	level := 0
	for level < len(s) && s[level] == '#' {
		level++
	}
	if level == 0 || level > 6 {
		return 0, "", false
	}
	rest := s[level:]
	if rest != "" && rest[0] != ' ' && rest[0] != '\t' {
		return 0, "", false // "#hashtag" is not a heading
	}
	title := strings.TrimSpace(rest)
	if closed := strings.TrimRight(title, "#"); closed != title {
		if closed == "" || strings.HasSuffix(closed, " ") || strings.HasSuffix(closed, "\t") {
			title = strings.TrimSpace(closed)
		}
	}
	return level, title, true
}

// pack greedily merges paragraphs into spans of at most maxSize runes.
func pack(text string, paras []span, maxSize, overlap int) []span {
	var out []span
	cur := span{-1, -1}
	for _, p := range paras {
		if utf8.RuneCountInString(text[p.start:p.end]) > maxSize {
			if cur.start >= 0 {
				out = append(out, cur)
				cur = span{-1, -1}
			}
			out = append(out, splitRunes(text, p.start, p.end, maxSize, overlap)...)
			continue
		}
		switch {
		case cur.start < 0:
			cur = p
		case utf8.RuneCountInString(text[cur.start:p.end]) <= maxSize:
			cur.end = p.end
		default:
			out = append(out, cur)
			cur = p
		}
	}
	if cur.start >= 0 {
		out = append(out, cur)
	}
	return out
}

// Package openai implements ragkit.Embedder on top of the OpenAI embeddings
// HTTP API (POST /v1/embeddings) using only net/http.
//
// The client batches inputs, retries rate-limit (429) and server (5xx)
// responses with exponential backoff, honours Retry-After, and respects
// context cancellation. Any OpenAI-compatible endpoint (Azure OpenAI via a
// proxy, vLLM, Ollama's OpenAI API, LiteLLM, ...) can be targeted with
// WithBaseURL.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// Defaults used by New.
const (
	DefaultModel      = "text-embedding-3-small"
	DefaultBaseURL    = "https://api.openai.com/v1"
	DefaultBatchSize  = 256
	DefaultMaxRetries = 3
)

// maxErrorBody bounds how much of an error response is read.
const maxErrorBody = 64 << 10

// Client is an OpenAI embeddings client. Create it with New; it is safe for
// concurrent use.
type Client struct {
	apiKey     string
	model      string
	baseURL    string
	dimensions int
	batchSize  int
	maxRetries int
	httpClient *http.Client
	backoff    func(attempt int) time.Duration
}

var _ ragkit.Embedder = (*Client)(nil)

// Option configures a Client.
type Option func(*Client)

// WithModel sets the embedding model (default DefaultModel).
func WithModel(model string) Option { return func(c *Client) { c.model = model } }

// WithBaseURL sets the API base URL, e.g. "https://api.openai.com/v1".
func WithBaseURL(u string) Option {
	return func(c *Client) { c.baseURL = strings.TrimRight(u, "/") }
}

// WithDimensions requests shortened embeddings (text-embedding-3 models
// only). Zero leaves the model default.
func WithDimensions(n int) Option { return func(c *Client) { c.dimensions = n } }

// WithBatchSize sets the maximum number of inputs per request.
func WithBatchSize(n int) Option { return func(c *Client) { c.batchSize = n } }

// WithMaxRetries sets how many times a retryable failure is retried. Zero
// disables retries.
func WithMaxRetries(n int) Option { return func(c *Client) { c.maxRetries = n } }

// WithHTTPClient sets the underlying HTTP client (default: 60s timeout).
func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.httpClient = hc } }

// New returns a Client authenticating with apiKey.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		apiKey:     apiKey,
		model:      DefaultModel,
		baseURL:    DefaultBaseURL,
		batchSize:  DefaultBatchSize,
		maxRetries: DefaultMaxRetries,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		backoff: func(attempt int) time.Duration {
			return time.Duration(500<<attempt) * time.Millisecond
		},
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// APIError is returned when the API responds with a non-2xx status.
type APIError struct {
	StatusCode int
	Type       string
	Code       string
	Message    string
}

func (e *APIError) Error() string {
	msg := e.Message
	if msg == "" {
		msg = http.StatusText(e.StatusCode)
	}
	return fmt.Sprintf("openai: HTTP %d: %s", e.StatusCode, msg)
}

func (e *APIError) retryable() bool {
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= 500
}

type request struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type response struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
}

// Embed implements ragkit.Embedder. Empty input strings are rejected because
// the API rejects them.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("%w: openai API key is empty", ragkit.ErrInvalidArgument)
	}
	if c.batchSize <= 0 || c.maxRetries < 0 || c.dimensions < 0 {
		return nil, fmt.Errorf("%w: invalid openai client options", ragkit.ErrInvalidArgument)
	}
	for i, t := range texts {
		if t == "" {
			return nil, fmt.Errorf("%w: input %d is empty", ragkit.ErrInvalidArgument, i)
		}
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += c.batchSize {
		batch := texts[start:min(start+c.batchSize, len(texts))]
		vecs, err := c.embedBatch(ctx, batch)
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (c *Client) embedBatch(ctx context.Context, batch []string) ([][]float32, error) {
	body, err := json.Marshal(request{Model: c.model, Input: batch, Dimensions: c.dimensions})
	if err != nil {
		return nil, fmt.Errorf("openai: encode request: %w", err)
	}
	for attempt := 0; ; attempt++ {
		vecs, retryAfter, err := c.do(ctx, body, len(batch))
		if err == nil {
			return vecs, nil
		}
		var apiErr *APIError
		retryable := errors.As(err, &apiErr) && apiErr.retryable() || isTransient(err)
		if !retryable || attempt >= c.maxRetries || ctx.Err() != nil {
			return nil, err
		}
		wait := c.backoff(attempt)
		if retryAfter > 0 {
			wait = retryAfter
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

// isTransient reports network-level failures worth retrying.
func isTransient(err error) bool {
	var netErr interface{ Timeout() bool }
	return errors.As(err, &netErr) && netErr.Timeout()
}

func (c *Client) do(ctx context.Context, body []byte, n int) ([][]float32, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, 0, fmt.Errorf("openai: build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("openai: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		apiErr := &APIError{StatusCode: resp.StatusCode}
		var er errorResponse
		if json.Unmarshal(raw, &er) == nil {
			apiErr.Message, apiErr.Type = er.Error.Message, er.Error.Type
			if er.Error.Code != nil {
				apiErr.Code = fmt.Sprint(er.Error.Code)
			}
		}
		var retryAfter time.Duration
		if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s > 0 {
			retryAfter = time.Duration(s) * time.Second
		}
		return nil, retryAfter, apiErr
	}

	var r response
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, 0, fmt.Errorf("openai: decode response: %w", err)
	}
	out := make([][]float32, n)
	for _, d := range r.Data {
		if d.Index < 0 || d.Index >= n || out[d.Index] != nil {
			return nil, 0, fmt.Errorf("openai: unexpected embedding index %d", d.Index)
		}
		if len(d.Embedding) == 0 {
			return nil, 0, fmt.Errorf("openai: empty embedding at index %d", d.Index)
		}
		out[d.Index] = d.Embedding
	}
	for i, v := range out {
		if v == nil {
			return nil, 0, fmt.Errorf("openai: missing embedding for input %d", i)
		}
	}
	return out, 0, nil
}

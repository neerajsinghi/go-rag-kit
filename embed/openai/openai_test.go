package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	ragkit "github.com/neerajsinghi/go-rag-kit"
)

// fakeAPI emulates the embeddings endpoint: each input of length L gets the
// vector [L, index]. Data is returned in reverse order to exercise index
// handling.
func fakeAPI(t *testing.T, failFirst int, status int) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if r.URL.Path != "/embeddings" || r.Method != http.MethodPost {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`))
			return
		}
		if int(n) <= failFirst {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit"}}`))
			return
		}
		var req request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		type item struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		var data []item
		for i := len(req.Input) - 1; i >= 0; i-- {
			data = append(data, item{Index: i, Embedding: []float32{float32(len(req.Input[i])), float32(i)}})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data, "model": req.Model})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func noBackoff(c *Client) { c.backoff = func(int) time.Duration { return time.Millisecond } }

func TestEmbedBatchesAndOrders(t *testing.T) {
	srv, calls := fakeAPI(t, 0, 0)
	c := New("test-key", WithBaseURL(srv.URL+"/"), WithBatchSize(2))
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	vecs, err := c.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("made %d requests, want 3", got)
	}
	for i, v := range vecs {
		if int(v[0]) != len(texts[i]) {
			t.Errorf("vector %d = %v, out of order", i, v)
		}
	}
}

func TestEmbedRetries(t *testing.T) {
	tests := []struct {
		name      string
		failFirst int
		status    int
		retries   int
		wantErr   bool
		wantCalls int32
	}{
		{"recovers from 429", 2, http.StatusTooManyRequests, 3, false, 3},
		{"recovers from 503", 1, http.StatusServiceUnavailable, 3, false, 2},
		{"gives up", 5, http.StatusInternalServerError, 2, true, 3},
		{"no retry on 400", 5, http.StatusBadRequest, 3, true, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, calls := fakeAPI(t, tt.failFirst, tt.status)
			c := New("test-key", WithBaseURL(srv.URL), WithMaxRetries(tt.retries))
			noBackoff(c)
			_, err := c.Embed(context.Background(), []string{"hello"})
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if got := calls.Load(); got != tt.wantCalls {
				t.Errorf("calls = %d, want %d", got, tt.wantCalls)
			}
			var apiErr *APIError
			if tt.wantErr && (!errors.As(err, &apiErr) || apiErr.StatusCode != tt.status) {
				t.Errorf("want APIError with status %d, got %v", tt.status, err)
			}
		})
	}
}

func TestEmbedAPIErrorDetails(t *testing.T) {
	srv, _ := fakeAPI(t, 0, 0)
	_, err := New("wrong", WithBaseURL(srv.URL)).Embed(context.Background(), []string{"x"})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want *APIError", err)
	}
	if apiErr.StatusCode != 401 || apiErr.Code != "invalid_api_key" || apiErr.Message != "bad key" {
		t.Errorf("unexpected error details: %+v", apiErr)
	}
}

func TestEmbedInvalidInput(t *testing.T) {
	tests := []struct {
		name  string
		c     *Client
		texts []string
	}{
		{"missing key", New(""), []string{"x"}},
		{"empty text", New("k"), []string{"ok", ""}},
		{"bad batch size", New("k", WithBatchSize(0)), []string{"x"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.c.Embed(context.Background(), tt.texts); !errors.Is(err, ragkit.ErrInvalidArgument) {
				t.Fatalf("err = %v, want ErrInvalidArgument", err)
			}
		})
	}
}

func TestEmbedEmptyInputMakesNoRequest(t *testing.T) {
	srv, calls := fakeAPI(t, 0, 0)
	vecs, err := New("test-key", WithBaseURL(srv.URL)).Embed(context.Background(), nil)
	if err != nil || len(vecs) != 0 || calls.Load() != 0 {
		t.Fatalf("vecs=%v err=%v calls=%d", vecs, err, calls.Load())
	}
}

func TestEmbedContextCancelledDuringBackoff(t *testing.T) {
	srv, _ := fakeAPI(t, 100, http.StatusTooManyRequests)
	c := New("test-key", WithBaseURL(srv.URL))
	c.backoff = func(int) time.Duration { return time.Hour }
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := c.Embed(ctx, []string{"x"}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
}

func TestEmbedMalformedResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":5,"embedding":[1]}]}`))
	}))
	defer srv.Close()
	if _, err := New("k", WithBaseURL(srv.URL)).Embed(context.Background(), []string{"x"}); err == nil {
		t.Fatal("expected error for out-of-range index")
	}
}

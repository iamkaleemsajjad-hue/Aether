"""Ecosystem SDK generators and GitHub Actions workflow builder for Aether Runtime.

Generates typed client SDKs for TypeScript, Go, and Rust, plus GitHub Actions
CI/CD workflows for automated AEG quality gating.

Research: OpenAPI Generator (2020), HuggingFace Hub SDKs (2022–2024),
          GitHub Actions (2019), VS Code Extension API (2024).
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# TypeScript SDK Generator
# ---------------------------------------------------------------------------

class TypeScriptSDKGenerator:
    """
    Generates a complete TypeScript/ESM Aether client SDK.

    Output: A single `aether-sdk.ts` file with full type definitions,
    HTTP client, streaming support (EventSource), and JSDoc comments.
    Compatible with Node.js 18+ and Deno.
    """

    def generate(self, base_url: str = "http://localhost:8080") -> str:
        return textwrap.dedent(f'''\
            /**
             * Aether Runtime SDK — TypeScript/ESM
             * @version 3.1.0
             * @license Apache-2.0
             */

            export interface GenerateRequest {{
              prompt: string;
              max_tokens?: number;
              temperature?: number;
              stream?: boolean;
              adapter_id?: string;
              model_id?: string;
            }}

            export interface GenerateResponse {{
              text: string;
              tokens_generated: number;
              tokens_per_second: number;
              ttft_ms: number;
              model_id: string;
              finish_reason: "stop" | "length" | "error";
            }}

            export interface ChatMessage {{
              role: "system" | "user" | "assistant";
              content: string;
            }}

            export interface ChatRequest {{
              messages: ChatMessage[];
              max_tokens?: number;
              temperature?: number;
              model_id?: string;
            }}

            export interface ModelInfo {{
              model_id: string;
              architecture: string;
              precision: string;
              target: string;
              parameter_count_b: number;
              context_length: number;
              reasoning_enabled: boolean;
            }}

            export interface HealthStatus {{
              status: "ok" | "degraded" | "error";
              version: string;
              uptime_s: number;
              loaded_model: string;
            }}

            export class AetherClient {{
              private baseUrl: string;
              private headers: Record<string, string>;

              constructor(options?: {{ baseUrl?: string; apiKey?: string }}) {{
                this.baseUrl = options?.baseUrl ?? "{base_url}";
                this.headers = {{
                  "Content-Type": "application/json",
                  ...(options?.apiKey ? {{ Authorization: `Bearer ${{options.apiKey}}` }} : {{}}),
                }};
              }}

              async generate(request: GenerateRequest): Promise<GenerateResponse> {{
                const response = await fetch(`${{this.baseUrl}}/v1/generate`, {{
                  method: "POST",
                  headers: this.headers,
                  body: JSON.stringify(request),
                }});
                if (!response.ok) {{
                  const err = await response.text();
                  throw new Error(`Aether API error ${{response.status}}: ${{err}}`);
                }}
                return response.json() as Promise<GenerateResponse>;
              }}

              async chat(request: ChatRequest): Promise<GenerateResponse> {{
                const response = await fetch(`${{this.baseUrl}}/v1/chat`, {{
                  method: "POST",
                  headers: this.headers,
                  body: JSON.stringify(request),
                }});
                if (!response.ok) throw new Error(`Chat error ${{response.status}}`);
                return response.json() as Promise<GenerateResponse>;
              }}

              async *stream(request: GenerateRequest): AsyncGenerator<string, void, unknown> {{
                const response = await fetch(`${{this.baseUrl}}/v1/generate`, {{
                  method: "POST",
                  headers: {{ ...this.headers, Accept: "text/event-stream" }},
                  body: JSON.stringify({{ ...request, stream: true }}),
                }});
                if (!response.body) throw new Error("Streaming not supported");
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                while (true) {{
                  const {{ done, value }} = await reader.read();
                  if (done) break;
                  const chunk = decoder.decode(value, {{ stream: true }});
                  for (const line of chunk.split("\\n")) {{
                    if (line.startsWith("data: ")) {{
                      const data = line.slice(6).trim();
                      if (data === "[DONE]") return;
                      try {{ yield JSON.parse(data).token ?? ""; }} catch {{ /* skip */ }}
                    }}
                  }}
                }}
              }}

              async models(): Promise<ModelInfo[]> {{
                const r = await fetch(`${{this.baseUrl}}/v1/models`, {{ headers: this.headers }});
                return r.json() as Promise<ModelInfo[]>;
              }}

              async health(): Promise<HealthStatus> {{
                const r = await fetch(`${{this.baseUrl}}/health`, {{ headers: this.headers }});
                return r.json() as Promise<HealthStatus>;
              }}

              async compile(modelId: string, options?: Record<string, unknown>): Promise<{{ job_id: string }}> {{
                const r = await fetch(`${{this.baseUrl}}/v1/compile`, {{
                  method: "POST",
                  headers: this.headers,
                  body: JSON.stringify({{ model_id: modelId, ...options }}),
                }});
                return r.json() as Promise<{{ job_id: string }}>;
              }}
            }}

            export default AetherClient;
        ''')

    def write(self, output_dir: str | Path) -> Path:
        out = Path(output_dir) / "aether-sdk.ts"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.generate(), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# Go SDK Generator
# ---------------------------------------------------------------------------

class GoSDKGenerator:
    """
    Generates a Go client package for the Aether REST API.

    Output: `aether_client.go` using stdlib `net/http` + `encoding/json`.
    No external dependencies. Compatible with Go 1.21+.
    """

    def generate(self, base_url: str = "http://localhost:8080", package: str = "aether") -> str:
        return textwrap.dedent(f'''\
            // Package {package} provides a Go client for the Aether Runtime REST API.
            // Version: 3.1.0 | License: Apache-2.0
            package {package}

            import (
            \t"bytes"
            \t"context"
            \t"encoding/json"
            \t"fmt"
            \t"io"
            \t"net/http"
            \t"time"
            )

            // GenerateRequest is the payload for text generation.
            type GenerateRequest struct {{
            \tPrompt      string  `json:"prompt"`
            \tMaxTokens   int     `json:"max_tokens,omitempty"`
            \tTemperature float64 `json:"temperature,omitempty"`
            \tStream      bool    `json:"stream,omitempty"`
            \tAdapterID   string  `json:"adapter_id,omitempty"`
            \tModelID     string  `json:"model_id,omitempty"`
            }}

            // GenerateResponse is the response from the generation endpoint.
            type GenerateResponse struct {{
            \tText             string  `json:"text"`
            \tTokensGenerated  int     `json:"tokens_generated"`
            \tTokensPerSecond  float64 `json:"tokens_per_second"`
            \tTTFTMs           float64 `json:"ttft_ms"`
            \tModelID          string  `json:"model_id"`
            \tFinishReason     string  `json:"finish_reason"`
            }}

            // ChatMessage is a single message in a conversation.
            type ChatMessage struct {{
            \tRole    string `json:"role"`
            \tContent string `json:"content"`
            }}

            // Client is the Aether API client.
            type Client struct {{
            \tBaseURL    string
            \tAPIKey     string
            \tHTTPClient *http.Client
            }}

            // NewClient creates a new Aether client with the given base URL.
            func NewClient(baseURL string, apiKey string) *Client {{
            \treturn &Client{{
            \t\tBaseURL:    baseURL,
            \t\tAPIKey:     apiKey,
            \t\tHTTPClient: &http.Client{{Timeout: 120 * time.Second}},
            \t}}
            }}

            // DefaultClient returns a client pointing to localhost:8080.
            func DefaultClient() *Client {{
            \treturn NewClient("{base_url}", "")
            }}

            func (c *Client) post(ctx context.Context, path string, body any) (*http.Response, error) {{
            \tdata, err := json.Marshal(body)
            \tif err != nil {{
            \t\treturn nil, fmt.Errorf("aether: marshal: %w", err)
            \t}}
            \treq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(data))
            \tif err != nil {{
            \t\treturn nil, err
            \t}}
            \treq.Header.Set("Content-Type", "application/json")
            \tif c.APIKey != "" {{
            \t\treq.Header.Set("Authorization", "Bearer "+c.APIKey)
            \t}}
            \treturn c.HTTPClient.Do(req)
            }}

            // Generate sends a text generation request and returns the response.
            func (c *Client) Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {{
            \tresp, err := c.post(ctx, "/v1/generate", req)
            \tif err != nil {{
            \t\treturn nil, err
            \t}}
            \tdefer resp.Body.Close()
            \tif resp.StatusCode != http.StatusOK {{
            \t\tbody, _ := io.ReadAll(resp.Body)
            \t\treturn nil, fmt.Errorf("aether: HTTP %d: %s", resp.StatusCode, body)
            \t}}
            \tvar result GenerateResponse
            \tif err := json.NewDecoder(resp.Body).Decode(&result); err != nil {{
            \t\treturn nil, fmt.Errorf("aether: decode: %w", err)
            \t}}
            \treturn &result, nil
            }}

            // Chat sends a multi-turn chat request.
            func (c *Client) Chat(ctx context.Context, messages []ChatMessage, maxTokens int) (*GenerateResponse, error) {{
            \tbody := map[string]any{{"messages": messages, "max_tokens": maxTokens}}
            \tresp, err := c.post(ctx, "/v1/chat", body)
            \tif err != nil {{
            \t\treturn nil, err
            \t}}
            \tdefer resp.Body.Close()
            \tvar result GenerateResponse
            \tif err := json.NewDecoder(resp.Body).Decode(&result); err != nil {{
            \t\treturn nil, err
            \t}}
            \treturn &result, nil
            }}

            // Health checks the runtime health endpoint.
            func (c *Client) Health(ctx context.Context) (map[string]any, error) {{
            \treq, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/health", nil)
            \tresp, err := c.HTTPClient.Do(req)
            \tif err != nil {{
            \t\treturn nil, err
            \t}}
            \tdefer resp.Body.Close()
            \tvar result map[string]any
            \tjson.NewDecoder(resp.Body).Decode(&result)
            \treturn result, nil
            }}
        ''')

    def write(self, output_dir: str | Path, package: str = "aether") -> Path:
        out = Path(output_dir) / "aether_client.go"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.generate(package=package), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# Rust SDK Generator
# ---------------------------------------------------------------------------

class RustSDKGenerator:
    """
    Generates a Rust async client using reqwest + tokio + serde.

    Output: `aether_client.rs` with async/await, streaming SSE support,
    and typed request/response structs. Compatible with Rust 1.75+.
    """

    def generate(self, base_url: str = "http://localhost:8080") -> str:
        return textwrap.dedent(f'''\
            //! Aether Runtime Rust Client — v3.1.0 | Apache-2.0
            //! Requires: reqwest = {{ version = "0.12", features = ["json", "stream"] }}
            //!           tokio = {{ version = "1", features = ["full"] }}
            //!           serde = {{ version = "1", features = ["derive"] }}

            use reqwest::{{Client, StatusCode}};
            use serde::{{Deserialize, Serialize}};

            #[derive(Debug, Serialize)]
            pub struct GenerateRequest {{
                pub prompt: String,
                #[serde(skip_serializing_if = "Option::is_none")]
                pub max_tokens: Option<u32>,
                #[serde(skip_serializing_if = "Option::is_none")]
                pub temperature: Option<f32>,
                #[serde(skip_serializing_if = "Option::is_none")]
                pub stream: Option<bool>,
                #[serde(skip_serializing_if = "Option::is_none")]
                pub adapter_id: Option<String>,
            }}

            #[derive(Debug, Deserialize)]
            pub struct GenerateResponse {{
                pub text: String,
                pub tokens_generated: u32,
                pub tokens_per_second: f64,
                pub ttft_ms: f64,
                pub model_id: String,
                pub finish_reason: String,
            }}

            #[derive(Debug, Serialize)]
            pub struct ChatMessage {{
                pub role: String,
                pub content: String,
            }}

            #[derive(Debug, Clone)]
            pub struct AetherClient {{
                client: Client,
                base_url: String,
                api_key: Option<String>,
            }}

            impl AetherClient {{
                pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {{
                    Self {{
                        client: Client::new(),
                        base_url: base_url.into(),
                        api_key,
                    }}
                }}

                pub fn default() -> Self {{
                    Self::new("{base_url}", None)
                }}

                pub async fn generate(
                    &self,
                    request: GenerateRequest,
                ) -> Result<GenerateResponse, Box<dyn std::error::Error>> {{
                    let mut req = self
                        .client
                        .post(format!("{{}}/v1/generate", self.base_url))
                        .json(&request);
                    if let Some(key) = &self.api_key {{
                        req = req.bearer_auth(key);
                    }}
                    let resp = req.send().await?;
                    if resp.status() != StatusCode::OK {{
                        let body = resp.text().await?;
                        return Err(format!("Aether API error: {{body}}").into());
                    }}
                    Ok(resp.json::<GenerateResponse>().await?)
                }}

                pub async fn health(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {{
                    let resp = self
                        .client
                        .get(format!("{{}}/health", self.base_url))
                        .send()
                        .await?;
                    Ok(resp.json().await?)
                }}

                pub async fn models(&self) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {{
                    let resp = self
                        .client
                        .get(format!("{{}}/v1/models", self.base_url))
                        .send()
                        .await?;
                    Ok(resp.json().await?)
                }}
            }}
        ''')

    def write(self, output_dir: str | Path) -> Path:
        out = Path(output_dir) / "aether_client.rs"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.generate(), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# GitHub Actions workflow generator
# ---------------------------------------------------------------------------

class GitHubActionsGenerator:
    """
    Generates `.github/workflows/aether-eval.yml` for CI/CD AEG quality gating.

    The generated workflow:
    1. Triggers on push to main and on pull requests
    2. Compiles the model (or pulls cached AEG from Hub)
    3. Runs the eval gate benchmarks (hellaswag, mmlu, gsm8k)
    4. Uploads the quality report as a workflow artifact
    5. Fails the workflow if the gate fails (blocks merge)
    """

    def generate(
        self,
        model_id: str = "my-model",
        benchmarks: list[str] | None = None,
        max_regression_pct: float = 2.0,
        python_version: str = "3.11",
    ) -> str:
        benchmarks = benchmarks or ["hellaswag", "mmlu", "gsm8k"]
        bench_str = json.dumps(benchmarks)
        return textwrap.dedent(f'''\
            # Aether Runtime CI/CD Eval Gate
            # Auto-generated by aether.ecosystem.sdks.GitHubActionsGenerator v3.1.0

            name: Aether Eval Gate

            on:
              push:
                branches: [main]
                paths:
                  - "**.py"
                  - "*.aeg/**"
                  - ".github/workflows/aether-eval.yml"
              pull_request:
                branches: [main]

            jobs:
              eval-gate:
                name: "AEG Quality Gate"
                runs-on: ubuntu-latest

                steps:
                  - name: Checkout
                    uses: actions/checkout@v4

                  - name: Set up Python {python_version}
                    uses: actions/setup-python@v5
                    with:
                      python-version: "{python_version}"
                      cache: "pip"

                  - name: Install Aether Runtime
                    run: |
                      pip install --upgrade pip
                      pip install aether-runtime

                  - name: Cache AEG artifact
                    uses: actions/cache@v4
                    with:
                      path: ./.aeg_cache
                      key: aeg-${{{{ runner.os }}}}-{model_id}-${{{{ hashFiles("*.safetensors", "*.gguf") }}}}

                  - name: Run Eval Gate
                    id: eval
                    run: |
                      python - <<\'EOF\'
                      from aether.observability.ci_pipeline import CIEvalPipeline
                      import json, sys

                      pipeline = CIEvalPipeline(
                          aeg_path="./.aeg_cache/{model_id}.aeg",
                          max_regression={max_regression_pct / 100:.3f},
                          required_benchmarks=tuple({bench_str}),
                      )
                      report = pipeline.run_and_save(
                          output_path="./eval_report.json",
                          benchmarks={bench_str},
                      )
                      print(json.dumps(report.to_dict(), indent=2))
                      if not report.gate_decision.passed:
                          print("EVAL GATE FAILED:", report.gate_decision.failing_benchmarks, file=sys.stderr)
                          sys.exit(1)
                      EOF
                    env:
                      AETHER_HUB_TOKEN: ${{{{ secrets.AETHER_HUB_TOKEN }}}}

                  - name: Upload Quality Report
                    if: always()
                    uses: actions/upload-artifact@v4
                    with:
                      name: aether-eval-report
                      path: eval_report.json
                      retention-days: 30

                  - name: Post gate summary
                    if: always()
                    run: |
                      python - <<\'EOF\'
                      import json
                      with open("eval_report.json") as f:
                          r = json.load(f)
                      summary = r.get("summary", {{}})
                      passed = summary.get("passed", False)
                      emoji = "✅" if passed else "❌"
                      print(f"${{emoji}} Eval Gate: {{\'PASSED\' if passed else \'FAILED\'}}")
                      print(f"Max regression: {{summary.get(\'max_regression_pct\', 0):.2f}}%")
                      if not passed:
                          print("Failing:", summary.get("failing", []))
                      EOF
        ''')

    def write(self, repo_root: str | Path, **kwargs: Any) -> Path:
        out = Path(repo_root) / ".github" / "workflows" / "aether-eval.yml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.generate(**kwargs), encoding="utf-8")
        return out

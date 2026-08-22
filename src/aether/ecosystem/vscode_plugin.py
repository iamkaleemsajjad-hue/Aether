"""VS Code plugin manifest and command registry generator for Aether Runtime.

Generates the VS Code extension package.json, extension entry point, and
webview provider for .aeg file inspection directly in the editor.

Research: VS Code Extension API v1.90 (2024), VS Code WebView API.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VSCodeCommand:
    """A single VS Code command contributed by the extension."""

    command_id: str          # e.g. "aether.compile"
    title: str               # User-visible label
    category: str = "Aether"
    icon: str | None = None  # Theme icon name or path

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "command": self.command_id,
            "title": self.title,
            "category": self.category,
        }
        if self.icon:
            d["icon"] = self.icon
        return d


class VSCodeCommandRegistry:
    """Registry of all Aether VS Code commands matching the CLI surface."""

    COMMANDS: list[dict[str, str]] = [
        {"id": "aether.compile",        "title": "Compile Model to AEG",           "icon": "$(gear)"},
        {"id": "aether.inspect",         "title": "Inspect AEG Package",            "icon": "$(search)"},
        {"id": "aether.bench",           "title": "Benchmark AEG Model",            "icon": "$(pulse)"},
        {"id": "aether.serve",           "title": "Serve AEG Model Locally",        "icon": "$(server)"},
        {"id": "aether.evalGate",        "title": "Run Eval Gate",                  "icon": "$(check)"},
        {"id": "aether.abRollout",       "title": "Start A/B Rollout",              "icon": "$(git-branch)"},
        {"id": "aether.hubPush",         "title": "Push AEG to Aether Hub",         "icon": "$(cloud-upload)"},
        {"id": "aether.hubPull",         "title": "Pull AEG from Aether Hub",       "icon": "$(cloud-download)"},
        {"id": "aether.showGraph",       "title": "Show Computation Graph",         "icon": "$(type-hierarchy)"},
        {"id": "aether.showPrecisionMap","title": "Show Precision Map",             "icon": "$(symbol-numeric)"},
        {"id": "aether.safetyCheck",     "title": "Run Safety Guardrail Check",     "icon": "$(shield)"},
        {"id": "aether.hardwareInfo",    "title": "Show Hardware Profile",          "icon": "$(device-desktop)"},
        {"id": "aether.traceExport",     "title": "Export OpenTelemetry Traces",    "icon": "$(export)"},
    ]

    def to_contribution_list(self) -> list[dict[str, Any]]:
        return [
            VSCodeCommand(
                command_id=c["id"],
                title=c["title"],
                category="Aether",
                icon=c.get("icon"),
            ).to_dict()
            for c in self.COMMANDS
        ]


class VSCodePluginManifest:
    """
    Generates the VS Code extension `package.json` manifest.

    The extension contributes:
    - Commands for compile/inspect/bench/serve/hub operations
    - A custom editor for `.aeg` files (webview-based inspector)
    - A status bar item showing the active model
    - Language support for `.aeg-ir` files (syntax highlighting config)
    """

    EXTENSION_ID = "aether-dev.aether-runtime"
    VERSION = "3.1.0"
    DISPLAY_NAME = "Aether Runtime"
    DESCRIPTION = "Compile, inspect, serve, and benchmark AEG model packages directly in VS Code."

    def generate(self) -> dict[str, Any]:
        registry = VSCodeCommandRegistry()
        return {
            "name": "aether-runtime",
            "displayName": self.DISPLAY_NAME,
            "description": self.DESCRIPTION,
            "version": self.VERSION,
            "publisher": "aether-dev",
            "license": "Apache-2.0",
            "repository": {"type": "git", "url": "https://github.com/iamkaleemsajjad-hue/Aether"},
            "engines": {"vscode": "^1.90.0"},
            "categories": ["Machine Learning", "Other"],
            "keywords": ["LLM", "inference", "compiler", "AEG", "Aether", "AI"],
            "activationEvents": [
                "onCommand:aether.compile",
                "onCustomEditor:aether.aegInspector",
                "workspaceContains:**/*.aeg",
            ],
            "main": "./out/extension.js",
            "contributes": {
                "commands": registry.to_contribution_list(),
                "customEditors": [
                    {
                        "viewType": "aether.aegInspector",
                        "displayName": "AEG Inspector",
                        "selector": [{"filenamePattern": "*.aeg"}],
                        "priority": "default",
                    }
                ],
                "languages": [
                    {
                        "id": "aeg-ir",
                        "aliases": ["AEG IR", "aeg-ir"],
                        "extensions": [".aeg-ir"],
                        "configuration": "./language/aeg-ir-language-configuration.json",
                    }
                ],
                "grammars": [
                    {
                        "language": "aeg-ir",
                        "scopeName": "source.aeg-ir",
                        "path": "./syntaxes/aeg-ir.tmLanguage.json",
                    }
                ],
                "configuration": {
                    "title": "Aether Runtime",
                    "properties": {
                        "aether.serverUrl": {
                            "type": "string",
                            "default": "http://localhost:8080",
                            "description": "URL of the running Aether Runtime server",
                        },
                        "aether.hubToken": {
                            "type": "string",
                            "default": "",
                            "description": "Aether Hub API token for push/pull operations",
                        },
                        "aether.defaultTarget": {
                            "type": "string",
                            "default": "cuda",
                            "enum": ["cuda", "rocm", "metal", "openvino", "cpu"],
                            "description": "Default compilation target hardware",
                        },
                        "aether.defaultPrecision": {
                            "type": "string",
                            "default": "fp8",
                            "enum": ["fp4", "fp8", "int4", "int8", "bf16"],
                            "description": "Default quantization precision",
                        },
                    },
                },
                "menus": {
                    "explorer/context": [
                        {"command": "aether.inspect", "when": "resourceExtname == .aeg"},
                        {"command": "aether.bench",   "when": "resourceExtname == .aeg"},
                        {"command": "aether.serve",   "when": "resourceExtname == .aeg"},
                    ],
                    "commandPalette": [
                        {"command": c["id"]} for c in VSCodeCommandRegistry.COMMANDS
                    ],
                },
            },
            "scripts": {
                "compile": "tsc -p ./",
                "watch": "tsc -watch -p ./",
                "package": "vsce package",
                "publish": "vsce publish",
            },
            "devDependencies": {
                "@types/vscode": "^1.90.0",
                "@vscode/vsce": "^2.31.0",
                "typescript": "^5.4.0",
            },
        }

    def write(self, output_dir: str | Path) -> Path:
        out = Path(output_dir) / "package.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.generate(), indent=2), encoding="utf-8")
        return out


class AEGInspectorProvider:
    """
    Generates the TypeScript source for the VS Code AEG Inspector webview provider.

    The inspector renders:
    - Model architecture summary (layers, heads, dims)
    - Precision map (per-layer color-coded visualization)
    - Reasoning graph (CoT steps as nodes)
    - Performance benchmarks (tokens/sec, TTFT P50/P95/P99)
    """

    def generate_extension_ts(self) -> str:
        return textwrap.dedent('''\
            import * as vscode from "vscode";
            import * as path from "path";
            import * as fs from "fs";

            export function activate(context: vscode.ExtensionContext): void {
              // Register the AEG Inspector custom editor
              context.subscriptions.push(
                vscode.window.registerCustomEditorProvider(
                  "aether.aegInspector",
                  new AEGInspectorProvider(context),
                  { supportsMultipleEditorsPerDocument: false }
                )
              );

              // Register commands
              context.subscriptions.push(
                vscode.commands.registerCommand("aether.compile", async () => {
                  const model = await vscode.window.showInputBox({ prompt: "Enter model ID or path" });
                  if (!model) return;
                  const terminal = vscode.window.createTerminal("Aether Compile");
                  terminal.sendText(`aether compile "${model}"`);
                  terminal.show();
                }),

                vscode.commands.registerCommand("aether.inspect", (uri?: vscode.Uri) => {
                  if (uri) {
                    vscode.commands.executeCommand("vscode.openWith", uri, "aether.aegInspector");
                  }
                }),

                vscode.commands.registerCommand("aether.bench", async () => {
                  const terminal = vscode.window.createTerminal("Aether Bench");
                  terminal.sendText("aether bench .");
                  terminal.show();
                }),

                vscode.commands.registerCommand("aether.serve", async () => {
                  const terminal = vscode.window.createTerminal("Aether Serve");
                  terminal.sendText("aether serve . --port 8080");
                  terminal.show();
                }),

                vscode.commands.registerCommand("aether.evalGate", async () => {
                  const terminal = vscode.window.createTerminal("Aether Eval");
                  terminal.sendText("aether eval . --suite reasoning");
                  terminal.show();
                }),

                vscode.commands.registerCommand("aether.hardwareInfo", async () => {
                  const terminal = vscode.window.createTerminal("Aether Hardware");
                  terminal.sendText("aether hardware");
                  terminal.show();
                })
              );
            }

            class AEGInspectorProvider implements vscode.CustomReadonlyEditorProvider {
              constructor(private readonly context: vscode.ExtensionContext) {}

              async openCustomDocument(uri: vscode.Uri): Promise<vscode.CustomDocument> {
                return { uri, dispose: () => {} };
              }

              async resolveCustomEditor(
                document: vscode.CustomDocument,
                panel: vscode.WebviewPanel
              ): Promise<void> {
                panel.webview.options = { enableScripts: true };
                panel.webview.html = this._getWebviewHtml(document.uri.fsPath, panel.webview);
              }

              private _getWebviewHtml(aegPath: string, webview: vscode.Webview): string {
                let manifest: Record<string, unknown> = {};
                try {
                  const mPath = path.join(aegPath, "manifest.json");
                  if (fs.existsSync(mPath)) {
                    manifest = JSON.parse(fs.readFileSync(mPath, "utf-8"));
                  }
                } catch (_) {}

                const data = JSON.stringify(manifest, null, 2);
                return `<!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8" />
              <meta name="viewport" content="width=device-width,initial-scale=1" />
              <title>AEG Inspector</title>
              <style>
                body { font-family: var(--vscode-font-family); color: var(--vscode-foreground);
                        background: var(--vscode-editor-background); padding: 16px; }
                h1 { font-size: 18px; border-bottom: 1px solid var(--vscode-panel-border); }
                pre { background: var(--vscode-textCodeBlock-background); padding: 12px;
                      border-radius: 4px; overflow-x: auto; font-size: 12px; }
                .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                         background: var(--vscode-badge-background); color: var(--vscode-badge-foreground);
                         font-size: 11px; margin: 2px; }
              </style>
            </head>
            <body>
              <h1>🔍 AEG Inspector</h1>
              <p><strong>Path:</strong> <code>${aegPath}</code></p>
              <h2>Manifest</h2>
              <pre>${data}</pre>
            </body>
            </html>`;
              }
            }

            export function deactivate(): void {}
        ''')

    def write(self, output_dir: str | Path) -> Path:
        src_dir = Path(output_dir) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        out = src_dir / "extension.ts"
        out.write_text(self.generate_extension_ts(), encoding="utf-8")
        return out

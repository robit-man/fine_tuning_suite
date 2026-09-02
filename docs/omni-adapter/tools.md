# Portal tools and tool chaining

The reference portal includes a deliberately small tool harness for validating
the logical Omni tag's standard Ollama-compatible tool calling. The language
stage still produces ordinary `message.tool_calls`; the portal executes only
its server-owned allowlist, appends normal `role: "tool"` results, and asks the
same model for the final answer. Audio, image, and video comprehension therefore
can lead into the same tool loop without changing the adapter ABI.

This is runtime plumbing, not a claim that web access or memory is embedded in
the GGUF weights. Direct adapter users may continue to pass and execute their
own Ollama tool schemas. Automatic execution is an authenticated portal
extension enabled with `portal_auto_tools: true`.

## Built-in tools

| Tool | Purpose | State or network scope |
|---|---|---|
| `get_current_time` | Current portal-host date, time, timezone, and UTC offset | Read-only host metadata |
| `get_portal_capabilities` | Report model, media, document, and safe-tool capabilities | Read-only runtime metadata |
| `web_search` | Keyless DuckDuckGo titles, URLs, and snippets | Public web only |
| `web_fetch` | Fetch and extract bounded text from one source URL | Public HTTP(S) only |
| `document_search` | Search already attached PDF, DOCX, text, or code chunks | Current browser session only |
| `memory_write` | Store a compact temporary fact or research note | Current browser session only |
| `memory_read` | Read an exact temporary topic/key | Current browser session only |
| `memory_search` | Lexically retrieve temporary memories by relevance | Current browser session only |

The portal publishes the exact JSON schemas as `safe_tools` from `/api/status`.
The phone UI sends that array on chat and call turns, while the server replaces
it with its authoritative copy whenever automatic execution is requested. A
client cannot redefine a safe tool's implementation by changing its schema.

## Tool round lifecycle

```text
user/media turn
  -> optional Qwen3-Omni comprehension
  -> Qwen3.8 emits structured tool_calls
  -> portal validates and executes allowlisted calls
  -> portal appends assistant tool_calls + role=tool results
  -> Qwen3.8 may call another tool or produce the final answer
  -> optional Qwen3-TTS after no unresolved calls remain
```

Up to four tool rounds and four calls per round are allowed. Identical calls in
one turn are de-duplicated. Media bytes and raw document envelopes are removed
from follow-up rounds; tagged observations, retrieved text, and prior dialogue
remain as bounded text context. TTS is deferred while a tool call is unresolved
and runs only for the final answer.

The portal's NDJSON stream adds `type: "tool"` start/completion events between
normal adapter events. Its authoritative final response contains a redacted
`portal.safe_tools_executed` trace with names, compact arguments, and success
flags. Full web pages, memory values, and tool results are not duplicated into
the browser trace. The UI exposes this evidence in a collapsible **Tools** row.

## Request example

The server injects its schemas, so callers need only opt into execution:

```json
{
  "model": "robit/qwen3.8-27b-e03-obliterated-omni:q4km",
  "messages": [{
    "role": "user",
    "content": "Find the current project release, read the primary source, remember the version, and summarize it."
  }],
  "omni": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat"
  },
  "response_modalities": ["text"],
  "speech_mode": "never",
  "think": false,
  "stream": false,
  "portal_auto_tools": true
}
```

A typical chain is `web_search -> web_fetch -> memory_write -> final answer`.
A later turn in the same browser session can use `memory_read` or
`memory_search`. Tool choice remains model-driven; a prompt should state when a
fresh source, exact page, or retained fact is required.

## Web safety and limits

The web split is adapted from the adjacent Omnius implementation: search finds
candidate pages and fetch reads one selected page. It preserves Omnius's
keyless DuckDuckGo path and short-lived same-URL cache while applying the
portal's public-tunnel constraints:

- only absolute HTTP(S) URLs are accepted;
- URL credentials, localhost, `.local`, metadata endpoints, and every
  non-global resolved IPv4/IPv6 address are blocked;
- every redirect is revalidated, with at most four redirects;
- response bodies are capped at 2 MiB and extracted output at 12,000
  characters;
- only text, HTML, JSON, XML, RSS, and Atom responses are accepted;
- scripts and styles are stripped; JavaScript is never executed;
- authentication, cookies, forms, downloads, and browser automation are not
  supported;
- fetched pages and search snippets are labelled untrusted data and cannot
  change system or tool policy.

Production deployments that require a stronger network boundary should place
the portal behind an egress proxy or firewall allowlist. DNS validation in an
application process is defense in depth, not a substitute for network policy.

## Memory and document boundaries

Temporary memory is keyed by a SHA-256 hash of the opaque Secure browser
session cookie. It allows 64 entries, 4,096 characters per entry, and 32,768
characters per session. Memory is in process only, expires after five idle
minutes, and is deleted immediately with the Trash control. It is never shared
between users and is not persistent knowledge-base storage.

`document_search` queries the same session-isolated index used by automatic
PDF/DOCX/text retrieval. It does not reopen arbitrary paths or execute scripts.
Its maximum eight excerpts remain within the 12,000-character retrieval budget.
Scanned PDF OCR, semantic vector search, AST execution, and arbitrary shell or
filesystem tools are intentionally outside this public demonstration harness.

## Verification

Unit gates cover:

- structured multi-round `memory_write -> memory_search -> final` streaming;
- search-result parsing followed by page fetch;
- script/style removal, response bounding, and private-address rejection;
- browser-session memory and document isolation;
- Trash-triggered memory, web-cache, document-index, and diagnostic cleanup;
- preservation of the original media-removal and no-context-bleed invariants.

Run the focused suite with:

```bash
PYTHONPATH=. .venv-omni/bin/pytest -q tests/test_omni_portal.py
```

# Portal tools and tool chaining

The reference portal includes a deliberately small tool harness for validating
the logical Omni tag's standard Ollama-compatible tool calling. The language
stage still produces ordinary `message.tool_calls`; the portal executes only
its server-owned allowlist, appends normal `role: "tool"` results, and asks the
same model for the final answer. Audio, image, and video comprehension therefore
can lead into the same tool loop without changing the adapter ABI.

This is runtime plumbing, not a claim that web access or memory is embedded in
the GGUF weights. Direct adapter users may continue to pass and execute their
own Ollama tool schemas. Portal execution is off by default. A user opts in for
the current browser session with the small wrench button beside the brain
button, which makes the client send `portal_auto_tools: true`. The server still
owns the schemas and implementations.

## Built-in tools

| Tool | Purpose | State or network scope |
|---|---|---|
| `get_current_time` | Current portal-host date, time, timezone, and UTC offset | Read-only host metadata |
| `get_portal_capabilities` | Report model, media, document, and safe-tool capabilities | Read-only runtime metadata |
| `web_search` | Discover public result links in a locally launched Chromium/Chrome process, or search this session's local page index | Public search page for `discover`; no network for `session` |
| `web_fetch` | Fetch and extract bounded text from one source URL | Public HTTP(S) only |
| `document_search` | Search already attached PDF, DOCX, text, or code chunks | Current browser session only |
| `memory_write` | Store a compact temporary fact or research note | Current browser session only |
| `memory_read` | Read an exact temporary topic/key | Current browser session only |
| `memory_search` | Lexically retrieve temporary memories by relevance | Current browser session only |
| `tool_search` | Search the server-owned allowlisted tool catalog | Read-only runtime metadata |
| `safe_math_eval` | Evaluate bounded arithmetic and common math functions with an AST interpreter | Pure computation; no code execution |
| `structured_read` | Read/query attached JSON, JSONL, CSV, TSV, or YAML | Current browser-session attachments only |
| `web_crawl` | Fetch a bounded same-origin page graph | Public HTTP(S), 8 pages and depth 2 maximum |
| `ocr_pdf` | OCR an attached scanned PDF and index the recognized text | Current browser-session attachments only |
| `session_search` | Federated recall over dialogue, memory, notes, tasks, documents, and fetched pages | Current browser session only |
| `audio_analyze` | Inspect observed audio streams, duration, format, and volume | Current browser-session media only |
| `video_scan` | Inspect observed video/audio streams and timeline metadata | Current browser-session media only |
| `working_notes` | Add, list, search, or remove bounded research notes | Current browser session only |
| `task_list` | Maintain bounded pending/in-progress/completed/blocked tasks | Current browser session only |

The portal publishes the exact JSON schemas as `safe_tools` from `/api/status`.
It also advertises `tool_execution.client_opt_in=true` and
`default_enabled=false`. The phone UI sends the schemas only while the wrench
is enabled, while the server replaces them with its authoritative copy whenever
automatic execution is requested. A client cannot redefine a safe tool's
implementation by changing its schema. The trusted tool-use contract is also
injected only for opted-in turns; tools-off turns receive neither that contract
nor schemas.

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

Up to 50 tool calls per turn are allowed, whether the model emits them one at a
time across 50 sequential rounds or batches several calls in a round. A single
round is also capped at 50, and the combined per-turn total never exceeds 50.
Identical calls in one turn are de-duplicated. Media bytes and raw document
envelopes are removed from follow-up rounds; tagged observations, retrieved
text, and prior dialogue remain as bounded text context. TTS is deferred while
a tool call is unresolved and runs only for the final answer.

The portal's NDJSON stream adds `type: "tool"` start/completion events between
normal adapter events. Start events identify running calls; completion events
carry their success state and a bounded result preview. The authoritative final
response repeats this evidence in `portal.safe_tools_executed`. The UI merges
the two phases by call ID and exposes arguments/results in a collapsible
**Tools** row, parallel to the reasoning disclosure. JSON arguments and results
render as bounded, nested key/value rows; plain-text previews are capped at
12,000 characters. Traces are retained only in that browser session's existing
five-minute cache.

Native `message.tool_calls` remain authoritative. For compatible renderers that
emit Omnius-style `<tool_call>{...}</tool_call>` text, the portal parses the
bounded JSON into the same structure and removes the control block from visible
answer text. Mixed or malformed textual calls fail closed; they are never shown
or executed as plausible answer text.

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

A typical dependent chain is
`tool_search -> web_search(mode=discover) -> web_fetch -> memory_write -> final answer`.
A later turn in the same browser session can use `memory_read` or
`memory_search`; `web_search(mode=session)` searches the already indexed result
and fetched-page text without another discovery request. Independent read-only
calls may share a tool round, but dependent calls wait for `role="tool"`
evidence. The runtime contract directs the model to select the narrowest tool,
avoid duplicate calls, fetch primary sources before citing them, and write
memory only when the user requests it or a compact fact is needed later in the
same session.

## Omnius-derived design selection

The implementation was selected after reviewing Omnius's search, fetch, crawl,
browser-action, network-egress, tool-executor, exposure-policy, batching,
textual-call parser, trace-collapse, and memory search/read/write paths. This
portal retains the pieces appropriate to a small public demonstration:

- trusted server-owned schemas and explicit client opt-in;
- native structured calls with a strict textual compatibility parser;
- bounded multi-round execution and `role="tool"` observations;
- dependent chaining, duplicate suppression, and read-only batching guidance;
- local browser discovery separated from direct page retrieval and bounded crawl;
- per-session fetched-page indexing and lexical term/bigram recall;
- attachment-scoped structured reads and OCR, pure AST math, session working
  state, and technical media probes;
- compact, collapsible running/completed tool receipts; and
- URL, DNS, redirect, media-type, size, session, and TTL boundaries.

The portal deliberately does not expose Omnius's general browser-action
surface. Its crawl is read-only, same-origin, and bounded to eight pages at
depth two. Models cannot click arbitrary DOM nodes, submit forms, reuse an
authenticated profile, download files, access local URLs, run shell commands,
or turn retrieved text into tool authority. Those capabilities require a
different trust profile and approval model.

## Web safety and limits

There is no hosted/keyless search API, API SDK, or API credential in this
harness. `web_search(mode=discover)` starts an installed Chromium/Chrome binary
with an ephemeral local profile and reads links from a normal public search
results page. The default page is Bing Web Search; it can be replaced with a
public search-page template containing `{query}` via
`OMNI_WEB_SEARCH_URL_TEMPLATE`. `OMNI_WEB_BROWSER` pins the local browser
executable. Linux, macOS, and Windows browser locations are discovered when no
override is present. Bot challenges fail closed instead of returning navigation
or promotional links as results.

Discovery indexes at most 48 result/fetched pages and 128,000 characters for
the opaque browser session. `web_search(mode=session)` ranks that local index
with a deterministic lexical term/bigram scorer and performs no network call.
`web_fetch` separately retrieves one chosen page with a direct bounded HTTP
client. `web_crawl` applies the same checks to at most eight same-origin pages,
depth two, and 20,000 returned characters. This split applies the portal's
public-tunnel constraints:

- only absolute HTTP(S) URLs are accepted;
- URL credentials, localhost, `.local`, metadata endpoints, and every
  non-global resolved IPv4/IPv6 address are blocked;
- every redirect is revalidated, with at most four redirects;
- response bodies are capped at 2 MiB and extracted output at 12,000
  characters;
- only text, HTML, JSON, XML, RSS, and Atom responses are accepted;
- scripts and styles are stripped from fetched evidence; the search page may
  execute only inside its disposable local browser profile;
- page fetching does not support authentication, cookies, forms, downloads, or
  arbitrary browser automation;
- fetched pages and search snippets are labelled untrusted data and cannot
  change system or tool policy.

Production deployments that require a stronger network boundary should place
the portal behind an egress proxy or firewall allowlist. DNS validation in an
application process is defense in depth, not a substitute for network policy.

## Memory and document boundaries

Temporary memory is keyed by a SHA-256 hash of the opaque Secure browser
session cookie. It allows 64 entries, 4,096 characters per entry, and 32,768
characters per session. Exact reads suggest related keys on a miss; searches
use the same deterministic lexical term/bigram relevance scorer as the local
web index. Memory is in process only, expires after five idle minutes, and is
deleted immediately with the Trash control. It is never shared between users
and is not persistent knowledge-base storage.

`document_search` queries the same session-isolated index used by automatic
PDF/DOCX/text retrieval. Raw attachment bytes are retained in process for five
idle minutes, bounded to 48 MiB per session, solely so `structured_read` and
`ocr_pdf` can operate without host paths. `ocr_pdf` uses `pdftoppm` and
`tesseract`, processes at most 50 pages, then adds recognized text to the same
session index. `structured_read` uses safe JSON/YAML parsers and bounded CSV/TSV
rows; it never evaluates document content.

`working_notes`, `task_list`, observed-media metadata, and conversation recall
share the same opaque session boundary and Trash/TTL cleanup. `audio_analyze`
and `video_scan` use `ffprobe` (and `ffmpeg` volume detection for audio) during
ingestion, retain only bounded technical results, and do not retain a second
copy of media bytes. `safe_math_eval` walks a limited arithmetic AST; imports,
attributes, variables, comprehensions, and arbitrary Python are impossible.
Arbitrary shell, host-filesystem, process-control, messaging, device-control,
and credentialed-browser tools remain intentionally excluded.

## Verification

Unit gates cover:

- structured multi-round `memory_write -> memory_search -> final` streaming;
- local-browser result parsing/redirect decoding followed by page fetch;
- fail-closed browser-challenge handling and network-free session-index recall;
- script/style removal, response bounding, and private-address rejection;
- browser-session memory and document isolation;
- allowlisted tool discovery and forbidden math-expression rejection;
- structured JSON/YAML paths and attachment-scoped OCR indexing;
- bounded same-origin crawling and federated session recall;
- audio/video observation isolation, working notes, and task state;
- Trash-triggered memory, web-cache, document-index, and diagnostic cleanup;
- preservation of the original media-removal and no-context-bleed invariants.

Run the focused suite with:

```bash
PYTHONPATH=. .venv-omni/bin/pytest -q tests/test_omni_portal.py
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TanitBot is a Tunisian Arabic (Derja) digital-safety RAG chatbot. Two independent halves:

1. **Python backend** (`rag_commandr.py`, single file, ~1000 lines) — a FAISS RAG pipeline over the PDFs in `RAG files/`, generating with Cohere Command R (35B) via `transformers`. Designed to run on a GPU pod (RunPod), not locally on this Mac.
2. **Next.js frontend** (`Interface/`) — bilingual (ar/en, RTL-aware) chat UI that proxies chat requests to the backend.

There are no automated tests. `test_models.py` is not a test suite — it's a benchmark harness for evaluating LLMs on Derja prompts (results land in the `eval_*.json` files).

## Commands

Backend (on a GPU host):
```bash
pip install torch transformers accelerate bitsandbytes faiss-cpu sentence-transformers pypdf fastapi uvicorn

python rag_commandr.py --server --port 8000   # FastAPI server mode (what the frontend talks to)
python rag_commandr.py --interactive          # terminal chat
python rag_commandr.py --query "..."          # one-shot query
python rag_commandr.py --build-index          # force-rebuild FAISS index from "RAG files/"
python test_models.py --models "id1,id2"      # benchmark models, writes eval_results.json
```
Gated HF models need `HF_TOKEN` (env var or `--hf-token`). Default quantization is 4-bit (`--quantize 0|4|8`).

Frontend (in `Interface/`, uses pnpm):
```bash
pnpm dev      # dev server
pnpm build
pnpm lint     # eslint
```

## Architecture

Request flow: browser → `Interface/app/api/chat/route.ts` (Next.js route) → POST to `RUNPOD_ENDPOINT_URL` (the Python server's `/api/chat`) → plain-text token stream piped back through both layers to `Interface/components/chat-interface.tsx`. If `RUNPOD_ENDPOINT_URL` is unset or unreachable, the route streams a canned fallback reply so the UI stays usable. `RUNPOD_ENDPOINT_URL` / `RUNPOD_API_KEY` go in `Interface/.env` (gitignored).

Backend pipeline inside `rag_commandr.py`:
- **Indexing**: `build_and_save_index` extracts PDF text (pypdf), chunks it (LangChain splitter with a pure-Python fallback), embeds with `paraphrase-multilingual-MiniLM-L12-v2` **on CPU** (deliberate — all VRAM is reserved for the 35B model), and saves FAISS index + chunk metadata to `rag_index/`. On startup `is_index_up_to_date` compares against the PDFs and auto-rebuilds if stale.
- **Generation**: `load_generation_model` loads Command R quantized; at 16-bit it manually offloads `embed_tokens` and `lm_head` to CPU to fit VRAM (see recent commits — this balance is fragile, OOMs were fixed by trial and error).
- **Server**: `run_server` exposes `POST /api/chat` (streams via `TextIteratorStreamer`, keeps only the last 8 messages of history to bound context), plus `/health`, `/api/status`, `/index-info`.

Prompting: `SYSTEM_INSTRUCTION` (~line 150) is the product's core. It enforces pure Tunisian Derja (explicitly bans Egyptian/Gulf dialect words), safety protocol for blackmail/digital-violence cases, and **verbatim full source names in citations** in the exact format `[المصدر: <name>, صفحة X]` — the model tends to truncate source names, and several commits have progressively tightened this rule. `get_clean_source_name` maps PDF filenames to human-readable Arabic titles; a new PDF in `RAG files/` needs an entry there.

The frontend has its own separate system prompts and a `retrieveContext` stub in `route.ts` — that path is legacy/fallback only; real RAG happens in Python.

## Gotchas

- The top ~100 lines of `rag_commandr.py` (and similar code in `test_models.py`) are deliberate environment shims for broken GPU-pod setups: mock `torchvision`/`torchaudio` imports, preload NVIDIA `.so` libs via ctypes, monkey-patch `nn.Module.set_submodule` for PyTorch < 2.5. Do not "clean up" or reorder them — they must run before `torch`/`transformers` imports.
- The backend can't run on this macOS machine (CUDA-dependent); code changes here are verified on the RunPod side.
- User-facing strings are Arabic-first; frontend copy lives in `Interface/lib/i18n.ts` (both `ar` and `en` must be updated together).

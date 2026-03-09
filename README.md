# 📡 Berlin AI Talent Radar

> Production-grade RAG intelligence platform analyzing Berlin's AI job market 
> against EU AI Act compliance. Built by a Data Engineer in Berlin.

## What This Does

1. **Analyzes 700-900 Berlin AI job postings** using a custom 73-skill taxonomy
2. **Cross-references every posting against the EU AI Act** — enforceable August 2, 2026
3. **Classifies every role** on an original Architecture-Execution Spectrum

## Quick Start
```bash
# No API keys needed for demo
python main.py demo
```

## Stack

- Python 3.11, Pydantic v2
- OpenAI embeddings (text-embedding-3-small) + GPT-4o-mini
- pgvector / numpy fallback
- Streamlit + Plotly dashboard

## Status

🚧 **Under active development** — Chat 1 of 7 complete

| Stage | Status |
|-------|--------|
| Config + Models | ✅ Complete |
| Collectors | 🔄 In Progress |
| Processors | ⏳ Pending |
| Embeddings + RAG | ⏳ Pending |
| Analytics | ⏳ Pending |
| Dashboard | ⏳ Pending |
| CLI + Tests | ⏳ Pending |

## Built By

Aditya Badagandi | M.Sc. in  Data Science,AI and Digital Business(Gisma University of Applied Sciences)
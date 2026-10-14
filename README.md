---
title: Cairn Study Companion
emoji: 🪨
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# Cairn: a source-grounded study companion

Multimodal AI Hackathon 2026 (IIT Mandi), Track D: Personalized Tutoring and Adaptive Learning.

Cairn turns lecture videos, slides and textbooks into one cited knowledge base, then uses it to
tutor, quiz and track what a student actually knows.

## Status

| Step | Component | State |
|---|---|---|
| 0 | Project skeleton, Spaces deployment | done |
| 1 | Ingestion (PDF, slides, video) into cited chunks | next |
| 2 | Concept graph and chunk tagging | |
| 3 | Grounded tutor chat with refusal | |
| 4 | Quiz generation, verification, learner model, dashboard | |
| 5 | Evaluation (RAGAS, simulated students) | |
| 6 | Optional: flow map, flashcards, schedule, Hindi | |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add at least one provider key
streamlit run app.py
pytest
```

## Deploy (Hugging Face Spaces, Docker SDK)

1. Create a Space, choose **Docker**, blank template.
2. Add the provider keys from `.env.example` under Settings, Variables and secrets.
3. Push this repository to the Space (`git push space main`).

## Layout

```
app.py            home and system status
pages/            Streamlit pages (Tutor, Quiz, Dashboard, Upload)
core/config.py    settings and environment
core/schema.py    Source, Chunk, Concept, Citation models
core/citations.py deep links to page, slide and video timestamp
core/llm.py       provider-agnostic LLM calls with cache and rate limit
core/db.py        SQLite schema for learners, questions, attempts
core/ui.py        shared Streamlit components (citation viewer)
static/courses/   course PDFs served at /app/static/courses/
data/kb/          pre-built knowledge base (committed)
data/runtime/     SQLite and caches (not committed)
tests/
```

# Intelligent Customer Support Agent — RAG + Classification + MLOps

*[Русская версия здесь](README_ru.md) · [Українська версія тут](README_uk.md)*

An end-to-end customer support automation system built on real, public Twitter support data (Apple, Amazon, Uber, Spotify, Delta — 20,000 real customer service exchanges). Given a new incoming ticket, the system classifies it, retrieves semantically similar past resolutions, and decides — draft an automatic response grounded in real precedent, or escalate to a human.

This is a portfolio project. It demonstrates the full lifecycle: real (not synthetic) data acquisition and cleaning, honest empirical model selection, a working retrieval-augmented generation pipeline, and a documented set of real limitations rather than a polished but misleading demo.

## Architecture

```
New ticket
    │
    ▼
Single embedding (Voyage AI voyage-4-lite)
    │
    ├──────────────┬─────────────────┐
    ▼              ▼                 ▼
Category        Priority        Semantic search
classifier      classifier      (top-5 similar
(8 classes)     (4 levels)       past tickets)
    │              │                 │
    └──────────────┴─────────────────┘
                   │
                   ▼
        Agent decision rule:
   priority-dependent similarity
   threshold → draft or escalate
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Draft response         Escalate to human
  (Claude, grounded
  in retrieved cases)
```

One embedding call per ticket serves all three downstream tasks (classification ×2, retrieval) — not three separate calls.

## Results — honestly reported

| Task | Metric | TF-IDF baseline | Embeddings (final) |
|---|---|---|---|
| Category (8 classes) | Accuracy | 58.95% | **70.57%** |
| Category (8 classes) | F1 macro | 0.567 | **0.686** |
| Priority (4 levels) | Accuracy | 50.28% | **60.40%** |
| Priority (4 levels) | F1 macro | 0.484 | **0.594** |
| Semantic retrieval | Category-match in top-5 | 36.3% (TF-IDF) | **58.1%** |

All metrics are evaluated on a **time-based** train/test split (earliest 80% of tickets as training data / knowledge base, most recent 20% as held-out test) — not a random split — to avoid the model being evaluated on data that leaks information from the future, and to mirror how the system would actually be used in production (searching past resolved tickets to help with a new one).

Category and priority labels were produced by an LLM (Claude Haiku) rather than hand-labeled, since the source dataset has no ground-truth labels for either. This is a form of weak supervision / distillation: the LLM labels the full dataset once, and a cheap, fast classical classifier (logistic regression) is trained on those labels for real-time use, rather than calling an LLM on every incoming ticket just to classify it.

## Why embeddings instead of TF-IDF

TF-IDF (bag-of-words) matches on shared *vocabulary*, regardless of meaning. A concrete, real failure case surfaced during development: a ticket about malfunctioning Amazon lockers retrieved a completely unrelated tweet about an iOS version number as its most "similar" match — the only thing they shared was the incidental phrase "I've got." Semantic embeddings fixed this specific case, and the aggregate retrieval-quality metric (category-match in top-5) improved from 36.3% to 58.1%. Both classifiers were then rebuilt on the same embeddings, given the same underlying representation problem plausibly affected them too — which the metrics above confirm it did.

## Known limitations (documented, not hidden)

- **Short, terse, negative messages remain a weak spot.** `"@Delta That's not good enough."` was misclassified by the TF-IDF version (as positive feedback, low priority) and, while the embeddings-based version corrected this specific example, sarcasm and dry understatement in general are not reliably detected by either approach — both rely on the text alone, with no conversation history or tone modeling.
- **Priority F1 macro (0.594) is meaningfully lower than category F1 (0.686).** Urgency is a more diffuse, tonal signal than topic, and is inherently harder to read from a single short message.
- **The "service disruption" category and "critical" priority level are the hardest classes** — both are minority classes in the training data, and both share vocabulary with adjacent categories (a flight delay complaint reads a lot like a general quality complaint).
- **Escalation thresholds were set by inspection of examples, not through a systematic threshold search** (e.g. ROC-based optimization) — a reasonable next step, not yet done.
- **Rate limits:** the free Voyage AI tier defaults to 3 requests/minute until a payment method is added to the account (the 200M free tokens remain free either way) — the provided scripts include retry/backoff logic to work within this, but expect the full pipeline to take significantly longer without a payment method on file.

## Repository structure

```
data_pipeline/
  prepare_dataset.py           # Filter raw Twitter CSV to target companies, reconstruct request/response pairs
  sample_and_filter.py         # Length/DM-escalation filtering, stratified sampling to a manageable size
  label_categories.py          # LLM (Claude) labeling of ticket category, batched
  label_priority.py            # LLM (Claude) labeling of ticket priority, batched
training/
  build_semantic_rag.py        # Embed knowledge base + evaluate retrieval quality (Voyage AI)
  train_final_embedding_classifiers.py  # Train final category + priority classifiers on embeddings
agent.py                       # Core orchestrator: embed -> classify -> retrieve -> decide -> respond/escalate
app.py                         # FastAPI wrapper exposing /handle_ticket
Dockerfile
requirements.txt
```

## Running it

### 1. Data pipeline (reproduces the dataset from scratch)

Requires the raw [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset from Kaggle (`twcs.csv`).

```bash
python data_pipeline/prepare_dataset.py
python data_pipeline/sample_and_filter.py
```

### 2. Labeling (requires an Anthropic API key - cost: a few dollars total)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python data_pipeline/label_categories.py
python data_pipeline/label_priority.py
```

### 3. Build the semantic index and train classifiers (requires a Voyage AI key - free tier covers this comfortably)

```bash
export VOYAGE_API_KEY="..."
python training/build_semantic_rag.py
python training/train_final_embedding_classifiers.py
```

### 4. Run the API locally

```bash
export VOYAGE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
uvicorn app:app --reload
```

Then: `POST http://localhost:8000/handle_ticket` with `{"text": "...", "company": "AmazonHelp", "is_dm_escalation": false}`.

### 5. Or run it in Docker

```bash
docker build -t customer-support-agent .
docker run -e VOYAGE_API_KEY=... -e ANTHROPIC_API_KEY=... -p 8000:8000 customer-support-agent
```

Model artifacts (`.joblib`, `.npy`, `kb_with_index.csv`) are not committed to git (see `.gitignore`) due to file size - generate them via the pipeline above, or use Git LFS if you'd rather commit them directly.

## Tech stack

Python - scikit-learn - Voyage AI (embeddings) - Anthropic Claude (labeling + response drafting) - FastAPI - Docker

## About the author

Viktor Romenskiy — GenAI/ML engineer (RAG systems, LLM evaluation, fine-tuning). GitHub: [viktorromenskiy-glitch](https://github.com/viktorromenskiy-glitch) · LinkedIn: [profile](https://www.linkedin.com/in/%D0%B2%D0%B8%D0%BA%D1%82%D0%BE%D1%80-%D1%80%D0%BE%D0%BC%D0%B5%D0%BD%D1%81%D0%BA%D0%B8%D0%B9-029b2086/) · Hugging Face: [ViktorPetrov123](https://huggingface.co/ViktorPetrov123) · Contact: viktorromenskiy@gmail.com

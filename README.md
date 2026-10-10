# Customer Support Agent — RAG + Classification + MLOps

*[Русская версия](README_ru.md) · [Українська версія](README_uk.md)*

A support-ticket automation system built on real public Twitter data (Apple, Amazon, Uber, Spotify, Delta — 20,000 real support requests). For each new ticket it classifies the ticket, retrieves semantically similar past resolutions, and decides whether to generate an automatic draft reply grounded in a real precedent or to escalate to a human.

This is a portfolio project. It shows the full lifecycle: collecting and cleaning real (not synthetic) data, an honest empirical model choice, a working RAG pipeline, and an explicit list of real limitations instead of a polished but misleading demo.

## Architecture

```
New ticket
    │
    ▼
One embedding (Voyage AI voyage-4-lite)
    │
    ├──────────────┬─────────────────┐
    ▼              ▼                 ▼
Category         Priority         Semantic search
classifier       classifier       (top-5 similar
(8 classes)      (4 levels)       past tickets)
    │              │                 │
    └──────────────┴─────────────────┘
                   │
                   ▼
        Agent decision rule:
   similarity threshold depends on
   priority -> draft reply or escalate
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  Draft reply            Escalate to a human
  (Claude, grounded
  in retrieved cases)
```

A single embedding call per ticket serves all three downstream tasks (two classifiers and retrieval), instead of three separate calls.

## Results

| Task | Metric | TF-IDF (baseline) | Embeddings (final) |
|---|---|---|---|
| Category (8 classes) | Accuracy | 58.95% | **70.57%** |
| Category (8 classes) | F1 macro | 0.567 | **0.686** |
| Priority (4 levels) | Accuracy | 50.28% | **60.40%** |
| Priority (4 levels) | F1 macro | 0.484 | **0.594** |
| Semantic search | Category match in top-5 | 36.3% (TF-IDF) | **58.1%** |

All metrics are measured on a **temporal** train/test split (the earliest 80% of tickets for training and the knowledge base, the latest 20% as a held-out test), not a random split. This avoids evaluating on data that "peeks from the future" and matches the real use case: searching already-solved tickets to help with a new one.

Category and priority labels come from an LLM (Claude Haiku), not from manual labeling, because the source dataset has neither. This is a form of weak supervision / distillation: the LLM labels the whole dataset once, and a cheap, fast classical classifier (logistic regression) is trained on those labels for real-time use, instead of calling the LLM for every incoming ticket just to classify it.

## Why embeddings instead of TF-IDF

TF-IDF matches on shared **vocabulary**, regardless of meaning. A real failure found during development: a ticket about faulty Amazon pickup lockers returned an unrelated tweet about an iOS version number as its "most similar" case; the only thing they shared was the accidental phrase "I've got". Semantic embeddings fixed that case, and the aggregate retrieval metric (category match in top-5) rose from 36.3% to 58.1%. Both classifiers were then rebuilt on the same embeddings, since the same representation problem likely affected them too, which the metrics above confirmed.

## Known limitations

- **Short, curt, negative messages remain a weak spot.** `"@Delta That's not good enough."` was misclassified by the TF-IDF version as positive feedback with low priority. The embedding version fixed this particular example, but sarcasm and dry understatement are not reliably caught by either approach. Both rely on the text alone, without conversation history or tone modeling.
- **Priority F1 macro (0.594) is clearly lower than category F1 macro (0.686).** Urgency is a vaguer, tone-based signal than topic and is harder to read from one short message.
- **The "service outage" category and the "critical" priority are the hardest classes.** Both are minority classes in the training data and overlap in vocabulary with neighboring categories (a complaint about a delayed flight sounds much like an ordinary quality complaint).
- **Escalation thresholds were set by inspecting examples, not by a systematic threshold search** (for example, ROC-curve optimization). That is a reasonable next step and has not been done yet.
- **Rate limits.** The free Voyage AI tier allows 3 requests per minute until a card is added to the account (the 200M free tokens stay free either way). The provided scripts include retry logic for this, but without a card the full pipeline takes noticeably longer.

## Repository structure

```
data_pipeline/
  prepare_dataset.py           # Filters the raw Twitter CSV by company, rebuilds request/reply pairs
  sample_and_filter.py         # Length / DM-escalation filtering, stratified sampling to a manageable size
  label_categories.py          # LLM (Claude) labeling of ticket category, in batches
  label_priority.py            # LLM (Claude) labeling of ticket priority, in batches
training/
  build_semantic_rag.py        # Embeds the knowledge base and evaluates retrieval quality (Voyage AI)
  train_final_embedding_classifiers.py  # Trains the final category and priority classifiers on embeddings
  embed_remaining_test_set.py  # Embeds the full held-out test set
agent.py                       # Main orchestrator: embedding -> classification -> retrieval -> decision -> reply / escalation
app.py                         # FastAPI wrapper with the /handle_ticket endpoint
Dockerfile
requirements.txt
RUNBOOK.md                     # Step-by-step guide from local files to a working repository
```

## How to run

### 1. Data pipeline (rebuilds the dataset from scratch)

Requires the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset from Kaggle (`twcs.csv`).

```bash
python data_pipeline/prepare_dataset.py
python data_pipeline/sample_and_filter.py
```

### 2. Labeling (needs an Anthropic API key, a few dollars in total)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python data_pipeline/label_categories.py
python data_pipeline/label_priority.py
```

### 3. Build the semantic index and train the classifiers (needs a Voyage AI key; the free tier is enough)

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

### 5. Or run in Docker

```bash
docker build -t customer-support-agent .
docker run -e VOYAGE_API_KEY=... -e ANTHROPIC_API_KEY=... -p 8000:8000 customer-support-agent
```

Model artifacts (`.joblib`, `.npy`, `kb_with_index.csv`) are not stored in git (see `.gitignore`) because of their size. Generate them with the pipeline above, or use Git LFS to commit them directly.

## Tech stack

Python · scikit-learn · Voyage AI (embeddings) · Anthropic Claude (labeling + reply generation) · FastAPI · Docker

## Author

Viktor Romenskiy — GenAI/ML engineer (RAG systems, LLM evaluation, fine-tuning). GitHub: [viktorromenskiy-glitch](https://github.com/viktorromenskiy-glitch) · LinkedIn: [profile](https://www.linkedin.com/in/%D0%B2%D0%B8%D0%BA%D1%82%D0%BE%D1%80-%D1%80%D0%BE%D0%BC%D0%B5%D0%BD%D1%81%D0%BA%D0%B8%D0%B9-029b2086/) · Hugging Face: [ViktorPetrov123](https://huggingface.co/ViktorPetrov123) · Contact: viktorromenskiy@gmail.com

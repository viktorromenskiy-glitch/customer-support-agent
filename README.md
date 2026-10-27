# Customer Support Agent — RAG + Classification + MLOps

*[Русская версия](README_ru.md) · [Українська версія](README_uk.md)*

**CSV triage tool (category, priority, pass-to-human flag): see [README_triage.md](README_triage.md).**

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
| Priority (4 levels) | Accuracy | 50.10% | **60.40%** |
| Priority (4 levels) | F1 macro | 0.482 | **0.594** |
| Semantic search | Category match in top-5 | 36.3% (TF-IDF) | **58.1%** |

All metrics are measured on a **temporal** train/test split (the earliest 80% of tickets for training and the knowledge base, the latest 20% as a held-out test), not a random split. This avoids evaluating on data that "peeks from the future" and matches the real use case: searching already-solved tickets to help with a new one. The test set is the last 4,000 tickets, which cover 12 calendar days (22 Nov - 3 Dec 2017). How the models behave in other periods or at other companies has not been checked.

Category and priority labels come from an LLM (Claude Haiku), not from manual labeling, because the source dataset has neither. This is a form of weak supervision / distillation: the LLM labels the whole dataset once, and a cheap, fast classical classifier (logistic regression) is trained on those labels for real-time use, instead of calling the LLM for every incoming ticket just to classify it. This also means that every accuracy figure here shows how closely the classifier agrees with the LLM labels, not how often it matches the decision of a human. A check on manually labeled tickets has not been done yet.

### Independent re-check (16 Oct 2026)

All numbers in the table above were recomputed from the saved embeddings and labels. They match, except for the points below.

- **TF-IDF, priority:** recomputing gives 50.10% (F1 0.482), not the 50.28% (0.484) reported earlier. The script that produced the original figure could not be found.
- **Search:** on all 4,000 test tickets the category matches in the top 5 in 58.6% of cases (95% CI 57.5-59.7%), using the saved embeddings (document type). The 58.1% above comes from a sample of 500 tickets embedded as queries, and those query embeddings were not saved. For comparison, random search gives 14.7% (using the real class shares; 12.5% would be right only if all 8 classes were equally common), and always returning neighbours from the most common category gives 19.8%.
- **Duplicates:** 55 of the 4,000 test tickets (1.4%) have an exact copy in the training data once @mentions and links are removed and case and spacing are ignored. Without them, category accuracy drops by about 0.3 percentage points. Reworded duplicates were not searched for.
- **Category accuracy:** 95% CI 69.1-71.9%. Embeddings beat TF-IDF by 10.1-13.1 points (paired test, p < 1e-50).
- **Feature removed:** the models used to take a `is_dm_escalation` flag that was computed from the company's reply, which the agent does not have when a new ticket arrives. It was removed. Without it the held-out results are the same: category 70.60% (F1 0.686), priority 60.42% (F1 0.594).

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

Then: `POST http://localhost:8000/handle_ticket` with `{"text": "...", "company": "AmazonHelp"}`.

### 5. Or run in Docker

```bash
docker build -t customer-support-agent .
docker run -e VOYAGE_API_KEY=... -e ANTHROPIC_API_KEY=... -p 8000:8000 customer-support-agent
```

Model artifacts (`.joblib`, `.npy`, `kb_with_index.csv`) are not stored in git (see `.gitignore`) because of their size. Generate them with the pipeline above, or use Git LFS to commit them directly.

## Tech stack

Python · scikit-learn · Voyage AI (embeddings) · Anthropic Claude (labeling + reply generation) · FastAPI · Docker

## Author

Viktor Romenskiy — GenAI/ML engineer (RAG systems, LLM evaluation, fine-tuning). GitHub: [viktorromenskiy-glitch](https://github.com/viktorromenskiy-glitch)

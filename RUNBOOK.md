# Runbook — from local files to a working, pushed repository

Follow this top to bottom. Every script name below matches exactly what's
in the repo structure described in README.md.

## Step 0 — Organize the folder

Create a fresh project folder (e.g. `customer-support-agent/`) with this layout,
copying/renaming the flat-named files you already have:

```
customer-support-agent/
├── data_pipeline/
│   ├── prepare_dataset.py       (was: data_pipeline_prepare_dataset.py)
│   ├── sample_and_filter.py     (was: data_pipeline_sample_and_filter.py)
│   ├── label_categories.py      (was: data_pipeline_label_categories.py)
│   └── label_priority.py        (was: data_pipeline_label_priority.py)
├── training/
│   ├── build_semantic_rag.py    (was: training_build_semantic_rag.py)
│   └── train_final_embedding_classifiers.py  (was: training_train_final_embedding_classifiers.py)
├── agent.py
├── app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
├── README.md
├── README_ru.md
└── README_uk.md
```

Also place `twcs.csv` (the raw Kaggle dataset) somewhere the pipeline scripts
can find it — either in the project root or in a `twcs/` subfolder (both are
auto-detected).

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
pip install langdetect  # used by sample_and_filter.py, not in requirements.txt (only needed once, at data-prep time)
```

## Step 2 — Data pipeline (no API keys needed yet)

```bash
cd data_pipeline
python prepare_dataset.py
python sample_and_filter.py
```

Check the printed language distribution and DM-escalation examples along the way —
worth a quick look before moving on, same as we did during development.

Produces: `twitter_support_final.csv` (~20,000 rows, English only, filtered and stratified).

## Step 3 — Labeling (needs `ANTHROPIC_API_KEY`, ~$2-4 total, ~50 min)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python label_categories.py    # ~25 min, produces twitter_support_labeled_categories.csv
python label_priority.py      # ~25 min, produces twitter_support_labeled_final.csv
```

If either script reports labeling errors at the end (should be well under 1%),
that's expected and fine — the error rows just got a placeholder label; not
worth a special fix pass unless the error rate looks unusually high.

## Step 4 — Semantic index + classifiers (needs `VOYAGE_API_KEY`, free tier, ~20-60 min depending on rate limit tier)

```bash
cd ../training
export VOYAGE_API_KEY="..."
python build_semantic_rag.py
```

This embeds the 16,000-record knowledge base and a 500-record test sample, and
reports the retrieval quality metric (should land close to 58% category-match
in top-5, matching what's in the README — if it's meaningfully different,
that's worth a second look before moving on).

**Important:** the training script also needs embeddings for the *full* 4,000-record
test set (not just the 500-sample used for the retrieval-quality check), saved as
`test_embeddings.npy`. Add this small script in `training/` to get them
(reuses the same rate-limit-aware `embed_batch` pattern):

```python
# training/embed_remaining_test_set.py
import re, time
import numpy as np
import pandas as pd
import voyageai

client = voyageai.Client()
MODEL = "voyage-4-lite"
BATCH_SIZE = 100

def clean(text):
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'https?://\S+', '', text)
    return text

df = pd.read_csv("../data_pipeline/twitter_support_labeled_final.csv")
df["request_created_at"] = pd.to_datetime(df["request_created_at"], errors="coerce", utc=True)
df = df.sort_values("request_created_at").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
test_df["text_clean"] = test_df["request_text"].apply(clean)

all_vecs = []
n_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE
for b in range(n_batches):
    batch = test_df["text_clean"].iloc[b*BATCH_SIZE:(b+1)*BATCH_SIZE].tolist()
    batch_safe = [t if t and t.strip() else "(empty)" for t in batch]
    for attempt in range(5):
        try:
            result = client.embed(batch_safe, model=MODEL, input_type="document")
            all_vecs.extend(result.embeddings)
            break
        except Exception as e:
            wait = 25 * (attempt + 1)
            print(f"  batch {b}, attempt {attempt+1}: waiting {wait}s...")
            time.sleep(wait)
    print(f"  {b+1}/{n_batches} batches")
    time.sleep(20)

np.save("test_embeddings.npy", np.array(all_vecs))
print(f"Saved {len(all_vecs)} test embeddings")
```

```bash
python embed_remaining_test_set.py
python train_final_embedding_classifiers.py
```

Check the printed accuracy/F1 — should be close to 70.6% / 0.686 (category)
and 60.4% / 0.594 (priority).

## Step 5 — Smoke-test the agent locally

```bash
cd ..
export VOYAGE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
python agent.py
```

Runs 5 real examples end-to-end and prints the full decision trace. Confirm
it completes without errors before moving on.

## Step 6 — Run the API (optional, but good to verify before Docker)

```bash
uvicorn app:app --reload
```

Test with: `curl -X POST http://localhost:8000/handle_ticket -H "Content-Type: application/json" -d "{\"text\": \"my package never arrived\", \"company\": \"AmazonHelp\"}"`

## Step 7 — Docker (optional)

```bash
docker build -t customer-support-agent .
docker run -e VOYAGE_API_KEY=$VOYAGE_API_KEY -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -p 8000:8000 customer-support-agent
```

## Step 8 — Push to GitHub

```bash
git init
git add .
git commit -m "Customer support agent: RAG + classification + MLOps, real Twitter data"
git remote add origin https://github.com/<your-username>/customer-support-agent.git
git push -u origin main
```

The `.gitignore` already excludes all the large model/data files (`.joblib`,
`.npy`, the various `twitter_support_*.csv`) - only code, README files,
Dockerfile, and requirements.txt get committed. That's intentional: the
repo documents and reproduces the pipeline; it doesn't need to ship the
20,000-row dataset or 65MB of embeddings to be a complete, honest portfolio piece.

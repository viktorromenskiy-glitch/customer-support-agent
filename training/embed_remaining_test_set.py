"""
Embeds the full 4,000-record time-based test set (not just the 500-record
sample used for the retrieval-quality check in build_semantic_rag.py).
Needed before train_final_embedding_classifiers.py can do an honest,
full-size evaluation.
"""
import re
import time
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

print(f"Embedding {len(test_df):,} test records...")
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
            print(f"  batch {b}, attempt {attempt+1}: waiting {wait}s... ({str(e)[:80]})")
            time.sleep(wait)
    if (b + 1) % 10 == 0:
        print(f"  {b+1}/{n_batches} batches")
    time.sleep(20)

np.save("test_embeddings.npy", np.array(all_vecs))
print(f"\nSaved {len(all_vecs):,} test embeddings to test_embeddings.npy")

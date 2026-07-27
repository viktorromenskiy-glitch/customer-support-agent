"""
LLM labeling of ticket category for all 20,000 records.

Two fixes applied after a pilot run on 800 records surfaced real issues:
temperature=0 (determinism - the default temperature occasionally caused
the model to transliterate category names into odd formats), and returning
a category NUMBER instead of the category text (transliteration becomes
physically impossible this way).
"""
import json
import time
import pandas as pd
from anthropic import Anthropic

client = Anthropic()

CATEGORIES = [
    "Delivery / order status",
    "Billing / refund",
    "Account access",
    "Quality complaint / compensation",
    "Technical issue / bug",
    "Service disruption",
    "Positive feedback / thanks",
    "General inquiry / other",
]

BATCH_SIZE = 25
MODEL = "claude-haiku-4-5-20251001"

df = pd.read_csv("../archive/twitter_support_final.csv")
print(f"Loaded: {len(df):,}")

system_prompt = f"""You classify customer support messages into one of these categories:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(CATEGORIES))}

You will be given a numbered list of customer messages. Return ONLY a valid JSON
array of integers, no explanation, with exactly as many elements as the input list -
the category number (1-8) for each message in order. Example response: [3, 1, 7, 2, ...]"""

all_labels = []
n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
start_time = time.time()

for b in range(n_batches):
    batch = df.iloc[b*BATCH_SIZE: (b+1)*BATCH_SIZE]
    user_msg = "\n".join(f"{i+1}. {row['request_text']}" for i, (_, row) in enumerate(batch.iterrows()))

    for attempt in range(3):  # up to 3 attempts per batch, in case of transient network issues
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=500,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            nums = json.loads(raw)
            if len(nums) != len(batch):
                raise ValueError(f"Expected {len(batch)} labels, got {len(nums)}")
            labels = [CATEGORIES[n - 1] if 1 <= n <= 8 else "NUMBER_ERROR" for n in nums]
            break
        except Exception as e:
            print(f"  Batch {b}, attempt {attempt + 1}: {e}")
            time.sleep(2)
    else:
        labels = ["ALL_ATTEMPTS_FAILED"] * len(batch)

    all_labels.extend(labels)
    if (b + 1) % 20 == 0 or b == n_batches - 1:
        elapsed = time.time() - start_time
        eta = elapsed / (b + 1) * (n_batches - b - 1)
        print(f"  Batch {b + 1}/{n_batches} ({elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min remaining)")
    time.sleep(0.3)

df["category_llm"] = all_labels[:len(df)]

errors = df["category_llm"].isin(["NUMBER_ERROR", "ALL_ATTEMPTS_FAILED"]).sum()
print(f"\nLabeling errors: {errors} ({errors / len(df) * 100:.2f}%)")
print("\nFinal distribution:")
print(df["category_llm"].value_counts())

df.to_csv("twitter_support_labeled_categories.csv", index=False)
print(f"\nSaved to twitter_support_labeled_categories.csv")
print(f"Total time: {(time.time() - start_time) / 60:.1f} min")

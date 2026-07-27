"""
LLM labeling of ticket priority for all 20,000 records.
Same reliable pattern as category labeling: numbers instead of text, temperature=0.
"""
import json
import time
import pandas as pd
from anthropic import Anthropic

client = Anthropic()

PRIORITIES = [
    "Critical",   # 1: service is down, direct financial harm right now, safety concern
    "High",       # 2: clear, serious problem, needs prompt attention
    "Medium",     # 3: ordinary inquiry, no particular urgency
    "Low",        # 4: general question, thanks, no urgent action needed
]

BATCH_SIZE = 25
MODEL = "claude-haiku-4-5-20251001"

df = pd.read_csv("../archive/twitter_support_labeled_categories.csv")
print(f"Loaded: {len(df):,}")

system_prompt = f"""You rate the priority of a customer support message on this scale:
1. {PRIORITIES[0]} - service is down, direct financial harm to the customer right now, safety concern
2. {PRIORITIES[1]} - a clear, serious problem, needs prompt attention (long unresolved, repeat contact, strong dissatisfaction)
3. {PRIORITIES[2]} - an ordinary inquiry with no particular urgency
4. {PRIORITIES[3]} - a general question, a thank-you, no urgent action needed

You will be given a numbered list of customer messages. Return ONLY a valid JSON
array of integers (1-4), no explanation, with exactly as many elements as the input list.
Example response: [3, 1, 4, 2, ...]"""

all_labels = []
n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
start_time = time.time()

for b in range(n_batches):
    batch = df.iloc[b*BATCH_SIZE: (b+1)*BATCH_SIZE]
    user_msg = "\n".join(f"{i+1}. {row['request_text']}" for i, (_, row) in enumerate(batch.iterrows()))

    for attempt in range(3):
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
            labels = [PRIORITIES[n - 1] if 1 <= n <= 4 else "NUMBER_ERROR" for n in nums]
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

df["priority_llm"] = all_labels[:len(df)]

errors = df["priority_llm"].isin(["NUMBER_ERROR", "ALL_ATTEMPTS_FAILED"]).sum()
print(f"\nLabeling errors: {errors} ({errors / len(df) * 100:.2f}%)")
print("\nFinal priority distribution:")
print(df["priority_llm"].value_counts())

df.to_csv("twitter_support_labeled_final.csv", index=False)
print(f"\nSaved to twitter_support_labeled_final.csv")
print(f"Total time: {(time.time() - start_time) / 60:.1f} min")

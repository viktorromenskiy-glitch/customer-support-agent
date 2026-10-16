"""
Measures how often drafts contain foreign links, sign-offs or @mentions.

Runs the same test tickets in three modes ("raw", "clean", "clean_filter") and
counts the drafts whose text, straight from the language model, contains a link,
sign-off or @mention. Needs the same keys as agent.py.

Usage (PowerShell, from the project root):
    python eval_draft_safety.py 20
    (20 = number of test tickets; each ticket costs 1 embedding + 3 language-model calls)

Writes eval_draft_safety_results.csv with every draft, so the numbers can be checked by hand.
Note: in "clean_filter" mode the final text is clean by construction, so the useful
number is how often the filter had to act ("defects_in_raw").
"""
import sys
import pandas as pd
import agent
from draft_safety import find_defects

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
df = pd.read_csv("archive/twitter_support_labeled_final.csv")
df["request_created_at"] = pd.to_datetime(df["request_created_at"], errors="coerce", utc=True)
df = df.sort_values("request_created_at").reset_index(drop=True)
test = df.iloc[int(len(df) * 0.8):].sample(n, random_state=11)

rows = []
for i, (_, row) in enumerate(test.iterrows(), 1):
    emb = agent.get_embedding(row["request_text"])
    retrieved = agent.retrieve_similar_from_embedding(emb, k=5)
    for mode in ("raw", "clean", "clean_filter"):
        det = agent.draft_response(row["request_text"], retrieved, mode=mode, return_details=True)
        codes = sorted({c for c, _ in det["defects_in_raw"]})
        rows.append({"ticket": i, "company": row["company"], "mode": mode,
                     "request": row["request_text"], "raw_draft": det["raw_draft"],
                     "final_draft": det["draft"], "defects_in_raw": ";".join(codes)})
    print(f"ticket {i}/{n} done")

out = pd.DataFrame(rows)
out.to_csv("eval_draft_safety_results.csv", index=False, encoding="utf-8-sig")
out["has_defect"] = out["defects_in_raw"] != ""
print("\nShare of drafts (straight from the language model) with a foreign link / sign-off / mention:")
print(out.groupby("mode")["has_defect"].agg(["sum", "count", "mean"]).round(3))
print("\nBy defect type:")
for code in ("foreign_url", "foreign_signature", "mention", "placeholder"):
    print(code, out.groupby("mode")["defects_in_raw"].apply(lambda s: s.str.contains(code).sum()).to_dict())
print("\nSaved eval_draft_safety_results.csv")

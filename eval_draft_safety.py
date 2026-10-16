"""
Measures how often drafts contain foreign links, sign-offs or @mentions.

Runs the same test tickets in three modes ("raw", "clean", "clean_filter") and
counts the drafts whose text, straight from the language model, contains a link,
sign-off or @mention. Needs the same keys as agent.py.

Usage (PowerShell, from the project root):
    python eval_draft_safety.py 20
    (20 = number of test tickets; each ticket costs 1 embedding + 3 language-model calls)

Results are saved after EVERY draft in eval_draft_safety_results.csv.
If the run stops (network, rate limit, Ctrl+C), run the same command again:
finished drafts are not repeated and not paid for twice. To start from scratch,
delete the csv file (or move it away) first. Use the same number of tickets
when resuming.

Note: in "clean_filter" mode the final text is clean by construction, so the useful
number is how often the filter had to act ("defects_in_raw").
"""
import csv
import os
import sys
import pandas as pd
import agent

OUT = "eval_draft_safety_results.csv"
FIELDS = ["ticket", "company", "mode", "request", "raw_draft", "final_draft", "defects_in_raw"]
MODES = ("raw", "clean", "clean_filter")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
df = pd.read_csv("archive/twitter_support_labeled_final.csv")
df["request_created_at"] = pd.to_datetime(df["request_created_at"], errors="coerce", utc=True, format="mixed")
df = df.sort_values("request_created_at").reset_index(drop=True)
test = df.iloc[int(len(df) * 0.8):].sample(n, random_state=11)

done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, encoding="utf-8-sig")
    done = set(zip(prev["ticket"], prev["mode"]))
    print(f"Resuming: {len(done)} drafts already saved in {OUT}")
else:
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

failed = 0
for i, (_, row) in enumerate(test.iterrows(), 1):
    todo = [m for m in MODES if (i, m) not in done]
    if not todo:
        continue
    try:
        emb = agent.get_embedding(row["request_text"])
        retrieved = agent.retrieve_similar_from_embedding(emb, k=5)
    except Exception as e:
        print(f"ticket {i}: embedding failed ({str(e)[:80]}) - skipped, run again to retry")
        failed += 1
        continue
    for mode in todo:
        try:
            det = agent.draft_response(row["request_text"], retrieved, mode=mode, return_details=True)
        except Exception as e:
            print(f"ticket {i} mode {mode}: failed ({str(e)[:80]}) - skipped, run again to retry")
            failed += 1
            continue
        codes = sorted({c for c, _ in det["defects_in_raw"]})
        with open(OUT, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({
                "ticket": i, "company": row["company"], "mode": mode, "request": row["request_text"],
                "raw_draft": det["raw_draft"], "final_draft": det["draft"], "defects_in_raw": ";".join(codes)})
    print(f"ticket {i}/{n} done")

out = pd.read_csv(OUT, encoding="utf-8-sig").fillna("")
out["has_defect"] = out["defects_in_raw"] != ""
print(f"\nDrafts saved: {len(out)} of {n * len(MODES)}" + (f"  ({failed} failed - run the same command again)" if failed else ""))
print("\nShare of drafts (straight from the language model) with a foreign link / sign-off / mention:")
print(out.groupby("mode")["has_defect"].agg(["sum", "count", "mean"]).round(3))
print("\nBy defect type:")
for code in ("foreign_url", "foreign_signature", "mention", "placeholder"):
    print(code, out.groupby("mode")["defects_in_raw"].apply(lambda s: s.str.contains(code).sum()).to_dict())
print(f"\nSaved {OUT}")

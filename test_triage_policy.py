"""Offline tests for triage.py (no network, no paid calls): fake embedding client + the saved classifiers."""
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import triage  # noqa: E402

triage.time.sleep = lambda s: None   # no waiting in tests
MODELS = sys.argv[1]
ok = fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print("FAIL:", name)


def vec(text):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=1024)
    return v / np.linalg.norm(v)


class Fake:
    def __init__(self, fail_from_call=None):
        self.calls = 0
        self.embedded = 0
        self.fail_from_call = fail_from_call

    def embed(self, texts, model, input_type):
        self.calls += 1
        if self.fail_from_call is not None and self.calls >= self.fail_from_call:
            raise ConnectionError("simulated outage")
        self.embedded += len(texts)

        class R:
            pass
        r = R()
        r.embeddings = [vec(t).tolist() for t in texts]
        r.total_tokens = 5 * len(texts)
        return r


work = tempfile.mkdtemp()
texts = [f"@AppleSupport my phone case number {i} stopped working after update {i * 7}" for i in range(40)]
texts[5] = ""            # empty
texts[6] = "@AppleSupport http://t.co/x"   # empty after cleaning
df = pd.DataFrame({"id": range(40), "text": texts,
                   "company": ["AppleSupport", "AmazonHelp", "Delta", "SpotifyCares", "Uber_Support"] * 8,
                   "extra": ["keep"] * 40})
inp = os.path.join(work, "in.csv")
df.to_csv(inp, index=False)

# direct prediction for the reference
cat_b, pri_b, _ = triage.load_models(MODELS)
live = [i for i in range(40) if triage.clean(texts[i])]
emb = np.array([vec(triage.clean(texts[i])) for i in live])
cats, cconf = triage.predict(emb, [df.company[i] for i in live], cat_b, triage.CATEGORY_RU_TO_EN)
pris, pconf = triage.predict(emb, [df.company[i] for i in live], pri_b, triage.PRIORITY_RU_TO_EN)
ref = {i: (cats[k], cconf[k], pris[k], pconf[k]) for k, i in enumerate(live)}

# 1. default policy: category only, 0.70
out = os.path.join(work, "o1.csv")
cl = Fake()
path = triage.run(inp, out, MODELS, batch_size=16, client=cl)
r = pd.read_csv(out, dtype=str, keep_default_na=False)
check("1 rows kept", len(r) == 40 and list(r["id"]) == [str(i) for i in range(40)])
check("1 extra column kept", (r["extra"] == "keep").all())
check("1 abs path returned", os.path.isabs(path))
check("1 partial removed", not os.path.exists(out + ".partial.csv"))
good = True
for i in live:
    row = r.iloc[i]
    good &= row["category"] == ref[i][0] and abs(float(row["category_confidence"]) - ref[i][1]) < 6e-5
    good &= row["priority"] == ref[i][2]
    want = "yes" if ref[i][1] < 0.70 else "no"
    good &= row["needs_human"] == want
    good &= row["needs_human_reason"] == ("low_category_confidence" if want == "yes" else "")
check("1 predictions and decisions equal the direct ones", good)
check("1 empty rows go to a human", all(r.iloc[i]["needs_human"] == "yes" and r.iloc[i]["needs_human_reason"] == "empty_text"
                                       and r.iloc[i]["category_confidence"] == "" for i in (5, 6)))
check("1 embedded only non-empty", cl.embedded == 38)
print("   share to human (random vectors, only a mechanics test):", (r["needs_human"] == "yes").mean())

# 2. priority threshold on top
out2 = os.path.join(work, "o2.csv")
triage.run(inp, out2, MODELS, batch_size=16, client=Fake(), min_category=0.70, min_priority=0.60)
r2 = pd.read_csv(out2, dtype=str, keep_default_na=False)
good = True
for i in live:
    reasons = []
    if ref[i][1] < 0.70:
        reasons.append("low_category_confidence")
    if ref[i][3] < 0.60:
        reasons.append("low_priority_confidence")
    good &= r2.iloc[i]["needs_human_reason"] == ";".join(reasons)
    good &= r2.iloc[i]["needs_human"] == ("yes" if reasons else "no")
check("2 priority threshold adds its own reason", good)

# 3. stopped run, restarted with other thresholds: all rows follow the final thresholds, paid once
out3 = os.path.join(work, "o3.csv")
cl3 = Fake(fail_from_call=3)
try:
    triage.run(inp, out3, MODELS, batch_size=16, client=cl3, min_category=0.50)
    check("3 outage must stop the run", False)
except RuntimeError:
    check("3 outage stops the run", True)
check("3 progress file kept", os.path.exists(out3 + ".partial.csv"))
saved_rows = len(pd.read_csv(out3 + ".partial.csv"))
cl3b = Fake()
triage.run(inp, out3, MODELS, batch_size=16, client=cl3b, min_category=0.80)
r3 = pd.read_csv(out3, dtype=str, keep_default_na=False)
good = all(r3.iloc[i]["needs_human"] == ("yes" if ref[i][1] < 0.80 else "no") for i in live)
check("3 every row follows the final threshold 0.80", good)
check("3 only missing rows were embedded again (paid once)", cl3b.embedded == 38 - (saved_rows - 2))
check("3 result complete", len(r3) == 40)

# 4. boundary: equal to the threshold is accepted, below goes to a human
check("4 equal accepted", triage.decide_human(0.70, 0.0, False, 0.70, None)[0] == "no")
check("4 just below goes", triage.decide_human(0.6999999, 0.0, False, 0.70, None)[0] == "yes")
check("4 priority ignored when None", triage.decide_human(0.9, 0.01, False, 0.70, None)[0] == "no")

# 5. invalid thresholds
for bad in (1.5, -0.1):
    try:
        triage.run(inp, os.path.join(work, "o5.csv"), MODELS, client=Fake(), min_category=bad)
        check(f"5 threshold {bad} rejected", False)
    except SystemExit:
        check(f"5 threshold {bad} rejected", True)

# 6. progress file from another format is not mixed in
out6 = os.path.join(work, "o6.csv")
pd.DataFrame(columns=["row"] + triage.OUT_COLUMNS).to_csv(out6 + ".partial.csv", index=False)
try:
    triage.run(inp, out6, MODELS, client=Fake())
    check("6 old progress file rejected", False)
except SystemExit as e:
    check("6 old progress file rejected", "unexpected format" in str(e))

# 7. dry run writes nothing
out7 = os.path.join(work, "o7.csv")
res = triage.run(inp, out7, MODELS, dry_run=True)
check("7 dry run", res is None and not os.path.exists(out7) and not os.path.exists(out7 + ".partial.csv"))

# 8. command line parsing
triage.main([inp, "--output", os.path.join(work, "o8.csv"), "--models-dir", MODELS, "--dry-run",
             "--min-category-confidence", "0.8", "--min-priority-confidence", "0.5"])

shutil.rmtree(work)
print(f"tests: {ok} passed, {fail} failed")
sys.exit(0 if fail == 0 else 1)

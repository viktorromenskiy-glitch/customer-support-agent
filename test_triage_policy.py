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
        assert model == triage.VOYAGE_MODEL and input_type == "document", (model, input_type)
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


# 9. a progress file must not be reused for another input file with the same number of rows
out9 = os.path.join(work, "o9.csv")
try:
    triage.run(inp, out9, MODELS, batch_size=16, client=Fake(fail_from_call=2))
except RuntimeError:
    pass
df2 = df.copy()
df2["text"] = [t + " (different ticket)" if t else t for t in df["text"]]
inp2 = os.path.join(work, "in_other.csv")
df2.to_csv(inp2, index=False)
try:
    triage.run(inp2, out9, MODELS, batch_size=16, client=Fake())
    check("9 other input with same progress file must be refused", False)
except SystemExit as e:
    check("9 other input with same progress file refused", "different input" in str(e))

# 10. embeddings that are not unit length stop the run before anything is saved
class NonUnit(Fake):
    def embed(self, texts, model, input_type):
        r = super().embed(texts, model, input_type)
        r.embeddings = [[2 * x for x in e] for e in r.embeddings]
        return r
out10 = os.path.join(work, "o10.csv")
try:
    triage.run(inp, out10, MODELS, client=NonUnit())
    check("10 non-unit embeddings refused", False)
except RuntimeError:
    check("10 non-unit embeddings refused", True)
check("10 nothing saved from the bad batch", len(pd.read_csv(out10 + ".partial.csv")) == 0)

# 11. confidence survives saving exactly (no flip at the threshold)
x = 0.69999999999
check("11 repr round trip exact", float(repr(x)) == x)
check("11 value just below threshold goes to a human", triage.decide_human(float(repr(x)), 0.0, False, 0.70, None)[0] == "yes")

# 12. final check catches a blank category for a non-empty ticket
orig_predict = triage.predict
def broken_predict(emb, comps, bundle, m):
    labels, conf = orig_predict(emb, comps, bundle, m)
    labels[0] = ""
    return labels, conf
triage.predict = broken_predict
try:
    triage.run(inp, os.path.join(work, "o12.csv"), MODELS, batch_size=64, client=Fake())
    check("12 blank category caught", False)
except SystemExit as e:
    check("12 blank category caught", "INCOMPLETE" in str(e) or "empty category" in str(e))
finally:
    triage.predict = orig_predict

# 13. input and parameter guards
def refused(fn, needle):
    try:
        fn()
        return False
    except SystemExit as e:
        return needle in str(e)
check("13 batch size 0", refused(lambda: triage.run(inp, os.path.join(work, "o13.csv"), MODELS, batch_size=0, client=Fake()), "batch-size"))
check("13 output equals input", refused(lambda: triage.run(inp, inp, MODELS, client=Fake()), "must not be the input"))
check("13 category threshold None", refused(lambda: triage.run(inp, os.path.join(work, "o13b.csv"), MODELS, client=Fake(), min_category=None), "required"))
empty_in = os.path.join(work, "empty.csv")
pd.DataFrame({"text": []}).to_csv(empty_in, index=False)
check("13 empty file, also in dry run", refused(lambda: triage.run(empty_in, dry_run=True), "no rows"))
bad_enc = os.path.join(work, "latin.csv")
with open(bad_enc, "wb") as f:
    f.write("text\ncaf\xe9 problem\n".encode("latin-1"))
check("13 not UTF-8 file", refused(lambda: triage.run(bad_enc, dry_run=True), "UTF-8"))
corrupt = os.path.join(work, "o13c.csv")
with open(corrupt + ".partial.csv", "w", encoding="utf-8") as f:
    f.write('row,text_hash\n"1,2\n')
check("13 damaged progress file", refused(lambda: triage.run(inp, corrupt, MODELS, client=Fake()), "damaged"))

# 14. duplicated ids give a warning, blank-space company equals the default company
import io, contextlib
dup = df.copy()
dup["id"] = 1
dup_in = os.path.join(work, "dup.csv")
dup.to_csv(dup_in, index=False)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    triage.run(dup_in, os.path.join(work, "o14.csv"), MODELS, client=Fake())
check("14 duplicated id warning", "duplicated values in the 'id' column" in buf.getvalue())
sp = pd.DataFrame({"text": ["my order is late please help"] * 2, "company": ["   ", "unknown"]})
sp_in = os.path.join(work, "sp.csv")
sp.to_csv(sp_in, index=False)
triage.run(sp_in, os.path.join(work, "o14b.csv"), MODELS, client=Fake())
rs = pd.read_csv(os.path.join(work, "o14b.csv"), dtype=str, keep_default_na=False)
check("14 blank company treated as the default company", rs.loc[0, "category_confidence"] == rs.loc[1, "category_confidence"])

# 15. separators, column names, encoding (v4)
base = pd.DataFrame({"id": [1, 2, 3], "text": ["my order is late, please help; thanks", "app crashes\nsecond line", 'he said "refund"'],
                     "company": ["Delta", "Delta", "Delta"]})
def run_quiet(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return triage.run(*a, **k)
ref_out = os.path.join(work, "o15ref.csv")
run_quiet(inp, ref_out, MODELS, client=Fake())
for name, sep in (("semi", ";"), ("tab", "\t")):
    fi = os.path.join(work, f"{name}.csv")
    base.to_csv(fi, index=False, sep=sep, encoding="utf-8-sig")
    fo = os.path.join(work, f"{name}_out.csv")
    run_quiet(fi, fo, MODELS, client=Fake())
    rr = pd.read_csv(fo, dtype=str, keep_default_na=False, sep=sep, encoding="utf-8-sig")
    check(f"15 {name} separator kept in output", len(rr) == 3 and "needs_human" in rr.columns and list(rr["id"]) == ["1", "2", "3"])
    check(f"15 {name} text with comma/newline/quote intact", list(rr["text"]) == list(base["text"]))
weird = base.rename(columns={"text": " Text ", "company": "Company "})
wi = os.path.join(work, "weird.csv"); weird.to_csv(wi, index=False)
wo = os.path.join(work, "weird_out.csv")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    triage.run(wi, wo, MODELS, client=Fake())
rw = pd.read_csv(wo, dtype=str, keep_default_na=False)
check("15 names with spaces/case accepted, own names kept", list(rw.columns[:3]) == [" Text ", "Company ", "id"] or " Text " in rw.columns)
check("15 'Company ' recognised (not silently unknown)", "Company 'Company '" in buf.getvalue() or "Company column: 'Company '" in buf.getvalue())
nc = os.path.join(work, "nocomp.csv"); base[["text"]].to_csv(nc, index=False)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    triage.run(nc, os.path.join(work, "nocomp_out.csv"), MODELS, client=Fake())
check("15 notice when no company column", "Company column: none" in buf.getvalue())
dupn = os.path.join(work, "dupn.csv")
with open(dupn, "w", encoding="utf-8") as f:
    f.write("text, Text\na,b\n")
check("15 duplicate names after normalisation refused", refused(lambda: triage.run(dupn, dry_run=True), "same"))
clash = os.path.join(work, "clash.csv")
pd.DataFrame({"text": ["a"], " Category ": ["x"]}).to_csv(clash, index=False)
check("15 output column clash ignoring case/space", refused(lambda: triage.run(clash, dry_run=True), "overwrite"))
cp = os.path.join(work, "cp1251.csv")
with open(cp, "wb") as f:
    f.write("text\nпроблема с заказом\n".encode("cp1251"))
check("15 cp1251 without option gives hint", refused(lambda: triage.run(cp, dry_run=True), "--encoding"))
cpo = os.path.join(work, "cp_out.csv")
run_quiet(cp, cpo, MODELS, client=Fake(), encoding="cp1251")
check("15 cp1251 with option works, output UTF-8", pd.read_csv(cpo, dtype=str, encoding="utf-8-sig").loc[0, "text"] == "проблема с заказом")

# 16. retries: only temporary errors
class Http(Exception):
    def __init__(self, status):
        super().__init__(f"http {status}")
        self.http_status = status
class AuthenticationError(Exception):
    pass
class Flaky(Fake):
    def __init__(self, err, times):
        super().__init__(); self.err = err; self.times = times; self.tries = 0
    def embed(self, texts, model, input_type):
        self.tries += 1
        if self.tries <= self.times:
            raise self.err
        return super().embed(texts, model, input_type)
for err, retriable in ((Http(429), True), (Http(503), True), (Http(401), False), (Http(400), False),
                       (Http(422), False), (AuthenticationError("bad key"), False), (ConnectionError("net"), True)):
    cl = Flaky(err, 1)
    try:
        triage.embed_batch(cl, ["hello there"])
        got = cl.tries == 2
    except RuntimeError:
        got = cl.tries == 1
    check(f"16 {type(err).__name__} {getattr(err, 'http_status', '')} retriable={retriable}", got == True and (cl.tries == 2) == retriable)
cl = Flaky(Http(500), 99)
try:
    triage.embed_batch(cl, ["x"], max_attempts=3)
    check("16 gives up after max attempts", False)
except RuntimeError:
    check("16 gives up after max attempts", cl.tries == 3)

# 17. delimiter detection ignores separators inside quotes
def delim(content, name):
    fp = os.path.join(work, name)
    with open(fp, "w", encoding="utf-8") as f:
        f.write(content)
    return triage.detect_delimiter(fp, "utf-8-sig")
check("17 ; file, comma inside quoted header name", delim('text;"notes, important"\nhi;x\n', "d1.csv") == ";")
check("17 ; file, two commas in quoted name", delim('text;"a, b, c"\nhi;x\n', "d2.csv") == ";")
check("17 tab file, comma in name", delim("text\tnotes, x\nhi\tz\n", "d3.csv") == "\t")
check("17 comma file, semicolon inside quoted name", delim('text,"a;b;c;d"\nhi,x\n', "d4.csv") == ",")
check("17 comma file plain", delim("id,text,company\n1,a,b\n", "d5.csv") == ",")
check("17 single column", delim("text\nhello; world, yes\n", "d6.csv") == ",")
d7 = os.path.join(work, "d7.csv")
with open(d7, "w", encoding="utf-8") as f:
    f.write('text;"notes, important"\nhello there;x, y\n')
r7, sep7, tc7, _, _ = triage.read_input(d7)
check("17 quoted-comma header parsed as 2 columns", sep7 == ";" and list(r7.columns) == ["text", "notes, important"] and r7.loc[0, "notes, important"] == "x, y")

# 18. checks added after the review of version 5
import copy
# 18a. empty / zero-byte input
z = os.path.join(work, "zero.csv"); open(z, "w").close()
check("18 zero-byte file", refused(lambda: triage.run(z, dry_run=True), "empty"))
nl = os.path.join(work, "nl.csv"); open(nl, "w").write("\n\n")
check("18 only line breaks", refused(lambda: triage.run(nl, dry_run=True), "empty"))
# 18b. malformed records: unquoted line break, missing and extra separators
bad1 = os.path.join(work, "bad1.csv"); open(bad1, "w", encoding="utf-8").write("id,text,company\n1,Hello\nworld,AppleSupport\n")
check("18 unquoted line break refused", refused(lambda: triage.run(bad1, dry_run=True), "fields like the header"))
bad2 = os.path.join(work, "bad2.csv"); open(bad2, "w", encoding="utf-8").write("id,text,company\n1,a,b,c,d\n")
check("18 extra separators refused", refused(lambda: triage.run(bad2, dry_run=True), "fields like the header"))
good = os.path.join(work, "good_q.csv"); open(good, "w", encoding="utf-8").write('id,text,company\n1,"Hello\nworld",Delta\n\n2,x,Delta\n')
check("18 quoted line break and blank line accepted", triage.read_input(good)[0].shape[0] == 2)
# 18c. model files that do not fit
cat_b2, pri_b2, _ = triage.load_models(MODELS)
check("18 matching models accepted", triage.check_models(cat_b2, pri_b2) is None)
broken = copy.deepcopy(pri_b2)
broken["classifier"].n_features_in_ = broken["classifier"].n_features_in_ + 1
check("18 priority model with wrong size refused", refused(lambda: triage.check_models(cat_b2, broken), "priority model"))
check("18 model dict without ohe refused", refused(lambda: triage.check_models({"classifier": 1}, pri_b2), "category model"))
# 18d. progress file with values the script could not have written
live_text = "my order is late please help me now"
src = os.path.join(work, "src18.csv"); pd.DataFrame({"text": [live_text] * 3}).to_csv(src, index=False)
ok_out = os.path.join(work, "ok18.csv"); run_quiet(src, ok_out, MODELS, client=Fake())
mv = pd.read_csv(ok_out, dtype=str)["model_version"][0]
hh = triage.row_hash(triage.clean(live_text), "unknown")
def partial_case(tag, **over):
    row = dict(row="0", text_hash=hh, category="Order status", category_confidence="0.9", priority="Low",
               priority_confidence="0.9", empty_text="0", model_version=mv)
    row.update(over)
    o = os.path.join(work, f"p18_{tag}.csv")
    pd.DataFrame([row]).to_csv(o + ".partial.csv", index=False)
    return refused(lambda: triage.run(src, o, MODELS, client=Fake()), "damaged")
check("18 row not a number", partial_case("a", row="abc"))
check("18 negative row", partial_case("b", row="-1"))
check("18 confidence not a number", partial_case("c", category_confidence="abc"))
check("18 confidence nan", partial_case("d", category_confidence="nan"))
check("18 confidence above 1", partial_case("e", priority_confidence="1.5"))
check("18 empty_text value", partial_case("f", empty_text="maybe"))
check("18 empty category for a real ticket", partial_case("g", category=""))
# 18e. service returns the wrong number of embeddings
class Empty(Fake):
    def embed(self, texts, model, input_type):
        r = super().embed(texts, model, input_type); r.embeddings = []; return r
class Short(Fake):
    def embed(self, texts, model, input_type):
        r = super().embed(texts, model, input_type); r.embeddings = r.embeddings[:-1]; return r
for cls in (Empty, Short):
    try:
        run_quiet(src, os.path.join(work, f"e18_{cls.__name__}.csv"), MODELS, client=cls())
        check(f"18 {cls.__name__} embeddings refused", False)
    except RuntimeError as e:
        check(f"18 {cls.__name__} embeddings refused", "expected" in str(e))
# 18f. only one run per output file
lk = os.path.join(work, "lk18.csv")
open(lk + ".lock", "w").write("123")
check("18 existing lock refuses a second run", refused(lambda: triage.run(src, lk, MODELS, client=Fake()), "Another run"))
check("18 lock of the other run is left alone", os.path.exists(lk + ".lock") and not os.path.exists(lk))
os.remove(lk + ".lock")
run_quiet(src, lk, MODELS, client=Fake())
check("18 lock removed after a good run", not os.path.exists(lk + ".lock"))
lk2 = os.path.join(work, "lk18b.csv")
try:
    run_quiet(src, lk2, MODELS, batch_size=1, client=Fake(fail_from_call=2))
except RuntimeError:
    pass
check("18 lock removed after a failed run", not os.path.exists(lk2 + ".lock") and os.path.exists(lk2 + ".partial.csv"))
run_quiet(src, lk2, MODELS, batch_size=1, client=Fake())
check("18 failed run can be resumed", len(pd.read_csv(lk2, dtype=str)) == 3)
# 18g. dry run does not take the lock
run_quiet(src, os.path.join(work, "dry18.csv"), MODELS, dry_run=True)
check("18 dry run leaves no lock", not os.path.exists(os.path.join(work, "dry18.csv.lock")))
# 18h. separator tie between ';' and tab goes to ';'
tie = os.path.join(work, "tie18.csv"); open(tie, "w").write("a;b\tc\n1;2\t3\n")
check("18 tie semicolon vs tab", triage.detect_delimiter(tie, "utf-8-sig") == ";")

# 19. checks added after the review of version 6
# 19a. an existing result file is not replaced silently
ow = os.path.join(work, "ow19.csv")
run_quiet(src, ow, MODELS, client=Fake())
before = open(ow, "rb").read()
cl19 = Fake()
check("19 repeated command refused", refused(lambda: triage.run(src, ow, MODELS, client=cl19), "already exists"))
check("19 refused run changed nothing and embedded nothing", open(ow, "rb").read() == before and cl19.embedded == 0
      and not os.path.exists(ow + ".lock") and not os.path.exists(ow + ".partial.csv"))
check("19 dry run also refuses", refused(lambda: triage.run(src, ow, MODELS, dry_run=True), "already exists"))
cl19b = Fake()
run_quiet(src, ow, MODELS, client=cl19b, overwrite=True)
check("19 --overwrite runs again", cl19b.embedded == 3 and os.path.exists(ow))
# resume of a stopped run is still allowed even if an older result file is there
ow2 = os.path.join(work, "ow19b.csv")
open(ow2, "w").write("old result")
try:
    run_quiet(src, ow2, MODELS, batch_size=1, client=Fake(fail_from_call=2), overwrite=True)
except RuntimeError:
    pass
cl19c = Fake()
run_quiet(src, ow2, MODELS, batch_size=1, client=cl19c)
check("19 stopped run resumes without --overwrite and replaces the old file", cl19c.embedded == 2 and len(pd.read_csv(ow2)) == 3)
# command line
ow3 = os.path.join(work, "ow19c.csv"); open(ow3, "w").write("x")
check("19 CLI refuses existing output", refused(lambda: triage.main([src, "--output", ow3, "--models-dir", MODELS, "--dry-run"]), "already exists"))
triage.main([src, "--output", ow3, "--models-dir", MODELS, "--dry-run", "--overwrite"])
check("19 CLI --overwrite accepted", True)

# 19b. progress file: duplicated rows and empty_text that does not match the input
def partial_rows(tag, rows):
    o = os.path.join(work, f"q19_{tag}.csv")
    pd.DataFrame(rows).to_csv(o + ".partial.csv", index=False)
    return o
def prow(**over):
    row = dict(row="0", text_hash=hh, category="Order status", category_confidence="0.9", priority="Low",
               priority_confidence="0.9", empty_text="0", model_version=mv)
    row.update(over); return row
o = partial_rows("dup", [prow(), prow(category_confidence="0.8")])
check("19 duplicated row numbers refused", refused(lambda: triage.run(src, o, MODELS, client=Fake()), "more than once"))
o = partial_rows("emptymark", [prow(empty_text="1", category="", category_confidence="", priority="", priority_confidence="")])
check("19 non-empty ticket marked empty refused", refused(lambda: triage.run(src, o, MODELS, client=Fake()), "empty_text does not match"))
src_e = os.path.join(work, "src19e.csv"); pd.DataFrame({"text": ["@x http://t.co/a", live_text]}).to_csv(src_e, index=False)
h_empty = triage.row_hash("", "unknown")
o = partial_rows("emptyreal", [prow(text_hash=h_empty)])
check("19 empty ticket marked as real refused", refused(lambda: triage.run(src_e, o, MODELS, client=Fake()), "empty_text does not match"))

# 19c. long texts are cut before they are sent
class Rec(Fake):
    def __init__(self):
        super().__init__(); self.longest = 0
    def embed(self, texts, model, input_type):
        self.longest = max([self.longest] + [len(x) for x in texts])
        return super().embed(texts, model, input_type)
src_l = os.path.join(work, "src19l.csv"); pd.DataFrame({"text": ["word " * 20000, live_text]}).to_csv(src_l, index=False)
rec = Rec(); buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    triage.run(src_l, os.path.join(work, "l19.csv"), MODELS, client=rec)
check("19 long text cut to the limit", 0 < rec.longest <= triage.MAX_TEXT_CHARS)
check("19 number of cut texts printed", "1 text(s) are longer" in buf.getvalue())
rl = pd.read_csv(os.path.join(work, "l19.csv"), dtype=str, keep_default_na=False)
check("19 long text still gets a result", rl.loc[0, "category"] != "" and len(rl) == 2)

# 19d. models with classes this script does not know
cat19, pri19, _ = triage.load_models(MODELS)
bad_c = copy.deepcopy(cat19); bad_c["classifier"].classes_ = np.array(["Something else"] + list(bad_c["classifier"].classes_[1:]))
check("19 unknown category class refused", refused(lambda: triage.check_models(bad_c, pri19), "classes this script does not"))
bad_p = copy.deepcopy(pri19); bad_p["classifier"].classes_ = np.array(["Urgent"] + list(bad_p["classifier"].classes_[1:]))
check("19 unknown priority class refused", refused(lambda: triage.check_models(cat19, bad_p), "classes this script does not"))

# 19e. the real exception classes of the embedding client library
try:
    import voyageai.error as ve
    for name, want in (("AuthenticationError", False), ("InvalidRequestError", False), ("MalformedRequestError", False),
                       ("RateLimitError", True), ("ServerError", True), ("ServiceUnavailableError", True),
                       ("Timeout", True), ("APIConnectionError", True)):
        check(f"19 real {name} retriable={want}", triage.is_retriable(getattr(ve, name)("x")) == want)
except ImportError:
    print("   (voyageai not installed: real exception classes not tested)")

# 19f. confidence just below the threshold survives saving, resuming and writing
o = partial_rows("edge", [prow(category_confidence=repr(0.69999999999))])
cl19e = Fake()
run_quiet(src, o, MODELS, client=cl19e)
re19 = pd.read_csv(o, dtype=str, keep_default_na=False)
check("19 edge value: sent to a human, shown as 0.7000, other rows embedded", re19.loc[0, "needs_human"] == "yes"
      and re19.loc[0, "category_confidence"] == "0.7000" and cl19e.embedded == 2)

shutil.rmtree(work)
print(f"tests: {ok} passed, {fail} failed")
sys.exit(0 if fail == 0 else 1)

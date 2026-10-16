"""
Draft safety: keep foreign links, agent sign-offs and ticket mentions from past
replies out of the draft that is shown to a human.

Pure Python (no API keys, no model files), so it can be tested on its own:
    python draft_safety.py          # runs the built-in self-tests

Three layers, each usable alone:
1. scrub_precedent(text)  - clean a past reply BEFORE it goes into the prompt.
2. find_defects(draft)    - list leftover links / sign-offs / mentions in a draft.
3. sanitize_draft(draft)  - remove those from a draft and (optionally) add the
                            client's own approved link and signature.

Forms seen in the Twitter support data (checked on 20,000 replies):
  links     https://t.co/XXXX           (47% of replies)
  sign-offs ^XX  *XXX  /XX              (by company: Amazon 97%, Delta 78%, Spotify 72%)
  mentions  @123456 at the start        (99.97% of replies, anonymised ids)
Known limits (not handled): first names inside the text ("Hey Vishal"), and
sign-offs written in other styles. A human still reviews every draft.
"""
import re
from dataclasses import dataclass, field

# --- patterns -------------------------------------------------------------
_TAIL = r"[^\s]*[^\s.,;:!?)\]\"']"   # a URL does not end with sentence punctuation
URL_RE = re.compile(r"(?:https?://|www\.)" + _TAIL + r"|\b\w[\w-]*\.(?:com|co|ly|me|net|org|io)/" + _TAIL, re.I)
# Sign-off: ^AB  *ABC  /AB  ~AB at the very end (1-4 capital letters).
SIGNOFF_RE = re.compile(r"(?<!\w)[\^*/~]\s?[A-Z]{1,4}\s*$")
# sign-off followed by a link that was replaced by the placeholder: "... /DQ [LINK]"
SIGNOFF_BEFORE_LINK_RE = re.compile(r"(?<!\w)[\^*/~]\s?[A-Z]{1,4}\s+\[LINK\]\s*$")
MENTION_RE = re.compile(r"(?<!\w)@\[?\w+\]?")   # also "@[CustomerHandle]"
# Unfilled template slots such as "[Customer]" or "[Order Number]" ([LINK] is our own marker, handled apart).
TEMPLATE_RE = re.compile(r"\[(?!LINK\])[A-Za-z][A-Za-z _-]{1,30}\]")
# A trailing punctuation mark glued to a removed link ("here: .") is tidied after.
TIDY_SPACES = re.compile(r"[ \t]{2,}")
TIDY_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
PLACEHOLDER = "[LINK]"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class DraftPolicy:
    """What is allowed in a draft. Defaults: nothing foreign is allowed."""
    approved_urls: list = field(default_factory=list)   # the client's own links
    signature: str = ""                                  # the client's own sign-off, e.g. "Support Team"


def _norm_url(u):
    return u.rstrip(".,;:!?)\"'").lower()


def scrub_precedent(text):
    """Layer 1: replace links with [LINK], remove sign-offs and @mentions from a past reply."""
    t = str(text)
    t = URL_RE.sub(PLACEHOLDER, t)      # a typed placeholder keeps the sentence readable
    t = MENTION_RE.sub("", t)
    t = SIGNOFF_BEFORE_LINK_RE.sub(PLACEHOLDER, t.rstrip())
    t = SIGNOFF_RE.sub("", t.rstrip())
    t = TIDY_BEFORE_PUNCT.sub(r"\1", t)
    t = TIDY_SPACES.sub(" ", t)
    return t.strip()


def find_defects(draft, policy=None):
    """Layer 2: return a list of (code, text) for things in the draft that are not allowed."""
    policy = policy or DraftPolicy()
    allowed = {_norm_url(u) for u in policy.approved_urls}
    out = []
    for m in URL_RE.finditer(draft):
        if _norm_url(m.group(0)) not in allowed:
            out.append(("foreign_url", m.group(0)))
    if PLACEHOLDER in draft:
        out.append(("placeholder", PLACEHOLDER))
    for m in TEMPLATE_RE.finditer(MENTION_RE.sub("", draft)):
        out.append(("template_slot", m.group(0)))
    sig = SIGNOFF_RE.search(draft.rstrip())
    if sig and sig.group(0).strip() != policy.signature.strip():
        out.append(("foreign_signature", sig.group(0).strip()))
    for m in MENTION_RE.finditer(draft):
        out.append(("mention", m.group(0)))
    return out


SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s|$)|$)\s*", re.S)   # a sentence plus its trailing whitespace


def sanitize_draft_ex(draft, policy=None):
    """Layer 3: remove disallowed elements; add the client's own signature if set.
    A sentence that carries a foreign link or a leftover [LINK] is dropped as a whole
    (cutting only the link leaves broken sentences such as "reach out to us here:").
    [LINK] becomes the client's first approved link when there is one.
    Original line breaks and lists are kept.
    Returns (clean_draft, defects_found, dropped_sentences)."""
    policy = policy or DraftPolicy()
    defects = find_defects(draft, policy)
    allowed = {_norm_url(u) for u in policy.approved_urls}
    t = MENTION_RE.sub("", draft)
    t = TEMPLATE_RE.sub("", t)
    t = t.rstrip()
    sig = SIGNOFF_RE.search(t)
    if sig and sig.group(0).strip() != policy.signature.strip():
        t = t[:sig.start()].rstrip()
    if policy.approved_urls:
        t = t.replace(PLACEHOLDER, policy.approved_urls[0])
    kept, dropped = [], []
    for m in SENTENCE_RE.finditer(t):
        sent = m.group(0)
        bad_url = any(_norm_url(u.group(0)) not in allowed for u in URL_RE.finditer(sent))
        if bad_url or PLACEHOLDER in sent:
            dropped.append(sent.strip())
        else:
            kept.append(sent)
    t = TIDY_SPACES.sub(" ", "".join(kept)).strip()
    t = TIDY_BEFORE_PUNCT.sub(r"\1", t)
    if policy.signature and t and not t.endswith(policy.signature):
        t = f"{t}\n{policy.signature}"
    return t, defects, dropped


def sanitize_draft(draft, policy=None):
    """Same as sanitize_draft_ex, returns (clean_draft, defects_found)."""
    t, defects, _ = sanitize_draft_ex(draft, policy)
    return t, defects


def needs_review(draft, dropped, min_chars=80):
    """True when the filter removed a sentence or what is left is very short:
    the draft may have lost its call to action and a person should look at it."""
    return bool(dropped) or len(draft.strip()) < min_chars


# --- self-tests -----------------------------------------------------------
def _selftest():
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print("FAIL:", name)

    # scrub_precedent: removes what it should
    s = scrub_precedent("@123 Check this: https://t.co/AbC123. We can help. ^BN")
    check("scrub link+sig+mention", "t.co" not in s and "^BN" not in s and "@123" not in s)
    check("scrub link becomes placeholder", "[LINK]" in s)
    check("scrub keeps text", "We can help" in s)
    check("scrub star sign-off", not scrub_precedent("Thanks for flying with us. *ALY").endswith("*ALY"))
    check("scrub slash sign-off", scrub_precedent("Keep us posted /AR") == "Keep us posted")
    # does not damage normal text
    for ok_text in [
        "Go to Settings > Account and tap Reset password.",
        "Refunds take 3-5 business days (e.g. Monday to Friday).",
        "Your order total is 5/10 items shipped.",
        "We are open 9 a.m. to 5 p.m. U.S. time.",
        "Use option A/B or C. Thanks!",
    ]:
        check("keeps: " + ok_text, scrub_precedent(ok_text) == ok_text)
    # find_defects / sanitize on the observed bad draft
    bad = ("I'm sorry to hear your package hasn't arrived! Please reach out to us here: "
           "https://t.co/hApLpMlfHN and we can investigate further. ^BN")
    d = find_defects(bad)
    check("defects url", any(c == "foreign_url" for c, _ in d))
    check("defects sig", any(c == "foreign_signature" for c, _ in d))
    clean, _ = sanitize_draft(bad)
    check("sanitize removes link", "t.co" not in clean)
    check("sanitize removes sig", "^BN" not in clean)
    check("sanitize result clean", find_defects(clean) == [])
    check("sanitize keeps first sentence", clean.startswith("I'm sorry to hear"))
    out2, _ = sanitize_draft("We checked. Please see [LINK] for steps. Thanks!")
    check("placeholder sentence dropped", out2 == "We checked. Thanks!")
    # approved link and signature are kept / added
    pol = DraftPolicy(approved_urls=["https://example.com/help"], signature="Example Support")
    out, defects = sanitize_draft("See https://example.com/help for steps.", pol)
    out3, _ = sanitize_draft("Please see [LINK] for steps.", pol)
    check("placeholder -> approved url", "https://example.com/help" in out3)
    check("approved url kept", "https://example.com/help" in out)
    check("approved: no defects", defects == [])
    check("signature added", out.endswith("Example Support"))
    out, _ = sanitize_draft("Visit https://t.co/xx. ^BN", pol)
    check("only-link draft becomes empty (goes to a human)", out == "")
    out, _ = sanitize_draft("We are sorry for the delay. Visit https://t.co/xx. ^BN", pol)
    check("foreign replaced by own sig", "^BN" not in out and out.endswith("Example Support") and "t.co" not in out)
    # empty / edge
    check("glued sign-off", scrub_precedent("Happy to help.^SQ") == "Happy to help.")
    check("sign-off before link", "/DQ" not in scrub_precedent("We'll check it out /DQ https://t.co/ab"))
    check("A/B kept", scrub_precedent("Try A/B") == "Try A/B")
    # bracketed template slots and mentions
    check("template mention flagged", any(c == "mention" for c, _ in find_defects("@[CustomerHandle] Thanks for writing.")))
    check("template mention removed", sanitize_draft("@[Customer] Thanks for writing.")[0] == "Thanks for writing.")
    check("template slot flagged", any(c == "template_slot" for c, _ in find_defects("Hi [Customer Name], thanks.")))
    check("[LINK] not a template slot", not any(c == "template_slot" for c, _ in find_defects("See [LINK].")))
    # list formatting survives
    lst = "Please send:\n\n- Your order number?\n- The date?\n\nThanks!"
    check("list kept", sanitize_draft(lst)[0] == lst)
    # review flag
    t2, _, dr = sanitize_draft_ex("We understand your frustration! Please reach out at https://t.co/x with details.")
    check("dropped reported", len(dr) == 1 and needs_review(t2, dr))
    t3, _, dr3 = sanitize_draft_ex("We are sorry for the delay on your order. Could you share the order number so we can check?")
    check("no review when nothing removed", not needs_review(t3, dr3))
    check("empty draft", sanitize_draft("")[0] == "")
    check("no defects on plain text", find_defects("We are sorry for the delay. Please share your order number.") == [])
    print(f"self-tests: {ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)

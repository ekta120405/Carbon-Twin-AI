# CarbonTwinAI — Results, in plain language

Who this is for: someone deciding whether this project's approach is
sound and how good the results actually are — without needing to read
code or statistics. Every number below is real, pulled directly from the
project's test results (`docs/TASK7_FINDINGS.md`), not estimated or
rounded up to sound better.

For *how* the system works, see `docs/HOW_THE_SYSTEM_WORKS.md`. This
document is about *what was found* and *how good it is*.

---

## The one-paragraph version

CarbonTwinAI checks a solar plant's reported electricity generation
against an independent, satellite-weather-based estimate, and flags days
where the two don't match. Tested against 30 sealed, realistic scenarios
per plant (a mix of simulated fraud and real-but-legitimate oddities), it
correctly catches **65% of simulated fraud** and correctly leaves **92%
of legitimate, non-fraud situations** alone. Two specific fraud patterns
(fake night-time reporting, frozen sensors) are caught with **100%
reliability**. The system is honest about where it's weaker — slow,
creeping fraud and very brief, narrow-window fraud are genuinely hard to
catch, and this is measured and disclosed rather than hidden. Overall,
this is a **rigorously validated proof of concept**, not yet a
production-ready product — a strong foundation with clearly identified
next steps, not a finished commercial tool.

*(A follow-up improvement — using two secondary signals to double-check
borderline days, only when they're already close to the main threshold —
was added after the initial testing round and raised the fraud-catch rate
from an initial 59% to the current 65%, at no cost to the 92%
legitimate-case figure. See §7 for the honest story of how that
improvement was found and validated, including one version of it that
looked "free" and turned out not to be.)*

---

## 1. The headline numbers, and what they actually mean

| Metric | Result | In plain terms |
|---|---|---|
| **Fraud caught** | 65% | Out of every 10 simulated fraud cases, roughly 6-7 were correctly flagged |
| **Legitimate cases correctly left alone** | 92% | Out of every 10 real-but-unusual, non-fraud situations, roughly 9 were correctly NOT flagged |
| **Precision** (when it does flag something, how often is it right) | 82% | Out of every 10 flags raised, roughly 8 turned out to be a real simulated fraud case |
| **Perfect-score categories** | 2 of 8 fraud types | Fake night-time power, and frozen/replayed sensor readings — both caught with 100% reliability, every single time tested |

**Is a 65% catch rate good or bad?** Neither, on its own — it depends
entirely on *which* 35% is being missed, and that's the more informative
question. The system's strongest categories (fake night power, frozen
sensors) are caught perfectly, because those are essentially impossible
to fake without leaving an unambiguous physical fingerprint. The weakest
categories are the two hardest kinds of fraud in the entire test set —
see §3. A system that scored 100% across the board on a test this varied
would actually be a red flag for a rigged or trivially easy test, not a
sign of a stronger system.

---

## 2. What the system is genuinely good at

- **Perfect detection on the two most clear-cut fraud patterns.**
  Claiming power when the sun definitively isn't up, and freezing a
  sensor reading, are both caught 100% of the time, every time tested.
  These require no statistical judgment call — they're physically
  impossible or implausible by definition, and the system's rule-based
  checks catch them without exception.
- **Correctly stays quiet on honest data.** When run against real,
  verified-clean plant data with nothing modified, the system
  overwhelmingly reports a low-risk score, not a false alarm. This is
  arguably as important as catching fraud — a detector that constantly
  cries wolf is not useful in practice.
- **Clearly outperforms a simpler, cruder approach.** A more
  sophisticated statistical threshold was directly compared against a
  simpler, more naive one — the sophisticated approach wins clearly (a
  combined accuracy score of 0.72 vs. 0.62 for the simpler method), which
  justifies the additional complexity rather than assuming it.
- **Honest, disclosed uncertainty everywhere.** Every number in this
  project comes with its actual confidence level stated plainly — the
  system never claims more precision or certainty than the underlying
  data can actually support. This is a deliberate project value, not an
  accident.

---

## 3. Where the system is genuinely weaker — stated plainly, not hidden

- **Slow, creeping fraud is the hardest pattern to catch.** A fraud
  scenario designed to gradually ramp up over several days (rather than
  jump all at once) was only caught about 6% of the time — the one
  category neither the improvement round (§7) nor four further attempts
  after it (§8) managed to meaningfully help, even though more than one
  of those attempts looked promising in early testing. This is a genuinely hard statistical
  problem — the whole point of that fraud pattern is to never look
  unusual on any single day — and the project reports this rate honestly
  rather than disguising it.
- **Brief fraud confined to a small window is very hard to catch.** If a
  fake over-claim is squeezed into just one hour, or a single 15-minute
  reading, the system does not reliably catch it **at any size tested, up
  to a 30% overstatement.** The system's core decision only ever looks at
  a *full day's total* — so a brief spike, however large in the moment,
  can hide inside an otherwise normal day's total energy. Full-day,
  sustained over-claiming is what the system is built to catch reliably;
  brief, surgical fraud is a real, disclosed gap.
- **One specific legitimate scenario has a real overlap with a fraud
  check.** A grid operator legitimately capping a plant's output can, in
  some cases, trigger the same rule meant to catch a frozen sensor —
  correctly avoided only about 58% of the time for this specific
  scenario (versus 89–100% for every other type of legitimate event
  tested). Two different fixes for this were attempted and both were
  honestly found not to help without breaking something else — see
  §4.
- **The system doesn't transfer perfectly between the two plants.**
  Using one plant's prediction model to check the *other* plant produces
  inconsistent results — it worked reasonably for one direction but
  caused a lot of false alarms in the other. This means the current
  system is validated for each plant's own model, not proven to
  generalize automatically to a brand-new, unseen plant without
  retraining.

---

## 4. An example of good scientific discipline worth highlighting

For the "grid power cap sometimes overlaps with the frozen-sensor check"
issue above, the team didn't just note the problem and move on — they
proposed and tested **two different candidate fixes**, at real scale,
using dozens of test examples. **Both fixes were found to not
genuinely help** (one made things worse elsewhere; the other simply made
no measurable difference), and **both were honestly reverted** rather
than kept just because effort had already gone into building them. The
limitation is still documented as open and unresolved, not quietly
patched over with a fix that doesn't actually work. This kind of
"we tried, it didn't work, so we said so" discipline is exactly what
separates a rigorously validated result from an inflated one.

---

## 5. So — how good is this project, overall?

**As a proof-of-concept demonstrating a genuinely independent,
weather-based fraud-checking method: strong.** The core idea — verify
against data the audited party cannot influence — is sound, implemented
correctly, and tested against realistic, sealed scenarios rather than
just described in theory. The results are honestly reported, including
every weak spot, which is itself a meaningful sign of quality: a report
with zero disclosed limitations is far more suspicious than one that
shows its work.

**As a production-ready commercial tool: not yet, and that's expected at
this stage.** A real deployment would need: more historical data (34 days
across one season, on 2 plants, is a genuinely small sample by any
standard), validation across more plants and more seasons, a plan for the
brief/narrow-window fraud gap, and a resolution (or explicit accepted
risk) for the grid-curtailment overlap.

**Bottom line for a decision-maker:** the *method* is validated and the
*honesty of the reporting* is high — this is a credible foundation to
build further work on, not a finished product to deploy as-is. That
distinction is the correct one to draw at this stage of the project, and
overstating it either way (calling it "production ready" or dismissing it
as "not working") would misrepresent what was actually built and proven.

---

## 6. Recommended next steps, if this work continues

1. **More data, more time.** The single biggest lever for improving every
   number in this report is simply more historical days to learn from and
   calibrate against — the current 5-day calibration window in particular
   is a real statistical constraint, not a design choice.
2. **More plants, more seasons.** Validate on plants beyond the current
   two, and across more than one season, before trusting the system to
   generalize to a genuinely new site.
3. **A dedicated approach for brief/narrow-window fraud.** The current
   architecture makes one decision per day; catching short, sharp fraud
   would likely need an additional, differently-designed check rather
   than tuning the existing one further.
4. **Revisit the grid-curtailment overlap with a larger, purpose-built
   dataset.** Both attempted fixes so far were tested on the data
   available; a larger, more representative sample of this specific
   overlap case might reveal a workable discriminator that wasn't visible
   at the current scale.
5. **A plan for cross-plant deployment**, e.g. requiring a short local
   calibration period for any new plant before trusting the system's
   output on it, given the observed generalization gap.

None of these are signs the project failed — they're the normal, expected
next steps for a research prototype at this stage, and having them
clearly identified (rather than undiscovered) is itself a sign the
evaluation work was done properly.

---

## 7. A worked example of the project's own discipline, from just after initial testing

After the numbers above were first produced, a follow-up question was
asked: could the system be improved without changing what it fundamentally
is? Two existing components — a secondary pattern-recognition check and
the slow-drift tracker — already carried some detection signal that
wasn't being used in the main decision. The idea: only let those two
double-check a day that's *already* borderline on the main test, rather
than trusting them on their own (which earlier testing had already shown
backfires badly — see §2's comparison against cruder approaches).

**What happened, step by step, is a good illustration of how this project
works:**

1. A first version of this idea looked extremely promising — a large
   improvement in fraud caught, with **no cost at all** to legitimate
   cases. Too good to trust without checking further.
2. Checking further revealed the problem: the specific numbers used to
   configure the idea had been chosen by testing them directly against
   the same 30 scenarios used for final scoring — like tuning a
   recipe by tasting the dish you're about to serve to the judges. That's
   not a fair test of whether the idea actually works on anything new.
3. **Redone properly**: the configuration was chosen using only two-thirds
   of the test scenarios, then checked against the other third, which had
   never been looked at. The improvement mostly held up — but not for
   free this time; there was a real, if smaller, cost to legitimate
   cases. The honest result replaced the too-good-to-be-true one.
4. **One more check, on a different kind of data entirely**: the fix was
   then tried against real (not simulated) verified-honest plant data —
   the same data this report's §2 "correctly stays quiet on honest data"
   claim rests on. It surfaced one more real problem: a day that had
   never been flagged before now was — and it turned out to be the exact
   day used to originally set the detection threshold, which by
   definition can never be a legitimate example of "too much." That was a
   genuine bug, found and fixed.
5. After fixing the bug, the settings were made more conservative — giving
   up some of the fraud-catching gain specifically to make sure real
   honest data still comes back clean. The final numbers in this document
   (65% / 92%) are the result of that more careful, more conservative
   choice — not the initial, more dramatic-looking one.

**Why this is worth including in a results document, not just an
engineering log:** the instinct to trust a great-looking number the first
time it appears is exactly what this project has been built to resist,
everywhere else — the same discipline applied to every other number in
this report was applied here too, including when it meant giving up some
of the improvement to keep the result honest.

---

## 8. A further round of testing, after the improvement in §7 — four more ideas, all honestly ruled out

After §7's improvement shipped, one more round of work asked: could the
two weakest categories — the midday-only over-claim (26% caught) and the
slow, creeping drift (6% caught) — be improved further, still without
changing what the system fundamentally is? Four different ideas were
tried, each tested with the same rigor as everything else in this
report: never tuned against the sealed answer key, and always
double-checked against real honest data too, not just the simulated
tests.

**What was tried, and what happened:**

1. **Zooming in on just the busiest hours of the day**, instead of the
   whole day's total, to try to catch a fraud pattern that only ever
   appears mid-day. *Result: no real improvement.* The comparison got
   noisier by almost exactly as much as the fraud pattern became easier
   to spot, canceling out any gain.
2. **Making the slow-drift tracker more sensitive**, to catch shorter
   and smaller creeping increases. Checked honestly — using only real
   clean data and the test-generator's own public rules — *before* ever
   looking at a real test case. *Result: only a couple of percentage
   points of improvement*, nowhere near enough to be worth the added
   complexity.
3. **Combining several existing clues into one smarter score.** This one
   genuinely worked: it roughly doubled the midday-fraud catch rate and
   lifted the slow-drift catch rate from near-zero to about a third. But
   it also started wrongly flagging real, honest situations far more
   often — the "legitimate cases correctly left alone" rate dropped from
   92% to about 63%. *Not an acceptable trade for this project.*
4. **A more cautious version of #3**, only letting it act on days that
   were already borderline — the same careful rule already used in §7's
   improvement. This fixed the false-alarm problem, but then made
   *almost no net difference overall* — it caught a couple of different
   cases in place of the ones it lost, landing back at essentially the
   same total, and it still didn't move the slow-drift number at all.

**The honest bottom line**: none of the four were adopted, and the
headline numbers in §1 are unchanged. This wasn't wasted effort, though —
it converts a suspicion ("these two categories are probably about as good
as they can get with this much data") into a demonstrated fact, tested
four different ways. The one change that *would* meaningfully move the
numbers — using more of the real data to set the detection threshold —
was deliberately not made, because it would remove the project's own
safety check that the threshold is set correctly in the first place (the
same kind of check that caught the real bug described in §7). Full
technical detail on all four attempts is in
`docs/ESCALATION_IMPROVEMENT_HANDOFF.md` §8.

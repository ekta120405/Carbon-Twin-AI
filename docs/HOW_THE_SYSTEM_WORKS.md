# How CarbonTwinAI Works — every decision explained, in plain language

This document explains **everything** about how the system is built and
why — every major design choice, every algorithm, every piece of data,
written so someone with no machine-learning background can follow the
whole chain from raw data to a dollar figure. Nothing here is "trust me,
it's complicated" — every choice has a reason, and the reason is spelled
out.

If you just want the headline results, read `docs/PROJECT_RESULTS_SUMMARY.md`
instead. This document is the "how and why," not the "so what."

---

## 1. The problem, restated simply

A solar power plant gets paid (via carbon credits) based on how much
clean electricity it says it generated. That claimed number comes
**from the plant itself** — there's no independent referee checking it by
default. If a plant slightly exaggerates its numbers, it gets paid for
carbon credits it didn't actually earn.

The obvious fix — "just check the plant's own sensors more carefully" —
doesn't really work, because the plant controls those sensors too. If
someone wanted to cheat, they could cheat at the sensor level and no
amount of double-checking the same sensor would catch it.

**The fix this project uses:** compare the plant's claim against
something the plant has **zero ability to influence** — independent
satellite weather data, plus the basic physics of how solar panels work.
If the weather says a cloudy day should have produced modest power, but
the plant claims a sunny-day's worth of output, that mismatch is the
signal, regardless of what the plant's own sensors say.

---

## 2. Where the data comes from, and why each source is trusted (or not)

| Source | What it is | Trusted? | Why |
|---|---|---|---|
| **Claimed generation** (Kaggle dataset) | The plant's own reported power output, every 15 minutes | **No — this is the thing being checked** | It's exactly the number that could be exaggerated |
| **On-site weather sensors** (Kaggle dataset) | The plant's own weather station readings | **No — evidence only, never used to predict** | Also controlled by the plant; only used later as a secondary cross-check ("does the plant's own sensor agree with satellite data") |
| **Independent weather data** (NASA POWER, satellite-based) | Sunlight, temperature, humidity, wind, etc. for the plant's location | **Yes — the backbone of the whole system** | A public, third-party, satellite-derived source the plant cannot alter |
| **Solar physics calculations** (a well-established open-source library called `pvlib`) | Sun position, expected clear-sky brightness, panel temperature effects | **Yes — pure physics, not data from anyone** | These are textbook equations, not a claim from any party |
| **Government CO₂ conversion rate** (India's official CEA database) | How many tonnes of CO₂ one megawatt-hour of solar power avoids | **Yes — official published reference** | Used only for the final dollar/CO₂ conversion, not for detecting fraud |

Two real plants' data are used, from a public Kaggle dataset — 34 days of
real generation each, in mid-2020. This is a genuinely small amount of
data (see §9 for what that limits), and that's stated openly rather than
hidden.

### 2.1 Why timestamps had to be handled carefully

The plant's data is recorded in local Indian time; the independent
weather data comes in a different time standard. If the two aren't
aligned to the same clock precisely, the system could end up comparing
"claimed power at 2pm" against "expected power at 8:30pm" — a
five-and-a-half-hour mismatch that would make the whole system
nonsensical (it would think the plant generates power at night and is
dark at noon). This was fixed early and double-checked with a simple
sanity test: does the plant's real daily peak output actually line up
with when the sun is highest in the sky? If not, something's still
misaligned.

---

## 3. The four "buckets" the data is split into, and why

This is one of the most important structural decisions in the whole
project, so it's worth explaining carefully. The 34 real days per plant
are split into four groups, and **each group is only ever allowed to be
used for one specific purpose**:

| Bucket | Days | What it's used for | What it must NEVER be used for |
|---|---|---|---|
| **Training** | 1–20 | Teaching the prediction model what "normal" weather-to-power behavior looks like | Deciding any threshold, or being peeked at during testing |
| **Calibration** | 21–25 | Setting the statistical "how unusual is unusual" threshold | Teaching the model itself |
| **Verified-clean test** | 26–30 | Confirming the finished system behaves correctly on real, honest, never-before-seen data | Training or calibrating anything |
| **Test pool for fake scenarios** | 31–34 (+ the 26–30 days, reused) | The 9-day pool that the 30 synthetic test scenarios are built from | Training or calibrating anything |

**Why bother separating these at all?** Imagine studying for an exam by
memorizing the actual exam questions in advance — you'd score perfectly,
but you wouldn't have learned anything, and the score would be
meaningless. The same trap applies here: if the model, the threshold, and
the test were all built from the same days, a good score wouldn't prove
the system actually works — it would just prove it memorized its own
answer key. Keeping these buckets strictly separate is what makes every
number in this project's results section actually mean something.

### 3.1 Why the calibration bucket is only 5 days, and what that costs

With only 34 real days total, there isn't much to go around. Calibration
gets just 5 days. Statistically, this means the system's guaranteed
false-alarm rate is **about 1 in 6 (≈16.7%)**, not a cleaner "1 in 20"
that a larger calibration set would allow. This is reported honestly
throughout the project, never rounded up to sound better than it is —
see §9.

---

## 4. The frozen test-scenario generator ("the injector")

To actually measure how well the detector works, the project needed
examples of **known** fraud — cases where the true answer is already
known, so a score can be computed. Since there's no real, labeled fraud
data available (nobody hands out a "here's a plant that definitely
cheated" dataset), the project built its own test generator.

### 4.1 How it works

A separate part of the team built a tool that takes the 9 real, honest
days in the test pool and creates **30 different randomized versions**
("replicates") of that same 9-day window per plant. In each version, a
handful of days get modified:

- **About 3 of the 9 days** get a genuinely fake fraud pattern injected
  (see the 8 fraud types below).
- **About 3 of the 9 days** get a real-but-legitimate oddity simulated
  (see the 7 "don't cry wolf" types below) — things that look unusual but
  are NOT fraud, and the system must learn not to flag them.
- **The rest stay untouched**, real, honest data.

### 4.2 The 8 fraud patterns simulated (fraud that MUST be caught)

| Code | What it simulates | In plain terms |
|---|---|---|
| A1 | Constant over-claim | The plant multiplies its true output by a fixed factor all day (e.g. reports 20% more than it made) |
| A2 | Peak-hour shaving | Inflates only the busiest midday hours, when it's easiest to hide in normal variation |
| A3 | Night injection | Reports non-zero power when the sun isn't even up — the crudest, most obvious fake |
| A4 | Day-swap | Copies a sunny day's output shape onto a cloudy day's record |
| A5 | Weather spoofing | Inflates both the claimed power AND the plant's own on-site weather sensor together, to make the inflated claim look "supported" by (fake) local weather |
| A6 | Slow drift | Gradually ramps up the over-claim across several days, so no single day looks obviously wrong |
| A7 | Capacity violation | Claims more power than the plant is physically capable of producing |
| A8 | Flatline / frozen replay | Freezes the reported power at one fixed value for a stretch of time, as if the meter got stuck or a recording was replayed |

### 4.3 The 7 legitimate oddities simulated (must NOT be flagged as fraud)

| Code | What it simulates | In plain terms |
|---|---|---|
| N1 | Inverter outage | A real equipment failure — power is genuinely lower than expected, which is an operations problem, not fraud |
| N2 | Gradual soiling loss | Dust/dirt build-up slowly reducing output — a maintenance issue |
| N3 | Grid curtailment | The grid operator orders the plant to cap its output — legitimately lower power, ordered from outside |
| N4 | Sensor dropout | Missing readings — a data-quality problem, not a claim to distrust |
| N5 | Heavy cloud day | A genuinely very cloudy day — low output for an honest reason |
| N6 | Extreme heat | Very hot weather reduces panel efficiency — physically expected, not suspicious |
| N7 | Cloud enhancement | A known quirk where scattered clouds can briefly make a site's actual sunlight reading diverge from the satellite estimate — real, not fraud |

### 4.4 Why this tool was "frozen" and locked

The exact same team that later tuned the detection system did **not**
build this test generator freely while also tuning the detector — the
generator was built, then formally locked ("frozen"), before any
detection thresholds were finalized. This mirrors how a real exam is
written by someone who doesn't also get to see the answer key while
grading their own test — it prevents the detector from being
(consciously or unconsciously) tuned to just pass the specific test cases
it was checked against, which would make the reported accuracy numbers
meaningless.

---

## 5. The "digital twin" — predicting what generation *should* have been

This is the heart of the system: a way to estimate, from weather data
alone, what a solar plant's power output *should* be at any given moment
— without ever looking at what the plant claims to have produced.

### 5.1 Two layers, not one

**Layer one — physics, no learning at all.** Using a well-established,
open-source solar physics library (`pvlib`), the system computes a
straightforward, formula-based estimate: given the sun's position, the
clear-sky brightness, the panel temperature, and some basic
loss/efficiency assumptions, what would a solar panel of this size
produce? This is pure textbook physics — no training data, no "learning,"
nothing that could possibly overfit or memorize anything. It exists for
three reasons: (1) it's the foundation the machine-learning layer builds
on top of, (2) it works as a sanity check even in the worst-case scenario
where the learned model turns out unreliable, and (3) it needs zero
training data, so it's trustworthy even for a brand-new plant with no
history.

**Layer two — a learned correction on top.** The physics formula is a
good starting point but isn't perfect — real panels have imperfections,
soiling, spectral effects, and small modeling errors the pure formula
can't capture. So a machine-learning model is trained to learn just the
**correction** — how far off the physics-only guess tends to be, given
the weather conditions — rather than trying to predict total power output
from scratch. This is why it's called "physics-informed": the learning
problem is much smaller and easier (predicting a small correction, not
the whole answer), which matters a lot given there are only about 900
usable training rows.

### 5.2 What "PI-GBQR" means, in plain terms

The learned correction model's technical name is **PI-GBQR** —
**P**hysics-**I**nformed **G**radient-**B**oosted **Q**uantile
**R**egression. Breaking that down:
- **Gradient-boosted** — a well-tested, industry-standard machine
  learning technique (the same family used by many real-world forecasting
  systems) that builds its prediction as a sum of many small, simple
  decision rules, each one correcting the mistakes of the ones before it.
- **Quantile regression** — instead of producing one single "best guess"
  number, the model produces **three** numbers: a low estimate, a central
  (typical) estimate, and a high estimate. Together, these describe a
  *range* of plausible expected output, not a false sense of exact
  certainty. This matters because weather-based prediction is genuinely
  uncertain — clouds, temperature, and panel behavior all introduce some
  unavoidable noise, and pretending otherwise would be dishonest.

### 5.3 A rule the model is not allowed to break

The model is built with hard-coded physical common sense baked in: it is
mathematically forbidden from predicting that power goes *down* when
sunlight goes *up*, or that power goes *up* when the panels get hotter
(heat reduces efficiency, it doesn't increase it). This isn't a soft
preference the model usually follows — it's a strict rule enforced during
training. Why this matters: without it, unusual data patterns could in
theory push the model into a nonsensical, physically-backwards
prediction, and a model that can behave nonsensically is a model that's
harder to trust and much easier to game.

**One honest technical wrinkle:** the "central estimate" is not a true
statistical median — it's closer to an average, because of a technical
limitation in how the physical-common-sense rule above interacts with the
underlying math library. This is a small, well-understood, and openly
documented approximation, and the difference in practice is minor — but
it's called "central estimate" throughout this project rather than
"median," because that's the accurate description.

### 5.4 What happens on very cloudy days

There isn't much training data for extremely overcast conditions (only
about 9–12 example days total across the whole dataset), which isn't
enough to responsibly trust a learned correction. So below a certain
cloudiness threshold, the system **automatically stops using the learned
correction and falls back to the pure physics estimate** — the same
formula-based number described in §5.1. This is a deliberate design
choice, not a workaround: the physics formula doesn't need training data
to be trustworthy, so it's the safer choice exactly when there isn't
enough learned experience to rely on.

### 5.5 The rule that makes the whole system honest

**The prediction model is never, under any circumstance, allowed to see
the plant's own claimed generation numbers or its own on-site weather
sensor readings.** This is the single most important rule in the entire
project. If the model could see the numbers it's supposed to be checking,
it could — even unintentionally — learn to just agree with them, which
would make the whole exercise pointless (a security camera that's allowed
to watch its own recording before deciding if anything suspicious
happened isn't a security camera). This rule is enforced by an automated
check that runs every time the model is trained, not just a promise in a
document.

---

## 6. Comparing claimed vs. expected, and deciding what counts as "unusual"

### 6.1 The basic comparison

For every day, the system computes: **claimed energy minus expected
energy**, expressed as a share of the plant's total capacity (so a small
plant and a large plant can be compared on the same fair scale). This
comparison is only made using daylight hours — comparing at night, when
both numbers should be near zero anyway, adds noise without adding
information.

### 6.2 Why the comparison is made per *day*, not per 15-minute reading

Weather (and by extension, expected power) is naturally jumpy from one
15-minute reading to the next — a passing cloud can cause a real,
harmless dip that has nothing to do with fraud. Totaling up a whole day's
worth of energy smooths that noise out. It also matches what actually
matters for carbon credits: credits are earned on *energy* generated
(a total), not on the instantaneous *rate* of generation at any single
moment.

### 6.3 How "unusual enough to flag" is decided — without guessing

A cruder approach would be to just pick a number ("anything more than 5%
over, flag it") out of thin air. Instead, this project uses a more
principled and honest method: it looks at the 5 verified-honest
calibration days, finds the single **largest** gap seen among those
genuinely clean days, and uses that as the cutoff. Anything past that
point is treated as more unusual than anything ever seen on honest data —
worth a flag.

**The honest limitation:** with only 5 honest days to learn "normal"
from, this method can only mathematically guarantee a false-alarm rate of
about 1-in-6 (not the cleaner, more standard 1-in-20 a bigger calibration
set would allow). This is reported plainly everywhere this number is
used — never rounded up to sound more precise than it really is.

### 6.4 Two simple, rule-based checks (no statistics needed)

Some kinds of fraud are so blatant that no statistical judgment call is
needed — a plain rule is enough:

- **Night injection check.** If a plant reports meaningful power at a
  time when the sun's known position says it should be completely dark,
  that's flagged immediately, regardless of any statistical threshold.
  This required care to get right: a naive "is it night" check
  accidentally flagged real dawn/dusk generation too often, so the rule
  is anchored to the actual clear-sky brightness calculation instead,
  with a small allowed tolerance for sensor noise.
- **Frozen/stuck sensor check.** If the exact same reading repeats many
  times in a row (8+ consecutive 15-minute readings) during daylight,
  that's flagged as a possible frozen meter or a replayed recording — real
  sensor noise essentially never produces that many bit-identical
  readings in a row.

**A known, honestly-documented limitation of the stuck-sensor check:** it
can also occasionally fire on a *legitimate* situation — a grid operator
capping the plant's output at a fixed value for a while (this is a real,
non-fraud event). A fix for this was attempted twice during the project
and, both times, honestly found not to actually help without breaking
something else — see §9.4. Rather than force a fix that didn't work, this
is documented as a known, accepted limitation.

### 6.5 The slow-drift detector — catching fraud that hides by never looking bad on any single day

Some of the cleverest possible fraud wouldn't be a single big jump — it
would be a **slow, creeping increase spread across many days**, small
enough on any individual day to slip under every threshold above, but
adding up to something real over time.

To catch this, the system keeps a **running tally**: each day, if the
gap is a little bit above the plant's own normal average, a small amount
gets added to a running total; if the gap is normal or below average, the
running total resets toward zero. If that running total ever climbs past
a threshold, the day gets flagged — not because that one day looked bad,
but because the *pattern building up over recent days* looks like a
sustained drift. (Think of it like noticing a slow water leak by tracking
the total water bill trend over weeks, not by staring at any single day's
reading.)

**Honestly reported result:** in testing, this specific tool caught only
a small fraction of the slow-drift fraud cases it was built for — the
slower and subtler the drift, the harder even this method finds it. This
is reported as a real, disclosed limitation (see §9), not hidden.

### 6.6 A second, experimental detection method (not used as the main decision-maker)

A different kind of pattern-recognition tool was also built and tested:
one that learns what a "normal-looking combination" of weather, expected
power, and claimed power looks like all together, and flags anything that
resembles a combination it's never seen in honest data — even if no
single number involved looks obviously wrong on its own. (Think of it
like noticing something's subtly "off" about a whole scene, even before
you can point to the one specific object that's out of place.)

**Why it's not the main detector:** in testing, blindly trusting this
tool's opinion on its own turned out to be far too sensitive — it caught
more true fraud, but at the cost of also incorrectly flagging a large
share of genuinely honest situations (nearly 3 out of 4 legitimate cases
got wrongly flagged). That trade-off isn't good enough to trust as a
primary decision-maker. It's used a second, much more careful way
instead — see §6.8.

### 6.8 Using secondary evidence carefully: only double-checking a borderline day

Both the slow-drift tracker (§6.5) and the pattern-recognition tool
(§6.6) carry real signal that the main daily check (§6.3) doesn't use on
its own — but blindly trusting either one, on every day, backfires (as
just described). The fix that actually works: **only let them
double-check a day that's *already* close to crossing the main
threshold** — not any day, and not on their own authority. A day gets
upgraded from "normal" to "flagged" only if both (a) it's within a
defined margin of the main threshold, and (b) at least one of the two
secondary tools also finds something unusual about that same day.

This is a deliberately narrow, cautious rule — the two secondary tools
are only ever asked to break a tie on a day that was already a near-miss,
never to raise a fresh alarm on a day the main check saw nothing
suspicious about at all.

**How this was validated, honestly:** the settings controlling exactly
how close is "close enough" and how much secondary evidence is "enough"
were chosen carefully, in three rounds, because the first two rounds each
revealed a real problem before being trusted:
- The first attempt looked like a free improvement (more fraud caught,
  zero added false alarms) — but that was because the settings had been
  chosen by testing them directly against the same scenarios used to
  measure the final result, which isn't a fair test.
- Redone honestly (choosing the settings on some test scenarios, then
  checking on completely different ones the settings had never seen), the
  improvement mostly held up, but with a real, smaller cost this time —
  not free after all.
- A further check against real, verified-honest plant data (not the
  synthetic test scenarios at all) caught one more real problem: a
  genuinely honest day was being flagged, and it turned out to be the
  exact day that had originally been used to set the main threshold — a
  day that, by definition, cannot be "too high" relative to itself. That
  was a real bug, found and fixed.
- The final, shipped settings are the more conservative choice that
  survived all three checks: real, meaningful improvement in fraud caught
  (about 6 percentage points), at no added cost to legitimate cases.

See `docs/ESCALATION_IMPROVEMENT_HANDOFF.md` for the full technical account, and
`docs/PROJECT_RESULTS_SUMMARY.md` §7 for why this whole process is a
useful example of the project's own standards, not just an engineering
detail.

### 6.9 A later round: four more ideas tried, none adopted, and why that's still useful

After the double-checking rule above shipped, the team tried four more
ideas specifically aimed at the two remaining weak spots: the midday-only
over-claim, and the slow, creeping drift. Every idea was tested the same
careful way as everything else in this project — checked against real
honest data, and never tuned by peeking at the sealed answer key before
the final check.

1. **Looking only at each day's busiest hours**, instead of the whole
   day, to try to catch the midday-only trick more easily by not diluting
   it into a full day's total. Result: almost no difference. Shrinking
   the time window makes the "what's normal" comparison noisier by about
   as much as it makes the fraud pattern stand out — the two effects
   cancel out.
2. **Making the slow-drift tracker react faster**, so it could catch
   shorter, smaller creeping increases. This was checked honestly
   *before* looking at any real test case — using only real clean data
   and the test-generator's own public settings, never a specific
   simulated example. Result: only a couple of percentage points of
   improvement, nowhere near enough to justify using it. Making the
   tracker more sensitive also makes it more prone to false alarms on
   ordinary days, so its own alarm threshold has to rise to compensate —
   and that trade-off cancels out almost all the gain.
3. **Combining several existing clues into one smarter score** — the
   day's gap, the drift tracker's running total, and the
   pattern-recognition tool's opinion — instead of checking them one at a
   time. Result: this genuinely worked at catching more fraud (it roughly
   doubled the midday-fraud catch rate and lifted the slow-drift catch
   rate from near-zero to about a third), but it also started wrongly
   flagging real, honest situations far too often — a cost this project's
   own standards don't accept. Real signal, unacceptable price.
4. **A more careful, restrained version of #3** — only letting it speak
   up on a day that's already borderline, the exact same cautious rule
   the double-checking trick above already uses. This fixed the
   false-alarm problem, but then it barely helped at all: it caught a
   couple of different cases instead of new ones, landing back at
   essentially the same overall result — and it still didn't touch the
   slow-drift problem, because a genuinely slow drift almost never looks
   borderline on any single day in the first place, which is exactly
   what makes it hard.

**The honest conclusion**: both remaining weak spots trace back to how
little genuinely honest history this project has to learn from — only 5
real days set the main detection threshold (§3.1). The one change that
*would* meaningfully help — using more of the real data to set that
threshold — was considered and deliberately not made, because it would
remove the project's last independent check that the threshold is set
correctly in the first place (the same kind of check that caught a real
bug once, described in the box above). Full technical detail on all four
attempts is in `docs/ESCALATION_IMPROVEMENT_HANDOFF.md` §8.

### 6.7 Why so many different checks, instead of just one?

Fraud doesn't have one single shape. A crude, obvious fake (reporting
power at night) looks nothing like a subtle, creeping fraud (a slow drift
over many days), which looks nothing like a frozen-meter replay. Rather
than force one method to catch every possible pattern (and inevitably do
a mediocre job at most of them), this project uses several
purpose-built tools, each aimed at a specific, well-understood pattern —
and is honest in the results about which tool is strong where, and where
each one still has real, documented gaps.

---

## 7. Turning a flag into a carbon and dollar number

Once a day is flagged, the next question is: how much energy does this
actually represent, in real terms?

### 7.1 Why the "how much energy" calculation is different per flag type

It would be tempting to just use one generic "claimed minus expected"
number for every flag — but that would be wrong in some cases and was an
actual bug caught and fixed during development. A "reported power at
night" flag's real fraud energy is concentrated specifically in the dark
hours it happened in — using the whole day's generic gap instead would
have hugely overstated the number (an early version of this bug inflated
one real case from a correct ~500 kWh to an incorrect ~173,000 kWh — over
300 times too high). So each flag type gets its own, matching
calculation: a plain "over-claim" flag uses the whole day's excess
energy; a "night injection" flag uses only the actual energy claimed
during the dark hours; a "frozen sensor" flag uses only the energy
difference during the specific frozen stretch, not the whole day.

### 7.2 From kWh to tonnes of CO₂ to dollars

```
excess energy (kWh) ÷ 1000 × grid emission factor  =  tonnes of CO₂ avoided (that shouldn't have been claimed)
tonnes of CO₂  ×  reference carbon price  =  dollar value at risk
```

The **emission factor** (how many tonnes of CO₂ one megawatt-hour of grid
electricity represents) comes from an official Indian government
database (CEA CO₂ Baseline Database), the specific edition meant for
renewable-energy crediting calculations. The **carbon price** is a
representative reference figure, not a live market quote (there isn't a
single agreed carbon price globally — it's a genuinely fragmented
market).

### 7.3 Two honest caveats, always shown with these numbers

1. **Vintage mismatch.** The emission-factor rate used is a current
   reference rate, but the actual generation data being analyzed is from
   2020. This is a stated simplification (this project is demonstrating
   the *method*, not issuing a real, registered credit) — not an error.
2. **Price uncertainty.** The carbon price used is one reasonable,
   representative estimate at the lower end of a wide, genuinely
   disputed range — never presented as a precise market rate.

Every number derived this way is described as **"at risk,"** never
**"confirmed fraud"** or **"credits over-issued."** A flag is a
calibrated statistical red flag, not a legal or financial finding — that
distinction is treated as important enough to repeat everywhere these
numbers appear.

---

## 8. Explaining a flag to a human reviewer

A flag alone isn't very useful to an auditor without context. For every
flagged day, two separate explanations are generated:

### 8.1 "What did we expect, and why?"

This breaks down *which specific weather factors* pushed the computer's
expected-power estimate up or down that day — e.g., "low sunlight and
high humidity pushed the expectation down." This lets a human sanity-check
the model's reasoning instead of just trusting a black box. One honest
technical note: this explanation shows only the *learned correction*
part of the prediction (§5.1's "layer two"), not the physics-only part,
because the physics formula is deterministic arithmetic with nothing to
"explain" — there's no pattern-learning involved in it to break down.

### 8.2 "Why is this considered suspicious?"

This depends on which kind of flag fired:
- For the main statistical flag (an unusually large gap), the system
  shows which specific measurements about that day (like how much the
  on-site sensor disagreed with satellite data, or how big a jump this
  day was compared to the plant's own recent trend) looked most unusual
  compared to a pool of genuinely honest days.
- For the two rule-based flags (night injection, frozen sensor), there's
  nothing statistical to explain — it's a simple rule that either fired
  or didn't — so the system instead states, in plain language, exactly
  which rule fired and the exact numbers involved (e.g. "187.3 kW claimed
  at a time when the sky brightness reading was zero").

Any known limitation tied to a specific flag type (like the frozen-sensor
check's occasional overlap with legitimate grid curtailment, §6.4) is
shown directly alongside that flag's explanation — not buried somewhere
a reviewer might miss it.

---

## 9. Testing it — and being honest about the results

### 9.1 The scoring method

Every one of the 30 sealed test scenarios per plant was run through the
full system, and the result was compared against the known, sealed
answer key. Two headline numbers came out of this:

- **How much real, simulated fraud did the system actually catch?**
  (technical term: recall)
- **How often did the system correctly leave real, legitimate,
  non-fraud oddities alone?** (technical term: specificity)

See `docs/PROJECT_RESULTS_SUMMARY.md` for the actual numbers and what
they mean. This document is about method, not the score itself.

### 9.2 Extra stress tests run, and why

Beyond the headline scoring, several additional checks were run to make
sure the results weren't a fluke or an artifact of one specific setup:

- **Wrong-plant test.** What happens if the prediction model trained on
  one plant is used to check the *other* plant instead? This tests
  whether the system generalizes, or was just quietly memorizing one
  specific plant's quirks. (Result: it depends heavily on which plant's
  model gets reused — reported honestly, including the case where it
  degrades badly, not just the case where it holds up.)
- **Alternative-method comparisons.** The chosen statistical method was
  compared head-to-head against a simpler, cruder threshold, and against
  a version that also flags unusually *low* output (which turns out to
  be a bad idea — it starts falsely flagging real equipment problems as
  fraud). These comparisons are what justify choosing the more careful
  approach actually used, with real numbers behind the choice rather than
  just an assertion.
- **"How small a fraud can we even catch" test.** A separate sweep
  specifically measured the smallest and shortest-lived fake over-claims
  the system can still detect — this is shown on the dashboard's Accuracy
  & Testing tab.

### 9.3 What this system is *not* claiming

- It does not claim to catch 100% of possible fraud — no realistic
  system could, and pretending otherwise would be dishonest. Some fraud
  patterns (very small, or very short-lived) are genuinely hard to
  distinguish from ordinary weather noise, and that limit is measured and
  reported, not hidden.
- It does not claim a flag is proof of fraud — only that a specific day's
  numbers look statistically unusual (or, for the two rule-based checks,
  physically implausible) enough to warrant a human review.
- It does not claim to work on data very different from what it was
  tested on (only 34 real days, 2 plants, one season) — extending this to
  new plants or seasons would need fresh validation, and that limitation
  is stated openly rather than glossed over.

### 9.4 An honest example of "we tried, it didn't work, so we didn't force it"

The frozen-sensor check's known overlap with legitimate grid curtailment
(§6.4) was investigated twice, using two different candidate fixes. Both
times, the fix was tested rigorously against real examples, and both
times it was found to either cause a worse problem than it solved, or to
simply not help at all. In both cases, the fix was reverted and the
original, simpler rule was kept, with the honest result written up rather
than hidden. This is deliberately included here as an example of the
project's actual working discipline: a fix only gets kept if it's proven
to genuinely help, not because effort was already spent trying it.

---

## 10. Where everything lives, for reference

| Concept | Where the logic actually lives |
|---|---|
| Data cleaning, splitting into buckets | `src/layer1_data/` |
| Physics estimate + learned correction (the "twin") | `src/layer2_twin/` |
| Comparison, thresholds, slow-drift tracker, pattern-recognition tool | `src/layer3_anomaly/` |
| Explanations, carbon/dollar conversion | `src/layer4_xai/` |
| The frozen test-scenario generator | `src/injection/` |
| Full-scale testing and scoring | `src/evaluation/` |
| The dashboard shown to an audience | `dashboard/app.py` |

Every one of these is documented in far more technical depth in its own
handoff document under `docs/` for anyone who wants the full engineering
detail behind any specific piece.

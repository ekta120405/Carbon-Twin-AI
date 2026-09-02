# Reading CarbonTwinAI — a complete, plain-language guide to the dashboard

This document explains **every single element** on the dashboard
(`dashboard/app.py`) — every number, every chart, every color, every
label — what it means and exactly how to interpret it. If you're
presenting this to a panel and someone points at any pixel on the screen
and asks "what's that," the answer is in here.

For a lighter, faster-to-skim version with just the demo script, see the
"Running order for the demo" and "Questions a panel is likely to ask"
sections near the end — you don't need to re-read the whole document
before every presentation, just before the first one.

For *how the underlying system works*, see `docs/HOW_THE_SYSTEM_WORKS.md`.
For *the actual results and what they mean*, see
`docs/PROJECT_RESULTS_SUMMARY.md`. This document is specifically about
**the dashboard as a piece of software** — what's on screen and what it's
showing you.

The dashboard is a **single page with five tabs** — not five separate
pages. Whatever plant and time period you pick in the sidebar stays
selected no matter which tab you click, because switching tabs never
reloads the page.

---

## The big idea, in one breath

A solar plant reports how much electricity it generated. That number
decides how many carbon credits it can claim. **CarbonTwinAI checks that
number against an independent estimate** — built purely from satellite
weather data the plant cannot influence — and flags any day where the
plant claimed meaningfully more than the weather says was possible. The
dashboard is where a human looks at what got flagged, why, and what it's
worth in tonnes of CO₂ and dollars.

---

## The sidebar — two controls that affect all five tabs

### Plant

A dropdown: **Plant 1** or **Plant 2**. These are two different, real
solar installations, kept completely separate throughout the whole
system — different physical size (measured in kW of capacity), different
location, different weather history. Switching this dropdown re-runs
every number and chart on every tab for the newly selected plant. There
is no "combined" view showing both plants at once, except on the
Accuracy & Testing tab, which always shows both plants pooled together
regardless of this dropdown (see that tab's section below for why).

### Period

A dropdown with several options, grouped into three kinds:

- **"Recent activity (real data)."** All of this plant's real, actual
  reported days on record — nothing simulated, nothing modified. Use this
  to look at the plant's genuine history.
- **"Verified clean period (real data)."** A shorter window of real days
  that the team specifically verified as honest before using it anywhere
  in this project. This is the period to show if you want to demonstrate
  the system correctly staying quiet on data with nothing wrong in it.
- **"Test case 1" through "Test case 30."** Not real data — each one is a
  synthetic scenario built by taking 9 real days and deliberately
  modifying a handful of them (see "What's a test case?" below). Every
  test case is different: different days get modified, different fraud
  or legitimate-oddity types get used, and how obvious or subtle the
  modification is also varies randomly. **This means some test cases will
  show zero or very few flags, and that's expected, not a bug** — not
  every test case is designed to be dramatic. If you want a test case
  guaranteed to show multiple different kinds of flags for a demo, use
  **Test case 11** for Plant 1 (internally this is "replicate 010") or
  **Test case 1** for Plant 2 (internally "replicate 000") — both were
  specifically verified to contain a mix of flag types.

**What's a "test case"?** A separate part of the team built a tool that
takes real, honest days and deliberately fakes 30 different test versions
per plant — some with a plausible-looking fraud pattern slipped in, some
with a real but legitimate oddity (a grid power cap, a cloudy day, a
sensor outage) that must *not* be mistaken for fraud. That tool was built
once and then formally locked, so nobody tuning the detector could see
the answer key while tuning it — the same discipline a real auditor's
test set would need. The answer key (what was actually faked, and where)
is what the Accuracy & Testing tab ultimately scores against — but note
that individual test-case answer keys are **not** shown anywhere on the
main four tabs; those four tabs show you what the *system decided*, not
what the *right answer was*, so you're seeing the same view an actual
auditor would have.

---

## Tab 1 — Overview

**The question this tab answers:** "Overall, for this plant and this
period, how worried should I be?"

### The verdict banner

A colored box at the top with an icon and a short headline. This is a
direct, plain-language translation of a 0–100 numeric **risk score**
(NOT called a "trust score" — see the note below on why the labeling
matters) into one of four bands:

| Score range | Color | Icon | Headline |
|---|---|---|---|
| 0–25 | Green | ✅ | "Looks trustworthy" |
| 25–50 | Yellow | 🟡 | "A few irregularities found" |
| 50–75 | Orange | 🟠 | "Multiple irregularities found" |
| 75–100 | Red | 🔴 | "Serious irregularities found" |

Directly underneath the banner, in small text: **"Risk score: NN / 100"**
— the exact number the color band was derived from. A score of 24 and a
score of 26 look almost identical in substance but land in different
color bands — if you're explaining a borderline case, it's worth
mentioning the exact number, not just the color.

**How to interpret this:** think of it like the opposite of a credit
score — **low is good**. A low number means this specific plant's claims,
over this specific period, closely matched independent reality; a high
number means they didn't. It is **not** a fixed property of the plant
overall — switching the "Period" dropdown to a different test case or a
different real window will very likely produce a different score,
because it's scoring *that period's data specifically*.

**Why it's called a "risk score," not a "trust score":** the underlying
number is 0 for spotless and 100 for maximally suspicious — it goes UP as
the plant looks WORSE. Calling that a "trust score" would read backwards
(a low "trust score" sounds like "not trustworthy," when a low number
here actually means the opposite). The label was corrected to match what
the number actually does, rather than inverting the number to match a
friendlier-sounding label.

### The three stat tiles

- **"Days reviewed"** — the total number of days included in whatever
  period is currently selected. For a test case, this is typically 9; for
  "Recent activity," it's the plant's full real history (34 days).
- **"Days flagged"** — how many of those days triggered at least one kind
  of flag (any category — see the bar chart below for the breakdown by
  type). A day counts here even if it was only borderline.
- **"Carbon credits at risk"** — the total estimated dollar value tied to
  every flagged day in this period, combined. This is the exact same
  total shown at the top of the Carbon Credits tab — it's placed here too
  because it's the number most people will want to see first (see the
  note below on why it's repeated).

**Why the carbon number is on this tab AND its own tab:** early feedback
was that the carbon-credit angle — which is literally the project's whole
subject — was too easy to miss if it only appeared on one tab out of
five. It's shown here as a headline number for visibility, and broken
down in full detail (by category, by day, with caveats) on the dedicated
Carbon Credits tab.

### The flag-type bar chart

**"What was flagged, and how much"** — one bar per category of flag that
actually occurred in this period (categories with zero occurrences don't
get a bar at all). The height of each bar is the number of days in that
category. The bar's color matches the same color used everywhere else on
the dashboard for that category:

| Bar color | Category | What it means |
|---|---|---|
| Dark red | Overclaimed energy | The plant's claimed total for that day was statistically unusual — more energy than the weather-based estimate says was plausible, by more than the calibrated threshold |
| Orange | Reported power at night | The plant claimed meaningful power at a time when the sun's position data says it should have been completely dark |
| Purple | Frozen sensor reading | The exact same reading repeated many times in a row during daylight — a possible stuck meter or replayed recording |
| Blue | Under-claimed (equipment issue, not fraud) | The plant reported *less* power than expected — this is flagged separately as a possible operational problem (an outage, soiling, a real fault), and is explicitly **not** treated as a fraud signal anywhere in this system |
| Grey | Normal | No flag — included in the chart's underlying data but typically not drawn as its own bar since "no flag" isn't the point of this chart |

If this tab shows "No flags in this period" instead of a chart, that
simply means every day in the currently selected period came back clean
— try a test case from the sidebar to see flagged examples.

### The "How is the risk score calculated?" note

A collapsed-by-default expander (click to open). It lists, in one short
paragraph, the five things that feed into the 0–100 score: how many days
were flagged, how much energy is unaccounted for, whether there's a slow
creeping pattern building up, whether this plant's numbers match its own
normal historical behavior, and whether its on-site sensors agree with
independent satellite data. It's intentionally brief — the full technical
breakdown of how these five things are weighted and combined lives in
`docs/HOW_THE_SYSTEM_WORKS.md`, not on the dashboard itself.

---

## Tab 2 — Evidence

**The question this tab answers:** "Show me the actual evidence, day by
day, not just a summary score."

### The main chart — "Power over time"

A line chart. Time runs left to right along the bottom; power (in
kilowatts, kW) runs bottom to top on the left.

- **The red line** is what the plant actually claimed, every 15 minutes.
- **The blue line** is the independent, weather-based expected power —
  what an honest plant should have produced given the actual weather that
  day.
- **The pale blue shaded band around the blue line** is the expected
  *range*, not just one number — solar prediction is never perfectly
  certain, and this band shows how much natural variation is expected. A
  claimed value sitting *inside* this shaded band, even if it's above the
  solid blue line, is within a plausible range. A claimed value sitting
  clearly *above* the top edge of the shaded band is the kind of gap that
  starts contributing to a flag.
- **Colored vertical stripes** mark specific flagged days — the stripe's
  color matches the flag-type legend from Tab 1 (dark red = overclaim,
  orange = night injection, purple = frozen sensor). If a whole day's
  worth of the chart has a faint colored tint behind it, that's this
  stripe — it's deliberately subtle (15% opacity) so it doesn't hide the
  lines underneath.

**How to interpret this chart:** the story you're looking for is the red
line pulling away *above* the blue line (and its shaded band) on a
flagged day. On a normal, unflagged day, the red and blue lines should
track each other reasonably closely, rising and falling together as the
sun moves through the sky.

### The second chart — "Daily gap vs. the flag threshold"

A bar chart underneath the main one. Each bar is one day; the bar's
height is that day's total gap between claimed and expected energy
(normalized so it's comparable regardless of the plant's size — this is
the same number, called "r_day" internally, that the whole detection
system's main statistical check is built on). Each bar is colored by
that day's flag outcome, using the same color key as everywhere else.

A **black dashed horizontal line** marks the flag threshold. This is the
literal cutoff: any bar whose top crosses above this dashed line is, by
definition, a statistically unusual day (this is the mechanism behind
the "overclaimed energy" / dark-red flag specifically — a bar can cross
this line and still not show up dark-red if a different, higher-priority
rule-based check caught that day first; see `docs/HOW_THE_SYSTEM_WORKS.md`
§6 for the exact priority order).

**How to interpret this chart:** this is the single most "underneath the
hood" honest view on the whole dashboard — it's the literal number the
system's core decision is made from, with no interpretation layered on
top. A bar just barely above the line represents a borderline case; a bar
far above it represents an unambiguous one.

### The "How to read this chart" note

A short, collapsed expander repeating the red-line/blue-line/stripe
explanation above in two sentences, for a quick reminder mid-presentation
without needing this whole document open.

---

## Tab 3 — Carbon Credits

**The question this tab answers:** "What's actually at stake, in money
and climate terms?" — this is the project's actual subject, so this tab
is deliberately front-and-center, not buried.

### The three headline stat tiles

- **"Flagged days"** — same count as Tab 1's "Days flagged."
- **"CO₂e at risk"** — the total estimated tonnes of carbon-dioxide-
  equivalent tied to every flagged day's excess energy, combined, for the
  currently selected period. "CO₂e" ("carbon dioxide equivalent") is the
  standard way climate accounting expresses different greenhouse gases on
  one common scale — here it's effectively just tonnes of CO₂, since
  solar generation accounting doesn't involve other gases.
- **"Value at risk"** — the same total, converted to an estimated dollar
  figure using a reference carbon credit price. See §7 of
  `docs/HOW_THE_SYSTEM_WORKS.md` for exactly how this conversion works
  and its two stated caveats (repeated in the note below too).

**How to interpret these:** a $0.00 / 0.00 t result is a **good** sign —
it means nothing in this period was flagged, so there's no unaccounted-
for energy to price. These numbers only grow when real flags exist; they
are never negative and never represent a plant's *legitimate* generation
being "charged" — only the specific excess tied to flagged days.

### "By category" — the bar chart

One bar per flag category that had a non-zero dollar value in this
period, colored the same way as Tab 1's chart, with the exact dollar
figure labeled above each bar. This answers "which *kind* of irregularity
is contributing the most financial exposure" — useful for prioritizing
which pattern to investigate first if multiple categories appear.

### "By day" — the table

One row per flagged day, showing: which category it falls into, how much
energy (in kWh) that day's flag represents, the tonnes of CO₂e, and the
dollar value. This is the same underlying numbers as the bar chart above,
just broken out day-by-day instead of summed by category — useful if you
want to point at one specific day's exact figures.

### The "About these numbers" note

Two short bullet points, always worth reading aloud in a presentation:

1. These are estimates against a *reference* carbon price, not a live
   market quote — the voluntary carbon market doesn't have one single
   agreed price.
2. Every figure is phrased as **"at risk,"** never **"confirmed fraud"**
   — a flag is a calibrated statistical red flag, not a legal or
   financial finding.

---

## Tab 4 — Flag Details

**The question this tab answers:** "Take me inside one specific flagged
day and explain exactly why it was flagged."

### The dropdown

Lists every flagged day in the currently selected period, ordered from
**highest dollar value at risk to lowest** (not chronologically) — so the
first option in the list is always the single most financially
significant flag in the period. Each option is labeled
`<date> — <category> — <dollar value>`, e.g. "2020-06-13 — Overclaimed
energy — $137.28," so you can see at a glance what you're about to open
before clicking. If this tab shows "No flagged days in this period"
instead, switch the sidebar's Period to a test case to see a worked
example.

### The three stat tiles (for the selected day only)

- **"Energy overclaimed"** — this specific day's excess energy, in kWh,
  using the category-specific calculation method described in
  `docs/HOW_THE_SYSTEM_WORKS.md` §7.1 (a night-injection day's number
  reflects only the dark-hour energy, not the whole day, for example).
- **"CO2e at risk"** and **"Value at risk"** — this one day's individual
  contribution to the totals shown on the Carbon Credits tab.

### The left panel — "What we expected"

A horizontal bar chart. Each bar is one weather factor (sunlight
intensity, cloud cover, air temperature, panel temperature, humidity,
etc. — see the full list in the code's `READABLE_FACTORS` mapping if
you're curious about all possible labels). The bar's length and direction
show how much that factor pushed the computer's power expectation up or
down for this specific day:

- **A green bar pointing right** — this factor pushed the expected power
  *up* (e.g., "more sunlight than typical" would push the estimate
  upward).
- **A red bar pointing left** — this factor pushed the expected power
  *down* (e.g., "unusually high panel temperature," since heat reduces
  panel efficiency).

The x-axis is labeled "Effect on expected power (kW)" — the actual
kilowatt contribution of that specific factor to the day's final
expectation number.

**If this panel instead shows a caption saying weather conditions were
too overcast for a detailed breakdown:** this means the day fell below
the cloudiness threshold where the system automatically switches to the
simpler, physics-only estimate (see `docs/HOW_THE_SYSTEM_WORKS.md` §5.4)
— there's no learned-model reasoning to show a breakdown of on a day like
that, because none was used.

### The right panel — "Why it's flagged"

This panel takes one of two completely different forms, depending on
*which kind* of check flagged the day:

**Form 1 — a bar chart** (for the main statistical "overclaimed energy"
category). Each bar is one measurement (e.g. "on-site vs. satellite
sunlight mismatch," "sudden jump vs. recent days") and its length shows
how many standard deviations away from normal, honest-day behavior that
measurement was for this specific day — this is a statistics term meaning
"how unusual, expressed in the same consistent unit regardless of what's
being measured." Bar color intensity signals how unusual:
- **Dark red** — extremely unusual (2 or more standard deviations away)
- **Orange** — moderately unusual (1 to 2 standard deviations)
- **Grey** — within a fairly normal range, but still among the top
  factors examined for this day

The x-axis is labeled "How unusual, vs. normal clean days" — a bar
sticking far to the right (or left) means that specific measurement
looked very different from what genuinely honest historical days look
like.

**Form 2 — a plain-text warning box** (for the two rule-based flags:
reported power at night, or frozen sensor). There's nothing statistical
to chart here — these are simple, deterministic rules that either fired
or didn't — so instead you get one plain sentence stating exactly what
was observed and exactly which rule it tripped, e.g. *"Flagged: 187.3 kW
claimed at 2020-06-08 03:15 where the clear-sky brightness reading was
essentially zero — physically implausible generation during deep
night."*

### The caveat line

If this particular flag category has a known limitation attached, a
one-line warning (⚠) appears below the two panels. There are currently
two possible messages:

- For a frozen-sensor flag: *"This kind of flag can occasionally also
  fire on a legitimate grid-ordered power cap, not just a frozen sensor —
  treat as suspected, not confirmed."*
- For a night-injection flag: *"The 'what we expected' panel shows the
  day's most unusual daytime moment, not the actual night-time reading
  that triggered this flag."* (This is a real, structural quirk: the
  left panel's weather-reasoning breakdown is always computed from a
  daylight moment, because the underlying model works with daylight data
  — so for a night-time flag specifically, that left panel is showing a
  *different* moment of the same day than the one that actually
  triggered the flag on the right.)

Any other flag type currently shows a generic fallback line: *"This is a
statistical flag, not a confirmed finding — worth a manual check."*

---

## Tab 5 — Accuracy & Testing

**The question this tab answers:** "How do I know any of this actually
works?" This is the **only tab not affected by the sidebar's Plant/Period
selection** — it always shows the full, pooled result across both plants
and all 30 test cases each, because it's answering a question about the
system's overall track record, not about one specific period.

### The two headline stat tiles

- **"Fraud caught"** — of all the simulated fraud cases in the sealed
  test set (both plants, all 30 test cases each), the percentage that
  were correctly flagged. Hover over this tile for a one-line tooltip
  repeating this definition.
- **"Legitimate cases correctly left alone"** — of all the real-but-
  unusual, non-fraud test cases (equipment issues, cloudy days, grid
  power caps, etc.), the percentage that were correctly **not** flagged.
  Also has a hover tooltip.

**How to interpret these together:** these two numbers trade off against
each other — a system tuned to catch more fraud will generally also flag
more legitimate cases by mistake, and vice versa. Reporting both, rather
than just one, is what makes the numbers meaningful — a system that
"catches 100% of fraud" by flagging literally everything would score
terribly on the second number, and this dashboard would show that
clearly rather than letting the first number stand alone and mislead.

### The "how small a fraud can we catch" chart

A line chart with three lines, one for each of: **"A single moment"**
(a fake overstatement squeezed into just one 15-minute reading), **"One
hour"** (squeezed into a single hour), and **"A full day"** (spread
across the entire day). The x-axis is the size of the fake overstatement,
as a percentage of the plant's normal claimed power (from 1% up to 30%).
The y-axis is the chance of it actually being caught (0% to 100%).

**How to interpret this chart:** the "A full day" line is the one that
rises — starting near 0% at small sizes and climbing to roughly 78% by
the time the overstatement reaches 30% of claimed power for the whole
day. The other two lines (a single moment, one hour) stay flat at
essentially 0% across the **entire** tested range, even at 30% — this is
a genuine, disclosed limitation, not a rendering issue: the system's core
decision only ever looks at a full day's total energy, so a large but
brief spike gets diluted into an otherwise-normal day's total and doesn't
cross the detection threshold. If you're asked "what's the smallest fraud
you can catch," this chart is the literal, honest answer, and it's
worth pointing out unprompted — showing a real limitation clearly is more
convincing to a technical audience than only showing favorable numbers.

### The "How was this tested?" note

A short expander explaining, in two sentences, that the 30 test cases per
plant were built with sealed answers before any tuning happened, mixing
simulated fraud with real legitimate oddities that must not be mistaken
for fraud — repeating the "What's a test case?" explanation from the
sidebar section above, for anyone who lands on this tab first.

### What's deliberately *not* shown on this tab

Additional, real technical validation was done beyond what's shown here
— testing the system with the *other* plant's prediction model
(to check how well it generalizes), head-to-head comparisons against
simpler alternative detection methods, and a documented, honest account
of one specific fix that was attempted and found not to work. All of this
is real, genuinely useful evidence of rigor — but it was intentionally
left off the visible dashboard as more implementation detail than this
audience needs to see on screen. It's fully written up in
`docs/TASK7_FINDINGS.md` and `docs/TASK7_HANDOFF.md` for anyone
(including a technical panel member) who asks a deeper question.

---

## A running order for the demo

Roughly 5–7 minutes end to end. Set the sidebar to **Plant 1 → Test case
11** before you start.

1. **Open on Overview, state the one-liner.** *"This checks a solar
   plant's reported generation against an independent weather-only
   estimate, and prices any gap in carbon credits."*
2. **Point at the verdict banner and the flag chart** (Overview tab).
3. **Move to Evidence, point at one flagged stripe.** Trace the red
   (claimed) line pulling away from the blue (expected) line and its
   shaded range on the flagged day.
4. **Jump to Carbon Credits — don't bury this.** State the total $ and
   CO₂ figures, then read the "about these numbers" note out loud —
   owning a caveat yourself, unprompted, is the strongest thing you can
   do in front of an academic panel.
5. **Go to Flag Details, pick that same flagged day.** Walk through both
   panels — what was expected, then why it's suspicious.
6. **Finish on Accuracy & Testing.** State the two headline numbers, then
   show the "how small a fraud can we catch" chart and explain the flat
   lines in one sentence: a brief spike is much harder to catch than a
   sustained over-claim — a real, honestly-reported limitation.
7. **Switch the period to "Verified clean period," show it goes quiet.**
   Same dashboard, same plant, now on real, unmodified honest data, and
   the score drops with almost nothing flagged.
8. **Close** with one sentence connecting back to the opening line.

---

## Questions a panel is likely to ask

**"How do you know it's not just flagging everything?"**
It was tested against real, legitimate-but-unusual events too — it
correctly leaves 92% of those alone. Step 7 of the demo shows this live.

**"Why does the fraud-caught number look less than 100%?"**
A genuine, disclosed limit: some fraud patterns (very brief, or very
gradual) are only detectable at a full-day scale — the last tab's chart
shows exactly which sizes are and aren't currently catchable.

**"What if your weather-based model is just wrong?"**
Every prediction is cross-checked against a simpler, model-free physics
estimate as an independent sanity check, and accuracy is reported
separately for clear, partly-cloudy, and overcast conditions.

**"Why only two plants?"**
That's the real limit of the public dataset used, stated plainly rather
than hidden.

**"Could the plant just tamper with its own weather sensor to hide
fraud?"**
That's exactly why the core comparison uses independent satellite
weather, never the plant's own on-site sensor.

**"Did you find any cases you couldn't fix?"**
Yes, and it's documented, not hidden — one known edge case had two
different attempted fixes, both honestly found not to help, so both were
reverted rather than forced in.

---

## Before you start — checklist

- [ ] Dashboard running locally and reachable in the browser you'll
      present from — click through all five tabs once beforehand.
- [ ] Sidebar pre-set to `Plant 1 → Test case 11` as your starting point.
- [ ] Know which flagged day you'll open on Flag Details — pick it before
      you're in the room, don't decide live.
- [ ] Have the "switch to clean data" moment (step 7) rehearsed — it's
      your strongest single moment.

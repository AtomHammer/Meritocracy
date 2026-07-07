# Meritocracy Workbench

A simulation tool for studying **meritocratic filtering**: scientists carry hidden
quality traits and are passed through a sequence of competitive Applications
(e.g. an REU, then graduate admission, then a fellowship). The workbench tracks
how each filter reshapes the population — who applies, who is selected, and how
faithfully the selection reflects the underlying (hidden) merit.

It is built as a **directed acyclic graph of distributions**: you draw a starting
population, apply filters to produce derived populations, mix populations
together, and inspect any node in the resulting lineage.

## Running it

Requires Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser. No data files are needed to start — the
workbench begins empty. You can save the current state to a `.pkl` file and
re-upload it later using the buttons in the **Library** tab.

## The interface

Three tabs:

- **🌍 World** — configure the shared world: the quality dimensions and their
  distributions, reviewer accuracy per quality, the mapping from raw weighted
  quality to overall Potential, self-assessment accuracy, and productivity.
- **📚 Library** — the distributions and filters you have created, plus the
  operations that build new ones (fresh draw, apply a filter, mix
  distributions) and save/load.
- **🔬 Explorer** — per-distribution visualizations: lineage, filter artifacts,
  reviewer-perception-vs-reality scatters, per-quality drill-downs, Potential
  histograms, and a filter-mistakes breakdown.

## How the model works (brief)

Each scientist has hidden quality values plus an overall **Potential**. At each
Application:

1. **Self-selection** — scientists decide whether to apply based on a noisy
   self-assessment of their own Potential relative to the current pool.
2. **Reviewers** — a fresh reviewer pool scores applicants. Each reviewer has a
   per-quality accuracy; observed scores blend true quality with noise, then are
   calibrated onto an integer scale (a reviewer spreading their pile across
   `1..max_score`).
3. **Selection** — applicants are ranked by their weighted observed score and the
   top scientists (up to the filter's cutoff) are selected, forming a new
   population in the graph.

Reviewer and self-assessment accuracies use logit-Normal distributions;
per-population provenance (which path through the graph each scientist took) is
tracked so results can be broken down by lineage.

## Source layout

- `sim.py` — primitives, configuration dataclasses, and `run_application`
  (the per-Application scoring engine).
- `workbench.py` — the `Workbench` object, distributions, filters, the graph
  operations, save/load, and lineage tracking.
- `app.py` — the Streamlit user interface.

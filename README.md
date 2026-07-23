# CMU Engineering Research

This repository is the single interactive visualization of the central
[`ccm/cmu-engineering-publications`](https://huggingface.co/datasets/ccm/cmu-engineering-publications)
dataset.

It does not scrape Google Scholar, run embedding models, or commit generated
publication data. The Pages build resolves the dataset to an immutable Hugging
Face revision, validates its precomputed full-corpus `maps/publications.json`
artifact, and packages that artifact with the deployed site. The browser loads
the same-origin copy with normal HTTP caching and renders it with the
dependency-free canvas client in `assets/map.js`. It never computes embeddings
or layouts.

Every department uses the same semantic layouts. The interface opens with the
t-SNE view of local topical neighborhoods so clusters are immediately visible;
visitors can switch to the PCA view of broad global structure. Both are computed
centrally over the same deduplicated works and shipped as coordinates in the
artifact. Visitors can build removable filter pills for publication-title
terms, research topics, departments, and faculty authors; constrain publication
years with exact bounds or recent-year presets; choose between highlighting
matches in context and showing matches alone; color comparisons by title term,
department, or faculty author; and zoom to the resulting subset. Faculty,
department, and topic suggestions recalculate their counts from the other
active dimensions and omit impossible choices. Publications can also be colored
by year or citation count.
The year scale is linear across the central 98% of dated works; the citation
scale is logarithmic through the 99th percentile. Explicit `≤` and `≥` legend
endpoints retain outliers without letting them compress the rest of the corpus
into one color. Thirty overview keywords label the topical landscape directly
on the map; approximately 120 smaller nested topics fade in as visitors zoom.
At the t-SNE overview, up to three nested topics promote to primary labels when
their cluster is compact and its centroid is at least 3.5 cluster radii from the
nearest other nested topic. This normalized separation test exposes genuine
islands without relying on faculty identity or an arbitrary distance from a
broad parent. Filters recalculate topic support and centroids from the current
matches. A nested topic can promote when it covers at least 5% of the matches
(and at least four works) and accounts for at least 65% of its parent; the four
strongest promotions are kept, their over-broad parents demote, and labels
follow the active points. Each publication detail card shows its original
overview-to-detail topic path. Topic labels are actionable: pointing at one
explains its level, path, support, and promotion status; selecting it adds the
same topic filter available from the keyboard-accessible topic search.

Visitors can keep dots uniform or size them by oldest publication, newest
publication, or citation count. Year sizing uses the same robust central range
as year coloring; citation sizing uses a logarithmic scale through the robust
99th percentile. Unknown years stay small instead of being treated as old or
new. Eight quantized radii preserve the canvas renderer's batched performance,
and baseline point size still adapts to result density so full-corpus structure
and sparse searches both remain legible. Search dimensions combine with AND;
multiple values within a dimension combine with OR. “Zoom to matches” fits the
spatially central 98% so a few remote works do not flatten the main result;
“Show every match” restores exact bounds.

Every filtered or configured view has a bounded query-string representation
covering filters, years, layout, colors, sizes, and display mode. Visitors can
copy that permalink, inspect an accessible matching-publications drawer, sort
the matches, review leading-topic shares and basic coverage statistics, and
download the exact result as CSV.

The map remains the primary surface at every viewport size. Desktop controls
float over it; compact screens begin with an unobstructed map and expose
Settings, the Color key, and Results as mutually exclusive overlays. Persistent
zoom-in, zoom-out, and fit controls keep the canvas usable without a mouse
wheel, while the same operations remain available from the keyboard.

Faculty and department color modes assign every cataloged entity a unique,
stable color. The client generates each set at load time with deterministic
farthest-point sampling in OKLab, restricted to bright, in-gamut colors with
strong contrast against the map canvas. It does not cycle through a fixed
categorical palette. Searchable color keys list the faculty or departments
represented in the current matches and can also add or remove filters. For a
work with multiple connections, an actively selected entity takes color
priority; otherwise the first cataloged connection is used.

Year and citation modes use separate ordered, theme-aware color ramps generated
in OKLab. Their legends retain the full-corpus domain while filters are active,
so a color keeps the same meaning as visitors narrow the map.

Remote configuration and artifact data pass through a small, independently
tested parser before reaching the UI. Unsafe links are discarded, structural
schema errors produce a recoverable unavailable state, transient requests get
one restrained retry, and isolated malformed publication rows are omitted with
a visible warning instead of taking down the complete map.

The canonical faculty registry, collection policy, normalized dataset schema,
and artifact builder live in
[`cmccomb/map-of-research`](https://github.com/cmccomb/map-of-research).
The former department-specific `map-of-*` repositories are archived; their
views are now filters here.

The Pages deployment stamps CSS and JavaScript URLs with the deployed commit so
browsers cannot combine a new page with cached assets from an older release.
A daily workflow compares the current Hugging Face revision with the deployed
revision and rebuilds only when the data changed. Manual and `dataset-published`
dispatches can rebuild immediately. Code pushes can deploy only after the full
validation job succeeds; scheduled and dispatched builds run the same
validation before publishing. Each build also stages a label-calibration report
and enforces the three-label overview and four-label active-view budgets.

## Local preview

Build the same site artifact used by Pages, then serve it with any static file
server. The build downloads the current public dataset revision; opening
`index.html` directly may be blocked by browser cross-origin rules.

```bash
npm run build
python -m http.server 8000 --directory _site
```

Then open <http://localhost:8000>.

## Tests

Install the pinned development dependencies and run the complete suite:

```bash
npm ci
npx playwright install chromium firefox webkit
npm test
```

The suite includes parser and color-generation checks plus browser coverage for
filters, layouts, topical keywords, dot sizing, color keys, canvas controls,
failure recovery, accessibility, responsive widths, focused Firefox and WebKit
smokes, and a low-end-mobile performance budget against a production-sized
artifact with more than 30,000 points. GitHub Actions runs the same checks for
every pull request and every push to `main`. Pure client logic must maintain at
least 99% line, 97% branch, and 100% function coverage; browser tests exercise
the DOM and canvas integration around it.

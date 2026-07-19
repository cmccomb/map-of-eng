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
artifact. Visitors can build removable filter
pills for publication-title terms, departments, and faculty authors; choose
between highlighting matches in context and showing matches alone; color
comparisons by title term, department, or faculty author; and zoom to the
resulting subset. Publications can also be colored by year or citation count.
The year scale is linear across the central 98% of dated works; the citation
scale is logarithmic through the 99th percentile. Explicit `≤` and `≥` legend
endpoints retain outliers without letting them compress the rest of the corpus
into one color. Point size adapts to result density so full-corpus structure and
sparse searches both remain legible. Search dimensions combine with AND;
multiple values within a dimension combine with OR.

The map remains the primary surface at every viewport size. Desktop controls
float over it; compact screens begin with an unobstructed map and expose
Settings and the Color key as mutually exclusive overlays. Persistent zoom-in,
zoom-out, and fit controls keep the canvas usable without a mouse wheel, while
the same operations remain available from the keyboard.

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
dispatches can rebuild immediately.

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
npx playwright install chromium
npm test
```

The suite includes parser and color-generation checks plus browser coverage for
filters, layouts, color keys, canvas controls, failure recovery, accessibility,
responsive widths, and a production-sized 32,958-point artifact. GitHub Actions
runs the same checks for every pull request and every push to `main`.

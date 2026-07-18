# CMU Engineering Research

This repository is the single interactive visualization of the central
[`ccm/cmu-engineering-publications`](https://huggingface.co/datasets/ccm/cmu-engineering-publications)
dataset.

It does not scrape Google Scholar, run embedding models, or commit generated
publication data. The browser fetches the precomputed full-corpus
`maps/publications.json` artifact and renders it with the dependency-free canvas
client in `assets/map.js`. It never computes embeddings or layouts.

Every department uses the same semantic layouts. Visitors can switch between a
PCA view of broad global structure and a t-SNE view of local topical
neighborhoods. Both are computed centrally over the same deduplicated works and
shipped as coordinates in the artifact. Visitors can build removable filter
pills for publication-title terms, departments, and faculty authors; choose
between highlighting matches in context and showing matches alone; color
comparisons by title term, department, or faculty author; and zoom to the
resulting subset. Point size adapts to result density so full-corpus structure
and sparse searches both remain legible. Search dimensions combine with AND;
multiple values within a dimension combine with OR.

Faculty and department color modes assign every cataloged entity a unique,
stable color. The client generates each set at load time with deterministic
farthest-point sampling in OKLab, restricted to bright, in-gamut colors with
strong contrast against the map canvas. It does not cycle through a fixed
categorical palette. Searchable color keys list the faculty or departments
represented in the current matches and can also add or remove filters. For a
work with multiple connections, an actively selected entity takes color
priority; otherwise the first cataloged connection is used.

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

## Local preview

Serve this directory with any static file server. Opening `index.html` directly
may be blocked by browser cross-origin rules.

```bash
python -m http.server 8000
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

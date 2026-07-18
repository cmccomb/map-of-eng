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

Faculty color mode assigns every cataloged faculty member a unique color. The
client generates the set at load time with deterministic farthest-point
sampling in OKLab, restricted to bright, in-gamut colors with strong contrast
against the map canvas. It does not cycle through a fixed categorical palette.
The searchable color key only lists faculty represented in the current matches
and can also be used to add or remove faculty filters. For a work connected to
more than one faculty member, an actively selected faculty member takes color
priority; otherwise the first cataloged faculty connection is used.

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

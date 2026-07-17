# CMU Engineering Research

This repository is a thin static visualization of the central
[`ccm/cmu-engineering-publications`](https://huggingface.co/datasets/ccm/cmu-engineering-publications)
dataset.

It does not scrape Google Scholar, run embedding models, or commit generated
publication data. The browser fetches the precomputed `map-of-eng` map artifact
and renders it with the dependency-free canvas client in `assets/map.js`.

The canonical faculty registry, collection policy, normalized dataset schema,
and artifact builder live in
[`cmccomb/map-of-research`](https://github.com/cmccomb/map-of-research).

## Local preview

Serve this directory with any static file server. Opening `index.html` directly
may be blocked by browser cross-origin rules.

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>.

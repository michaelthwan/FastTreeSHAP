# TurboSHAP

*(formerly a FastTreeSHAP fork — import name `turboshap`)*

> Exact TreeSHAP, **13–17× faster than the [`shap`](https://github.com/slundberg/shap) package on a single core** — 100×+ with all cores — while producing SHAP values that are identical to `shap`'s within floating-point noise (measured max difference ~1e-14).

This is a performance-focused fork of [linkedin/FastTreeSHAP](https://github.com/linkedin/FastTreeSHAP) (the reference implementation of the [Fast TreeSHAP paper](https://arxiv.org/abs/2109.09847), NeurIPS 2021 XAI4Debugging). On top of the paper's v1/v2 algorithms it adds:

- **v3 "batched descent"** — a new execution model. Every earlier TreeSHAP variant re-walks each tree once *per sample*; v3 walks each tree **once**, carrying all samples through the traversal as vectors. Same exact math, applied in the same order — so the output is bitwise equal to v2's — but traversal cost is amortized over the batch and the hot loops become dense and branchless.
- **An optimized v2** (~1.4× over the published v2): incremental subset indexing, copy-free backtracking traversal, and slimmer recursion frames.
- **Working OpenMP on Windows** — the original build passed GCC's `-fopenmp` to MSVC, which silently ignored it; stock Windows builds were single-threaded.

## Benchmarks

sklearn RandomForestClassifier, 100 trees, 2,000 [Adult](https://archive.ics.uci.edu/ml/datasets/census+income) samples, AMD Ryzen 9 7900X, single core unless noted, vs `shap` 0.45:

|Tree depth|shap (native)|TurboSHAP v2|**TurboSHAP v3**|v3 · 24 threads|
|---------:|------------:|--------------------------:|------------------------------:|--------------:|
|8|3.63s|0.92s (4.0×)|**0.24s (15.1×)**|0.04s (~91×)|
|12|26.21s|7.67s (3.4×)|**1.57s (16.7×)**|0.16s (~164×)|

Speedups grow with tree depth and batch size. Full experiment history (including rejected optimizations) is in [speedup_report.html](speedup_report.html); an interactive explainer of how each algorithm generation works is in [algorithm_report.html](algorithm_report.html).

## Installation

Not yet on PyPI — install from source (requires a C++ compiler; MSVC on Windows, GCC/Clang elsewhere, `brew install libomp` on macOS):

```sh
pip install git+https://github.com/michaelthwan/TurboSHAP.git@v2-speedup
```

## Usage

Drop-in replacement for `shap.TreeExplainer` — same API, same outputs:

```python
import turboshap

explainer = turboshap.TreeExplainer(model, algorithm="v3", n_jobs=-1)

shap_values = explainer.shap_values(X)   # ndarray (n_samples, n_features), or a list per class
explanation = explainer(X)               # Explanation object, works with shap's plotting API
```

Supported models: scikit-learn tree ensembles (RandomForest, ExtraTrees, GradientBoosting, DecisionTree), XGBoost, LightGBM, CatBoost, and PySpark tree models.

### Choosing an algorithm

| `algorithm=` | When to use |
|---|---|
| `"v3"` **(recommended)** | Explaining a batch of samples (the more samples, the bigger the win). Needs the same per-thread precompute table as v2: `L·2^D` doubles per tree (~32 KB per leaf at depth 12) — fine up to depth ~14. |
| `"v2"` | Same table cost as v3; useful mainly as a reference or for tiny batches. |
| `"v1"` | Low memory (O(D²)) — the fallback for very deep trees or tight memory. |
| `"auto"` (default) | Conservatively picks v1/v2 based on batch size and available memory. It does **not** yet select v3 — pass `"v3"` explicitly. |

All algorithms produce the same SHAP values; the choice only affects speed and memory.

### Key parameters

- **`n_jobs`** (default `-1`): number of OpenMP threads; `-1` uses all cores. v3 parallelizes over trees.
- **`memory_tolerance`** (default `-1`): memory budget in GB for the v2/v3 precompute tables; default caps at 25% of system RAM. If the budget doesn't fit, the explainer automatically falls back (v2_1 → v2_2 → v1) with a warning.
- **`shortcut`** (default `False`): delegate to the TreeSHAP built into XGBoost/LightGBM/CatBoost instead of this package (always on for CatBoost, whose native path isn't ported yet).
- **`data` / `feature_perturbation`**: as in `shap` — default `"tree_path_dependent"` needs no background data.
- **Interaction values**: `shap_interaction_values(X)` is supported; with `algorithm="v3"` it falls back to v1 (with a warning), as interactions aren't batched yet.

## How it works

- **v1/v2** (from the [paper](https://arxiv.org/abs/2109.09847)): v1 restructures per-leaf Shapley-weight accounting (~1.5× vs TreeSHAP, same memory); v2 precomputes the weight sums for all 2^D feature subsets of each leaf path, once per tree, turning per-sample leaf work into table lookups — O(TL2^D + MTLD) time, O(L·2^D) memory.
- **This fork's v2 optimizations**: constant-factor engineering in the recursion — O(1) incremental subset indexing, mutate-and-undo instead of copy-on-descend, 8-argument calls instead of 21.
- **v3 batched descent**: the observation that along any root-to-leaf path, the path *structure* is sample-independent — only one pass/fail bit per split and a running residual differ per sample. So the traversal runs once per tree with per-level vectors over samples (subset-index bits, residuals), transposed inputs and outputs, and branchless leaf scale computation.

Every optimization was gated on producing outputs identical to a frozen `shap` reference (atol 1e-8; measured ~1e-14) across binary, multiclass, regression, and gradient-boosting models with additivity checks enabled. Changes that reordered *bookkeeping* were allowed; anything that would reorder *arithmetic* (FMA, reciprocal division) was rejected by design.

See [algorithm_report.html](algorithm_report.html) for an interactive walkthrough of all five generations.

## Citation

If this fork helps your research, please cite the original Fast TreeSHAP paper:

```
@article{yang2021fast,
  title={Fast TreeSHAP: Accelerating SHAP Value Computation for Trees},
  author={Yang, Jilei},
  journal={arXiv preprint arXiv:2109.09847},
  year={2021}
}
```

## Acknowledgements & License

Built on [linkedin/FastTreeSHAP](https://github.com/linkedin/FastTreeSHAP) by Jilei Yang and the [SHAP](https://github.com/slundberg/shap) package by Scott Lundberg. Original documentation (legacy benchmark tables, notebooks) lives in the upstream repository.

Copyright (c) LinkedIn Corporation. All rights reserved. Licensed under the [BSD 2-Clause](https://opensource.org/licenses/BSD-2-Clause) License.

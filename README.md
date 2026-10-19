# Neural recipe selection for logic synthesis

This project trains a small neural network to recommend ABC transformation recipes for an unseen combinational circuit. The network predicts 120 scores from circuit features in one pass. It executes one recommendation or a short list, replacing a new exhaustive search with a small synthesis budget.

Read [the paper](docs/final_paper.pdf) for the circuit concepts, dataset construction, experiments, and measured results. Its editable source is [paper/main.tex](paper/main.tex). Numerical tables and result paragraphs are generated from saved experiment outputs by [scripts/report.py](scripts/report.py) and [scripts/report_neural.py](scripts/report_neural.py).

## Set up

Python 3.10+ is required for the learning pipeline; Python 3.12+ supports the source archive extraction script. A C++ compiler, GNU Make, and curl are needed to rebuild ABC. The paper compiles with pdfLaTeX and Latin Modern fonts.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

[requirements.lock.txt](requirements.lock.txt) records the package versions used for the supplied experiments.

## Use the trained selector

The default model is `results/neural_model.joblib`: a 50-input MLP with hidden layers of 64 and 32 units and 120 recipe outputs. It contains the recipe library and the feature schema. Provide a combinational `.blif`, `.bench`, `.aig`, or `.aiger` file:

```bash
circuit-recipe select examples/ripple8.blif
circuit-recipe select examples/ripple8.blif --verify 3 \
  --netlist results/optimized.blif
```

ABC must be built at `tools/abc/abc`, and the mapping library must be present at `data/raw/nangate45.lib`. Input paths passed to ABC must contain no spaces, quotes, or semicolons. The first command returns a model-ranked recommendation. The second measures three recommended recipes and writes the best mapped circuit. The predicted score is a relative ranking score; the verification output contains measured area, delay, and the normalized objective.

Use `--model results/neural_static_model.joblib` for the 24-input graph-only NN, which skips diagnostic probes. The earlier Extra-Trees artifact remains at `results/model.joblib`. The command reports feature extraction, prediction, verification, and model-load times separately. Retrain with `circuit-recipe train`; use `--family nn_static --output results/neural_static_model.joblib` for the graph-only model or `--family classical --output results/model.joblib` for the earlier model comparison.

## Measured quality and compute

The NN uses the same 205 circuits, 120 recipes, and 15 outer folds as the classical experiments. One full-network recommendation gives **0.19% mean improvement** over the best fixed training recipe (95% circuit-bootstrap interval **−0.40% to 0.79%**). The graph-only network gives **0.44%** (interval **−0.28% to 1.10%**). Their average objectives are close to the fixed recipe; the direction of the mean difference remains uncertain.

Executing the full NN's top five recommendations gives **5.12% improvement** over the fixed recipe and **3.99% mean regret** relative to the exhaustive 120-recipe oracle. A global top-five shortlist gives **5.01% improvement** and **4.13% regret**. The classical Extra-Trees one-recipe result remains stronger than the NN one-recipe result. The paper reports these comparators alongside the saved NN predictions.

Fresh serial timings compare exhaustive search with NN and global shortlists on 12 circuits selected by family and AIG-size quartiles. Each timing circuit is unseen by the corresponding saved outer-fold model. All 1,440 recipe runs reproduce their dataset area and delay. [results/neural_runtime/summary.csv](results/neural_runtime/summary.csv) records mean elapsed times, speedups, and time saved; the paper explains the cost components. Timings include diagnostics and executed recipes with the model already loaded. The expensive synthesis table is an offline development cost shared by the NN and best-fixed baseline.

The serial audit averaged **44.84 s** for exhaustive search, **1.39 s** for one full-NN recommendation (**32.2× faster; 96.9% less time**), and **2.92 s** for the full NN's top five (**15.4× faster; 93.5% less time**). One graph-only recommendation took **0.77 s** (**58.0× faster**). A global top-five shortlist took **2.22 s**. These times describe the 12-circuit audit; quality figures above use all 205 circuits. A single fixed recipe remains the cheapest option at **0.41 s**.

## Reproduce the paper from the supplied dataset

```bash
python3 scripts/run_experiments.py
python3 scripts/run_neural.py
python3 scripts/benchmark_neural_runtime.py
python3 scripts/audit_neural.py
python3 scripts/audit_synthesis.py
python3 scripts/report.py
circuit-recipe train
python3 scripts/build_paper.py
python3 -m pytest -q
```

Evaluation saves each outer fold before moving on. Running the same command resumes completed folds. The full suite includes three repeats of five-fold nested circuit-grouped validation, five model variants, an inner-selected model family, margin gating, top-k verification, family transfer, feature/target/encoding ablations, library and training-set sizes, and a 24-held-out-recipe experiment. The secondary controlled comparisons use one five-fold repeat and the boosting model. Seeds, settings, split assignments, and tuning choices are saved beside the results.

The NN comparison uses mean per-circuit percentage improvement over the best fixed training recipe at equal area/delay weight. Confidence intervals bootstrap the 205 circuit averages after combining repeats. NN settings are selected through three inner circuit folds, choosing 25, 75, or 200 epochs with all preprocessing and target statistics fitted on training circuits. Both NN input variants use one circuit as one example with 120 outcomes. Learned shortlists are compared with global and random shortlists at the same candidate count; runtime includes feature collection. The serial NN timing audit is separate from the earlier five-worker sweep timings. Global shortlists with multiple candidates include the baseline mapping needed to compare their normalized objectives.

## Regenerate the synthesis dataset

```bash
python3 scripts/fetch_sources.py
make -C tools/abc -j4 ABC_USE_NO_READLINE=1
circuit-recipe sweep --workers 5
python3 scripts/audit_synthesis.py
```

The source script downloads pinned ABC, EPFL, and ISP RAS archives and checks the mapping-library checksum. The sweep reconstructs the original 205-circuit inventory, preserves the pilot's 60 recipes, and deterministically extends it to 120. Each candidate and each probe starts from the original circuit's hashed AIG. Area and delay are measured after the complete recipe and standard-cell mapping. The runner saves one checkpoint per circuit and stops on a synthesis or parsing failure.

The synthesis audit compares all 12,300 original pilot pairs with the corresponding regenerated pairs, checks all source-file hashes, reproduces 24 sampled pairs, and checks their mapped circuits for Boolean equivalence to their inputs.

## Files

| Path | Contents |
|---|---|
| `data/raw/dataset_pilot_v1.csv` | Original 205 × 60 pilot observations |
| `data/processed/dataset.csv` | Final 205 × 120 measured observations |
| `data/processed/circuits.csv` | Source paths, families, and file SHA-256 values |
| `data/processed/recipes.csv` | All operation sequences and explicit ABC commands |
| `data/processed/manifest.json` | Tool/source pins, library and dataset checksums, sweep cost |
| `results/p1/` | Main nested evaluation, per-circuit selections, rankings, and split records |
| `results/neural/` | NN nested evaluation, identical outer splits, tuning records, fold models and predictions |
| `results/neural_runtime/` | Serial 12-circuit timing audit and all 1,440 fresh measurements |
| `results/neural_model.joblib` | Full NN fitted on all circuits for subsequent use |
| `results/neural_static_model.joblib` | Graph-only NN fitted on all circuits |
| `results/neural_facts.json` | NN quality, confidence intervals, component outcomes, and runtime facts |
| `results/p2/`, `results/p3/` | Distribution-shift and family-held-out results |
| `results/features_*`, `target_*`, `encoding_*` | Controlled ablation outputs |
| `results/library_*`, `train_*`, `unseen_recipes/` | Sensitivity experiments |
| `results/model.joblib` | Model fitted on the complete dataset for subsequent use |
| `results/data_audit.json` | Pilot agreement, source integrity, equivalence audit |
| `results/paper_facts.json` | Primary result and interval in machine-readable form |
| `paper/generated/`, `paper/figures/` | Reproducible tables, prose, and vector figures |
| `vendor/pilot_scripts/` | Original bundled scripts preserved for reference |

Circuit features have `c_` prefixes; probe responses have `p_` prefixes. `base_area` and `base_delay` describe the no-recipe mapped circuit. `area` and `delay` describe the candidate's result. `yA = area/base_area`, `yD = delay/base_delay`, and `J_0.5 = (yA+yD)/2`. Runtime columns describe recorded wall time during the five-worker sweep.

The effective sample size for generalization is 205 circuits. All recipe rows for one circuit stay together during validation. Source-suite identity, circuit/recipe IDs, measured post-recipe outputs, and runtimes are excluded from model features.

## Sources

- [Berkeley ABC](https://github.com/berkeley-abc/abc), commit `5d64f3b5f3bb9db6f3d77636a65e99ba3ee9d799`.
- [EPFL benchmarks](https://github.com/lsils/benchmarks), commit `82d8cc6910419298e713a46644ed59fd3df53038`.
- [ISP RAS hdl-benchmarks](https://github.com/ispras/hdl-benchmarks), commit `10c3fb51d0810ce862659211862660dcc270a6af`; ISCAS-85 BENCH and MCNC combinational BLIF directories.
- [OpenROAD NanGate45 platform](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/platforms/nangate45), commit `7bcaf475c884e15f674e428aa04dbf92bd0426ed`.

Third-party source trees retain their upstream license and attribution files. The bundled `paper/etoolbox.sty` is the upstream LaTeX support package with its original LPPL notice.

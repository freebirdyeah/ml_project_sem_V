.PHONY: help test sources abc sweep evaluate neural runtime train report paper submission
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS := 1
export OMP_NUM_THREADS := 1
export MPLCONFIGDIR := /tmp/ml_proj_mpl

help:
	@echo "Targets: test sources abc sweep evaluate train report paper submission"

test:
	python3 -m pytest -q

sources:
	python3 scripts/fetch_sources.py

abc:
	$(MAKE) -C tools/abc -j4 ABC_USE_NO_READLINE=1

sweep:
	python3 -m circuit_recipe.cli sweep --workers 5

evaluate:
	python3 scripts/run_experiments.py

neural:
	python3 scripts/run_neural.py

runtime:
	python3 scripts/benchmark_neural_runtime.py

train:
	python3 -m circuit_recipe.cli train

report:
	python3 scripts/report.py

paper:
	python3 scripts/build_paper.py

submission:
	python3 scripts/package_submission.py

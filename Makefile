# Hyaline kinase — release entry points (branch: kinase-real-descriptors)
# No GPU, no retraining. Requires network to KLIFS on first run (then cached).

PY ?= python
ARGS ?= 2hyy

install:            ## install the kinase tool dependencies
	$(PY) -m pip install requests numpy scikit-learn pandas matplotlib pyarrow

analyze:            ## annotate a structure: make analyze ARGS="2hyy"
	$(PY) scripts/analyze_kinase.py $(ARGS)

descriptors:        ## compute geometric descriptors + Figure 1 (grouped LOKO)
	$(PY) scripts/kinase_descriptors.py

benchmark:          ## the defensible number: grouped leave-one-kinase-out
	$(PY) scripts/kinase_benchmark.py

atlas:              ## build the offline kinase atlas (HTML + parquet)
	$(PY) scripts/build_kinase_atlas.py

demo:               ## annotate 5 experimental + 2 AlphaFold structures (schema-validated)
	$(PY) scripts/demo_analyze_batch.py

audit:              ## reproducibility audit (sequence classifier, leaky vs grouped)
	$(PY) scripts/kinase_audit.py

verify:             ## smoke test: analyze + benchmark + atlas must all succeed
	$(PY) scripts/analyze_kinase.py 2hyy > /dev/null && echo "analyze  OK"
	$(PY) scripts/kinase_benchmark.py > /dev/null && echo "benchmark OK"
	$(PY) scripts/build_kinase_atlas.py > /dev/null && echo "atlas    OK"

.PHONY: install analyze descriptors benchmark atlas demo audit verify

# Youngstown's divided inheritance - full pipeline.
#
# Every target runs from the repository root; some scripts resolve paths
# relative to the working directory, so do not run them from inside src/.
#
#   make setup     one-time: python deps + the embedding model
#   make all       corpus -> paragraphs -> embeddings -> result -> figures
#   make result    the headline number, if data/proc is already built
#   make negative  the three null results the paper reports
#   make clean     delete derived artefacts (keeps raw sources)

PY := python3
SRC := src

.PHONY: all setup corpus segment embed result robust negative figures clean distclean

# ---------------------------------------------------------------- setup ----
setup:
	$(PY) -m pip install -r requirements.txt
	./scripts/get_model.sh

# --------------------------------------------------------------- corpus ----
# Pre-2011 opinions from the Caselaw Access Project; 2014+ from the Court's
# own PDFs. Both are skipped if the files are already present.
corpus:
	$(PY) $(SRC)/fetch_cap.py
	$(PY) $(SRC)/fetch_modern.py
	$(PY) $(SRC)/refetch_pdfs.py

# Paragraph-level dataset with case/opinion/author/type metadata.
segment: data/proc/paragraphs.jsonl
data/proc/paragraphs.jsonl: $(SRC)/segment.py $(SRC)/manifest.py
	$(PY) $(SRC)/segment.py

embed: data/proc/embeddings.npy
data/proc/embeddings.npy: data/proc/paragraphs.jsonl $(SRC)/embed.py
	$(PY) $(SRC)/embed.py

# --------------------------------------------------------------- result ----
# The paper's finding: within-case difference-in-differences.
result: data/proc/embeddings.npy
	$(PY) $(SRC)/within.py

# Placebo partition, role permutation, quotation, boundaries, thresholds.
robust: data/proc/embeddings.npy
	$(PY) $(SRC)/within_robust.py

# ------------------------------------------------------------- negatives ----
# Reported in the paper as limits on what text models can do here.
negative: data/proc/embeddings.npy
	$(PY) $(SRC)/classify.py      # stance classifier, AUC 0.540
	$(PY) $(SRC)/diagnose.py      # stance vs role vs era
	$(PY) $(SRC)/fe.py            # pooled deference direction, AUC 0.485
	$(PY) $(SRC)/perm.py          # its permutation null, p = 0.62
	$(PY) $(SRC)/domains.py       # per-domain directions
	$(PY) $(SRC)/drift.py         # no time trend, p = 0.31

# --------------------------------------------------------------- figures ----
figures: out/figure3_inheritance.pdf out/figure2_limits.pdf
out/figure3_inheritance.pdf: data/proc/within.json data/proc/within_robust.json $(SRC)/figure3.py
	$(PY) $(SRC)/figure3.py
out/figure2_limits.pdf: data/proc/diagnostics.json $(SRC)/figure.py
	$(PY) $(SRC)/figure.py

data/proc/within.json: data/proc/embeddings.npy $(SRC)/within.py
	$(PY) $(SRC)/within.py
data/proc/within_robust.json: data/proc/embeddings.npy $(SRC)/within_robust.py
	$(PY) $(SRC)/within_robust.py
data/proc/diagnostics.json: data/proc/embeddings.npy $(SRC)/diagnose.py
	$(PY) $(SRC)/diagnose.py

all: corpus segment embed result robust figures

# ----------------------------------------------------------------- clean ----
clean:
	rm -rf data/proc $(SRC)/__pycache__

distclean: clean
	rm -rf models out/*.png out/*.pdf

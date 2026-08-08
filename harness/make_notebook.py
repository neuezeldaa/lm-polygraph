#!/usr/bin/env python3
"""Generates notebooks/spilled_energy_colab_t4.ipynb and validates it.

The notebook is GENERATED, never hand-edited: hand-editing cells in Colab is how
the wrong code ends up in a run. Every cell only orchestrates -- all real logic
lives in harness/*.py, so nothing here can drift from the code under test.

Run:  python harness/make_notebook.py
"""

import re
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "spilled_energy_colab_t4.ipynb"

REPO_URL = "https://github.com/neuezeldaa/lm-polygraph"
BRANCH = "spilled-energy-experiments"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
CFG_BASE = "configs/stage1/eval_triviaqa_qwen.yaml"
CFG_LADDER = "configs/stage1/eval_triviaqa_ladder.yaml"

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))


md(
    "# Spilled Energy — TriviaQA on a T4\n"
    "\n"
    "Run the cells **in order**. Each gate is designed to fail fast and loudly:\n"
    "a bad prompt format, a silent model swap, or drifting generations all stop the\n"
    "notebook rather than producing numbers that look fine and mean nothing.\n"
    "\n"
    "Two runs, deliberately: the **primary baseline table** and the **ablation\n"
    "ladder**. Both save to Drive as they finish, and cell 11 asserts they saw\n"
    "byte-identical generations — otherwise the two tables are not comparable.\n"
    "\n"
    "**Runtime → Change runtime type → T4 GPU** before starting."
)

# 1 --------------------------------------------------------------------------
md("## 1. Confirm you actually got a T4\n\nColab hands out different accelerators. Check before spending an hour.")
code(
    "import torch, subprocess\n"
    "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version',\n"
    "                      '--format=csv'], capture_output=True, text=True).stdout)\n"
    "assert torch.cuda.is_available(), 'No GPU! Runtime -> Change runtime type -> T4 GPU'\n"
    "name = torch.cuda.get_device_name(0)\n"
    "free, total = torch.cuda.mem_get_info()\n"
    "print(f'device    : {name}')\n"
    "print(f'VRAM free : {free/1e9:.2f} GB / {total/1e9:.2f} GB')\n"
    "if 'T4' not in name:\n"
    "    print(f'\\nWARNING: expected a T4, got {name!r}. Results stay valid; timings differ.')"
)

# 2 --------------------------------------------------------------------------
md(
    "## 2. Mount Drive\n"
    "\n"
    "Every run writes **straight to Drive**. lm-polygraph saves the manager inside a\n"
    "`finally:` block, so a run that raises still leaves its results behind, and each\n"
    "run lands before the next begins — a disconnect costs one run, not all of them."
)
code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "from pathlib import Path\n"
    "DRIVE_OUT = Path('/content/drive/MyDrive/spilled_energy/runs')\n"
    "DRIVE_OUT.mkdir(parents=True, exist_ok=True)\n"
    "DRIVE = DRIVE_OUT.as_posix()   # posix form, for shell interpolation\n"
    "print('results ->', DRIVE)"
)

# 3 --------------------------------------------------------------------------
md(
    "## 3. Clone the experiments branch\n"
    "\n"
    "`spilled-energy-experiments` = the PR branch **plus** `harness/` and this notebook.\n"
    "The PR is opened from `spilled-energy`, which holds only the StatCalculator, the\n"
    "Estimator, the tests and the configs."
)
code(
    f"REPO_URL = {REPO_URL!r}\n"
    f"BRANCH   = {BRANCH!r}\n"
    "REPO     = '/content/lm-polygraph'\n"
    "\n"
    "import os\n"
    "if not os.path.isdir(REPO):\n"
    "    !git clone --branch $BRANCH $REPO_URL $REPO\n"
    "%cd $REPO\n"
    "!git log -1 --oneline"
)

# 4 --------------------------------------------------------------------------
md(
    "## 4. Install, and set PYTHONPATH explicitly\n"
    "\n"
    "`PYTHONPATH` is set here rather than left to chance: the ablation ladder loads\n"
    "`harness.pooled_baseline` by dotted path from inside a `polygraph_eval`\n"
    "subprocess, which does not inherit the notebook's `sys.path`."
)
code(
    "!pip install -q -e .\n"
    "!pip install -q -r harness/requirements-repro.txt\n"
    "\n"
    "import os, sys, sysconfig\n"
    "os.environ['PYTHONPATH'] = REPO + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
    "os.environ['PATH'] = os.environ['PATH'] + os.pathsep + sysconfig.get_path('scripts')\n"
    "if REPO not in sys.path:\n"
    "    sys.path.insert(0, REPO)\n"
    "print('PYTHONPATH =', os.environ['PYTHONPATH'])\n"
    "!which polygraph_eval || echo 'NOTE: not on PATH; run_baselines.py prints a fallback'"
)

# 5 --------------------------------------------------------------------------
md(
    "## 5. Fail-fast import check\n"
    "\n"
    "Five seconds here beats discovering a broken import inside a subprocess after the\n"
    "3B model has loaded. Also proves the subprocess itself can resolve the dotted path."
)
code(
    "import subprocess, sys, os\n"
    "\n"
    "from harness.pooled_baseline import PooledBaseline          # in-process\n"
    "from lm_polygraph.estimators import SpilledEnergy\n"
    "from lm_polygraph.stat_calculators import EnergyCalculator\n"
    "print('in-process imports OK:', str(PooledBaseline(score='log_likelihood', pooling='max')))\n"
    "\n"
    "# the path that actually matters: a fresh subprocess, as polygraph_eval will be\n"
    "r = subprocess.run([sys.executable, '-c',\n"
    "                    'from lm_polygraph.utils.factory_estimator import FactoryEstimator;'\n"
    "                    'e=FactoryEstimator()(\"harness.pooled_baseline\",'\n"
    "                    '{\"score\":\"log_likelihood\",\"pooling\":\"max\"});print(\"subprocess OK:\",e)'],\n"
    "                   capture_output=True, text=True, env=dict(os.environ))\n"
    "print(r.stdout.strip() or r.stderr.strip())\n"
    "assert r.returncode == 0, 'subprocess cannot import harness.* -- fix PYTHONPATH before running anything'"
)

# 6 --------------------------------------------------------------------------
md(
    "## 6. Model provenance\n"
    "\n"
    "Printed before anything loads the model. `--expect` makes a silent model swap a\n"
    "hard failure rather than a footnote. Confirm `model.path`, `dtype` and `device`\n"
    "against the report."
)
code(f"!python harness/provenance.py --config {CFG_BASE} --expect {MODEL}")

# 7 --------------------------------------------------------------------------
md("## 7. Unit tests (CPU, seconds)\n\nCheap proof the install is sane before any long run.")
code("!python -m pytest test/test_spilled_energy.py -q")

# 8 --------------------------------------------------------------------------
md(
    "## 8. Dev run, n=150 — the gates fire here\n"
    "\n"
    "This one short run does four jobs:\n"
    "\n"
    "1. **Accuracy gate** — hard-fails outside 10–90% exact match. Outside that band\n"
    "   PRR is noise and every downstream number is meaningless. The usual cause is\n"
    "   prompt format.\n"
    "2. **Sign check** — a wrong sign shows up as a large *negative* normalized PRR\n"
    "   (~-0.7), which reads as a broken method rather than an inverted score.\n"
    "3. **Answer-span validation** (cell 9).\n"
    "4. **Runtime measurement** (cell 10).\n"
    "\n"
    "If the accuracy gate fails, **stop and fix the prompt** — do not widen the band."
)
code(
    "cmd = ('python harness/run_baselines.py'\n"
    f"       ' --config {CFG_BASE}'\n"
    "       f\" --save-dir '{DRIVE}/dev_n150'\"\n"
    "       ' --samples 150 --n-boot 0'\n"
    f"       ' --expect-model {MODEL}')\n"
    "print(cmd)\n"
    "!{cmd}"
)

# 9 --------------------------------------------------------------------------
md(
    "## 9. Validate the answer-span assumption\n"
    "\n"
    "The ablation ladder defines the answer window as the whole generation. That is\n"
    "only defensible if the generation really is a short answer — measured, not\n"
    "asserted. Reports single-line fraction, length distribution, ceiling-truncation\n"
    "rate and exact-match rate.\n"
    "\n"
    "**Re-run this for CoQA.** The assumption may not transfer."
)
code(
    "cmd = ('python harness/validate_answer_span.py'\n"
    "       f\" --save-dir '{DRIVE}/dev_n150'\"\n"
    f"       ' --config {CFG_BASE}'\n"
    "       ' --max-new-tokens 20')\n"
    "!{cmd}"
)

# 10 -------------------------------------------------------------------------
md(
    "## 10. How long will the real runs take?\n"
    "\n"
    "Projected from the measured n=150 pass, **before** committing to the long runs.\n"
    "Linear in n, and it double-counts fixed startup, so it slightly overestimates."
)
code(
    "cmd = ('python harness/estimate_runtime.py'\n"
    "       f\" --from '{DRIVE}/dev_n150'\"\n"
    "       ' --label baselines_n1000 --to-n 1000')\n"
    "!{cmd}\n"
    "print('\\nNOTE: the ladder run has fewer estimators but the same generation cost,')\n"
    "print('so budget roughly the same again for run B.')"
)

# 11 -------------------------------------------------------------------------
md(
    "## 11. Run A — primary baseline table, n=1000\n"
    "\n"
    "The mechanically derived `single_pass_cheap` + `single_pass_plus_aux_model` tiers\n"
    "**and** all Spilled Energy variants, in one `UEManager` — so generations and\n"
    "generation settings are identical by construction.\n"
    "\n"
    "If the sign check flagged a variant as inverted, set `cfg: {sign: -1}` in\n"
    "`configs/stage1/estimators/stage1_baselines.yaml` and commit. Do not patch it here."
)
code(
    "cmd = ('python harness/run_baselines.py'\n"
    f"       ' --config {CFG_BASE}'\n"
    "       f\" --save-dir '{DRIVE}/A_baselines_n1000'\"\n"
    "       ' --n-boot 1000'\n"
    f"       ' --expect-model {MODEL}')\n"
    "print(cmd)\n"
    "!{cmd}"
)

# 12 -------------------------------------------------------------------------
md(
    "## 12. Run B — ablation ladder, n=1000\n"
    "\n"
    "Same window, same three poolings on every rung, so adjacent rungs differ by\n"
    "exactly one ingredient: pooled log-likelihood → E^l → E^m → ΔE → ΔE_s.\n"
    "\n"
    "Run A is already saved to Drive before this starts."
)
code(
    "cmd = ('python harness/run_baselines.py'\n"
    f"       ' --config {CFG_LADDER}'\n"
    "       f\" --save-dir '{DRIVE}/B_ladder_n1000'\"\n"
    "       ' --n-boot 1000'\n"
    f"       ' --expect-model {MODEL}')\n"
    "print(cmd)\n"
    "!{cmd}"
)

# 13 -------------------------------------------------------------------------
md(
    "## 13. Are the two tables comparable? — hard gate\n"
    "\n"
    "Greedy decoding at a fixed seed *should* make the two runs byte-identical, but\n"
    "they resolve different stat calculators, and fp16 reductions are not associative.\n"
    "So it is **verified, not trusted**: sha256 of the generations and the full\n"
    "quality vector must match exactly.\n"
    "\n"
    "If this fails, the two tables are about different generations and must not be\n"
    "placed side by side — the fix is to make the configs agree on\n"
    "`output_attentions` and re-run, not to proceed."
)
code(
    "cmd = ('python harness/check_run_consistency.py'\n"
    "       f\" --a '{DRIVE}/A_baselines_n1000'\"\n"
    "       f\" --b '{DRIVE}/B_ladder_n1000'\"\n"
    "       ' --label-a baselines --label-b ladder')\n"
    "!{cmd}"
)

# 14 -------------------------------------------------------------------------
md(
    "## 14. The reported tables\n"
    "\n"
    "Primary metric is **normalized PRR@0.5** with bootstrap CIs. Everything on Drive,\n"
    "so tables can be rebuilt offline on CPU with\n"
    "`python harness/run_baselines.py --skip-run --save-dir <dir>`."
)
code(
    "from IPython.display import Markdown, display\n"
    "for tag, label in [('A_baselines_n1000', 'PRIMARY BASELINE TABLE'),\n"
    "                   ('B_ladder_n1000',   'ABLATION LADDER')]:\n"
    "    p = DRIVE_OUT / tag / 'prr_0.5_table.md'\n"
    "    display(Markdown(f'# {label}'))\n"
    "    display(Markdown(p.read_text() if p.exists() else f'_missing: {p}_'))"
)
code(
    "import torch, transformers, datasets, sys\n"
    "print('python      ', sys.version.split()[0])\n"
    "print('torch       ', torch.__version__)\n"
    "print('transformers', transformers.__version__)\n"
    "print('datasets    ', datasets.__version__)\n"
    "print('device      ', torch.cuda.get_device_name(0))\n"
    "!git rev-parse HEAD"
)

nb = new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
})

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))

# --- validate what was written, from disk ---------------------------------
reread = nbf.read(str(OUT), as_version=4)
nbf.validate(reread)

errs = 0
for i, c in enumerate(reread.cells):
    if c.cell_type != "code":
        continue
    # shell/magic lines are not Python; keep indentation so blocks stay valid
    py = "\n".join(
        re.match(r"\s*", ln).group(0) + "pass" if ln.strip().startswith(("!", "%")) else ln
        for ln in c.source.split("\n")
    )
    try:
        compile(py, f"<cell {i}>", "exec")
    except SyntaxError as e:
        errs += 1
        print(f"SYNTAX ERROR in cell {i}: {e}\n{c.source}\n")

heads = [c.source.split("\n")[0] for c in reread.cells if c.cell_type == "markdown"
         and c.source.startswith("## ")]
for n, h in enumerate(heads, 1):
    assert h.startswith(f"## {n}."), f"heading out of order: {h!r} (expected {n})"

print(f"wrote + validated {OUT}")
print(f"  cells={len(reread.cells)}  code={sum(1 for c in reread.cells if c.cell_type=='code')}"
      f"  syntax_errors={errs}  headings={len(heads)} (sequential)")
sys.exit(1 if errs else 0)

#!/usr/bin/env python3
"""Generates notebooks/spilled_energy_colab_t4.ipynb and validates it.

The notebook is generated (not hand-edited) so it stays valid and so all real
logic keeps living in harness/*.py -- cells only orchestrate.
"""

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "spilled_energy_colab_t4.ipynb"

REPO_URL = "https://github.com/<your-user>/lm-polygraph"
BRANCH = "spilled-energy-experiments"

cells = []

cells.append(new_markdown_cell(
    "# Spilled Energy — T4 run\n"
    "\n"
    "Runs **all estimators (baselines + SpilledEnergy) in a single UEManager pass**\n"
    "on the frozen config, and reports **normalized PRR@0.5**.\n"
    "\n"
    "**Runtime → Change runtime type → T4 GPU** first. All logic lives in\n"
    "`harness/*.py` in the repo; these cells only orchestrate, so nothing here can\n"
    "drift from the code in the PR."
))

# --- cell 1: confirm the hardware -----------------------------------------
cells.append(new_markdown_cell(
    "## 1. Confirm you actually got a T4\n"
    "\n"
    "Colab silently hands out different accelerators. Check before spending an hour."
))
cells.append(new_code_cell(
    "import torch, subprocess\n"
    "print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version',\n"
    "                      '--format=csv'], capture_output=True, text=True).stdout)\n"
    "assert torch.cuda.is_available(), 'No GPU! Runtime -> Change runtime type -> T4 GPU'\n"
    "name = torch.cuda.get_device_name(0)\n"
    "free, total = torch.cuda.mem_get_info()\n"
    "print(f'device      : {name}')\n"
    "print(f'VRAM free   : {free/1e9:.2f} GB / {total/1e9:.2f} GB')\n"
    "if 'T4' not in name:\n"
    "    print(f'\\nWARNING: expected a T4, got {name!r}. Timings and the fp16-only\\n'\n"
    "          'constraint were chosen for a T4; results are still valid but speed differs.')"
))

# --- cell 2: Drive ---------------------------------------------------------
cells.append(new_markdown_cell(
    "## 2. Mount Drive\n"
    "\n"
    "Results are written **straight to Drive**, not copied at the end: the manager is\n"
    "saved inside lm-polygraph's `finally:` block, so it survives even if the run\n"
    "raises. Losing a long run to a disconnect is the most likely way this goes wrong."
))
cells.append(new_code_cell(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "\n"
    "from pathlib import Path\n"
    "DRIVE_OUT = Path('/content/drive/MyDrive/spilled_energy/runs')\n"
    "DRIVE_OUT.mkdir(parents=True, exist_ok=True)\n"
    "print('results ->', DRIVE_OUT)"
))

# --- cell 3: clone ---------------------------------------------------------
cells.append(new_markdown_cell(
    "## 3. Clone the experiments branch\n"
    "\n"
    "`spilled-energy-experiments` = the PR branch **plus** `harness/` and this notebook.\n"
    "The PR itself is opened from `spilled-energy`, which contains only the\n"
    "StatCalculator, the Estimator, the tests and the config."
))
cells.append(new_code_cell(
    f"REPO_URL = {REPO_URL!r}  # <-- EDIT: your fork\n"
    f"BRANCH   = {BRANCH!r}\n"
    "\n"
    "import os\n"
    "if not os.path.isdir('/content/lm-polygraph'):\n"
    "    !git clone --branch $BRANCH $REPO_URL /content/lm-polygraph\n"
    "%cd /content/lm-polygraph\n"
    "!git log -1 --oneline"
))

# --- cell 4: install -------------------------------------------------------
cells.append(new_markdown_cell(
    "## 4. Install\n"
    "\n"
    "Colab already ships a CUDA torch; we install lm-polygraph and pin the rest.\n"
    "Restart the runtime **only** if pip reports a version conflict it had to resolve."
))
cells.append(new_code_cell(
    "!pip install -q -e .\n"
    "!pip install -q -r harness/requirements-repro.txt\n"
    "\n"
    "import sysconfig, os\n"
    "os.environ['PATH'] = os.environ['PATH'] + ':' + sysconfig.get_path('scripts')\n"
    "!which polygraph_eval || echo 'NOTE: not on PATH; the runner prints a fallback'"
))

# --- cell 5: unit tests ----------------------------------------------------
cells.append(new_markdown_cell(
    "## 5. Run the unit tests (seconds, CPU)\n"
    "\n"
    "Cheap guard that the install is sane before the long run."
))
cells.append(new_code_cell(
    "!python -m pytest test/test_spilled_energy.py -q"
))

# --- cell 6: sign check ----------------------------------------------------
cells.append(new_markdown_cell(
    "## 6. Fix the sign on a dev subset\n"
    "\n"
    "A wrong sign yields a large **negative** normalized PRR (~-0.7), which reads as a\n"
    "broken method rather than an inverted score. Settle it on ~150 examples before\n"
    "the real run. The accuracy gate also fires here first — if exact match is outside\n"
    "10–90%, **stop and fix the prompt**, because PRR is noise outside that band."
))
cells.append(new_code_cell(
    "cmd = (\"python harness/run_baselines.py\"\n"
    "       \" --config configs/stage1/eval_triviaqa_qwen.yaml\"\n"
    "       f\" --save-dir '{DRIVE_OUT}/dev_signcheck'\"\n"
    "       \" --samples 150 --n-boot 0\")\n"
    "print(cmd)\n"
    "!{cmd}"
))

# --- cell 7: final run -----------------------------------------------------
cells.append(new_markdown_cell(
    "## 7. Final run — all estimators, one pass, n=1000\n"
    "\n"
    "Baselines and SpilledEnergy share one `UEManager`, so generations and generation\n"
    "settings are identical by construction (fp16 + different batch composition can\n"
    "otherwise shift generations between runs).\n"
    "\n"
    "If the sign check said a variant is inverted, set it in\n"
    "`configs/stage1/estimators/stage1_baselines.yaml` (`cfg: {sign: -1}`) and commit —\n"
    "do not patch it here."
))
cells.append(new_code_cell(
    "cmd = (\"python harness/run_baselines.py\"\n"
    "       \" --config configs/stage1/eval_triviaqa_qwen.yaml\"\n"
    "       f\" --save-dir '{DRIVE_OUT}/final_n1000'\"\n"
    "       \" --n-boot 1000\")\n"
    "print(cmd)\n"
    "!{cmd}"
))

# --- cell 8: show the table ------------------------------------------------
cells.append(new_markdown_cell(
    "## 8. The reported table\n"
    "\n"
    "`per_sample_seed1.npz` is on Drive too — the table can be rebuilt offline on CPU\n"
    "with `python harness/run_baselines.py --skip-run --save-dir <dir>`."
))
cells.append(new_code_cell(
    "from IPython.display import Markdown, display\n"
    "display(Markdown((DRIVE_OUT / 'final_n1000' / 'prr_0.5_table.md').read_text()))\n"
    "\n"
    "import torch, transformers, datasets, sys\n"
    "print('python      ', sys.version.split()[0])\n"
    "print('torch       ', torch.__version__)\n"
    "print('transformers', transformers.__version__)\n"
    "print('datasets    ', datasets.__version__)\n"
    "print('device      ', torch.cuda.get_device_name(0))\n"
    "!git rev-parse HEAD"
))

nb = new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
})

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))

# validate what we just wrote, from disk
reread = nbf.read(str(OUT), as_version=4)
nbf.validate(reread)
print(f"wrote + validated {OUT}  ({len(reread.cells)} cells)")
sys.exit(0)

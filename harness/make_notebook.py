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
CFG_ATTN = "configs/stage1/eval_triviaqa_attention_bs1.yaml"

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))


md(
    "# Spilled Energy — TriviaQA on a T4\n"
    "\n"
    "**Runtime → Change runtime type → T4 GPU**, then `Runtime → Run all`.\n"
    "\n"
    "The notebook restarts its own kernel once, after the install (cell 5). That is\n"
    "not optional: `pip install -e .` writes an `__editable__*.pth` into\n"
    "site-packages, and `.pth` files are only processed at interpreter startup, so\n"
    "`lm_polygraph` cannot be imported in a kernel that was already running. After\n"
    "the restart, just **run all again from the top** — every setup cell is\n"
    "idempotent and the second pass takes seconds.\n"
    "\n"
    "Each gate fails fast and loudly: a stale module, a silent model swap, a bad\n"
    "prompt format, or drifting generations all stop the notebook rather than\n"
    "producing numbers that look fine and mean nothing."
)

# 1 --------------------------------------------------------------------------
md(
    "## 1. Confirm you actually got a T4\n"
    "\n"
    "Deliberately does **not** import torch. The install can change the torch version,\n"
    "and a module imported now would stay bound in this kernel while every subprocess\n"
    "got the new one — exactly the split this notebook exists to avoid. torch is\n"
    "asserted in cell 7, after the restart."
)
code(
    "import subprocess\n"
    "out = subprocess.run(['nvidia-smi',\n"
    "                      '--query-gpu=name,memory.total,memory.free,driver_version',\n"
    "                      '--format=csv'], capture_output=True, text=True).stdout\n"
    "print(out)\n"
    "assert out.strip(), 'nvidia-smi produced nothing: Runtime -> Change runtime type -> T4 GPU'\n"
    "if 'T4' not in out:\n"
    "    print('WARNING: expected a T4. Results stay valid; timings will differ.')"
)

# 2 --------------------------------------------------------------------------
md(
    "## 2. Mount Drive  *(idempotent)*\n"
    "\n"
    "Every run writes **straight to Drive**. lm-polygraph saves the manager inside a\n"
    "`finally:` block, so even a run that raises leaves its results behind, and each\n"
    "run lands before the next starts — a disconnect costs one run, not all of them.\n"
    "\n"
    "The mount is a VM-level FUSE process, so it survives the kernel restart;\n"
    "re-running this cell then just reports it is already mounted."
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
    "## 3. Clone the experiments branch  *(idempotent)*\n"
    "\n"
    "`spilled-energy-experiments` = the PR branch **plus** `harness/` and this notebook.\n"
    "The PR is opened from `spilled-energy`, which holds only the StatCalculator, the\n"
    "Estimator, the tests and the configs.\n"
    "\n"
    "Guarded on `isdir`, so the second pass is instant. It also pulls, so re-running\n"
    "the notebook picks up any pushed fixes without a fresh clone."
)
code(
    f"REPO_URL = {REPO_URL!r}\n"
    f"BRANCH   = {BRANCH!r}\n"
    "REPO     = '/content/lm-polygraph'\n"
    "\n"
    "import os\n"
    "if not os.path.isdir(REPO):\n"
    "    !git clone --branch $BRANCH $REPO_URL $REPO\n"
    "else:\n"
    "    print('already cloned; pulling latest')\n"
    "    !git -C $REPO pull --ff-only origin $BRANCH\n"
    "%cd $REPO\n"
    "!git log -1 --oneline"
)

# 4 --------------------------------------------------------------------------
md(
    "## 4. Environment paths  *(idempotent, must run in every kernel)*\n"
    "\n"
    "`PYTHONPATH` carries the repo root so a `polygraph_eval` subprocess can import\n"
    "`harness.pooled_baseline` by dotted path. Environment variables do **not** survive\n"
    "a kernel restart, so this cell has to run again on the second pass — which is why\n"
    "it is separate from the install."
)
code(
    "import os, sys, sysconfig\n"
    "os.environ['PYTHONPATH'] = REPO + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
    "scripts = sysconfig.get_path('scripts')\n"
    "if scripts not in os.environ['PATH']:\n"
    "    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + scripts\n"
    "if REPO not in sys.path:\n"
    "    sys.path.insert(0, REPO)\n"
    "print('PYTHONPATH =', os.environ['PYTHONPATH'])"
)

# 5 --------------------------------------------------------------------------
md(
    "## 5. Install  *(idempotent — skips entirely on the second pass)*\n"
    "\n"
    "Short-circuits when `lm_polygraph` is already importable, so after the restart\n"
    "this costs nothing.\n"
    "\n"
    "Note that torch is **not** pinned to an exact version. An earlier revision pinned\n"
    "`torch==2.6.0`, which forced a multi-GB downgrade of Colab's build and broke\n"
    "torchvision, the CUDA/driver match, and left the kernel holding a different torch\n"
    "than its subprocesses. Upstream only requires `>=2.6.0`, which Colab's build\n"
    "already satisfies."
)
code(
    "import importlib.util\n"
    "\n"
    "if importlib.util.find_spec('lm_polygraph') is not None:\n"
    "    print('lm_polygraph already importable -- skipping install')\n"
    "else:\n"
    "    print('installing (expect a few minutes on the first pass)')\n"
    "    !pip install -q -e .\n"
    "    !pip install -q -r harness/requirements-repro.txt\n"
    "\n"
    "!which polygraph_eval || echo 'NOTE: not on PATH; run_baselines.py prints a fallback'"
)

# 6 --------------------------------------------------------------------------
md(
    "## 6. Restart the kernel — READ THIS\n"
    "\n"
    "Restarts **only if the in-process state is actually stale**, so it cannot loop:\n"
    "on the second pass everything imports and this cell just prints OK and moves on.\n"
    "\n"
    "It restarts when either\n"
    "\n"
    "* `lm_polygraph` is not importable in-process (the `.pth` was written after this\n"
    "  kernel started), or\n"
    "* the imported `torch.__version__` differs from the version pip has on disk\n"
    "  (a stale module object).\n"
    "\n"
    "### When it restarts, Colab will say the session crashed. That is expected.\n"
    "### Just run `Runtime → Run all` again. Cells 1–5 will no-op."
)
code(
    "import importlib.util, importlib.metadata as md_, sys\n"
    "\n"
    "reasons = []\n"
    "if importlib.util.find_spec('lm_polygraph') is None:\n"
    "    reasons.append('lm_polygraph not importable in-process '\n"
    "                   '(editable-install .pth is only read at interpreter startup)')\n"
    "if 'torch' in sys.modules:\n"
    "    import torch\n"
    "    try:\n"
    "        on_disk = md_.version('torch')\n"
    "        if torch.__version__.split('+')[0] != on_disk.split('+')[0]:\n"
    "            reasons.append(f'stale torch: in-process {torch.__version__} '\n"
    "                           f'vs installed {on_disk}')\n"
    "    except Exception as e:\n"
    "        print('could not compare torch versions:', e)\n"
    "\n"
    "if reasons:\n"
    "    print('=' * 68)\n"
    "    print('RESTARTING THE KERNEL because:')\n"
    "    for r in reasons:\n"
    "        print('  -', r)\n"
    "    print()\n"
    "    print('  >>> Colab will report the session crashed. THAT IS EXPECTED. <<<')\n"
    "    print('  >>> Then choose  Runtime -> Run all  again.               <<<')\n"
    "    print('  >>> Cells 1-5 are idempotent and will no-op.              <<<')\n"
    "    print('=' * 68)\n"
    "    import IPython\n"
    "    IPython.Application.instance().kernel.do_shutdown(True)\n"
    "else:\n"
    "    print('in-process state is consistent with what is installed -- no restart needed')"
)

# 7 --------------------------------------------------------------------------
md(
    "## 7. Post-restart environment assertions\n"
    "\n"
    "torch may have changed version during the install, so the CUDA binding is\n"
    "re-confirmed here rather than inherited from cell 1. Fails loudly if the\n"
    "in-process torch is stale, CUDA is unavailable, or the device is not a T4."
)
code(
    "import importlib.metadata as md_\n"
    "import torch\n"
    "\n"
    "on_disk = md_.version('torch')\n"
    "print('torch in-process :', torch.__version__)\n"
    "print('torch on disk    :', on_disk)\n"
    "assert torch.__version__.split('+')[0] == on_disk.split('+')[0], (\n"
    "    'STALE TORCH: the kernel holds a different torch than is installed. '\n"
    "    'Re-run cell 6 to restart.')\n"
    "\n"
    "from packaging.version import Version\n"
    "assert Version(torch.__version__.split('+')[0]) >= Version('2.6.0'), (\n"
    "    f'torch {torch.__version__} is below lm-polygraph\\'s required >=2.6.0')\n"
    "\n"
    "assert torch.cuda.is_available(), 'CUDA not available after restart'\n"
    "dev = torch.cuda.get_device_name(0)\n"
    "free, total = torch.cuda.mem_get_info()\n"
    "print('device           :', dev)\n"
    "print(f'VRAM             : {free/1e9:.2f} GB free / {total/1e9:.2f} GB')\n"
    "if 'T4' not in dev:\n"
    "    print(f'WARNING: expected a T4, got {dev!r}. Results stay valid; timings differ.')\n"
    "assert total / 1e9 > 14, f'unexpectedly small GPU ({total/1e9:.1f} GB)'\n"
    "print('\\nenvironment OK')"
)

# 8 --------------------------------------------------------------------------
md(
    "## 8. Fail-fast import check  *(now testing the real post-install state)*\n"
    "\n"
    "Five seconds here beats discovering a broken import inside a subprocess after the\n"
    "3B model has loaded. Checks both in-process and in a **fresh subprocess** — the\n"
    "path `polygraph_eval` actually takes, and the one that matters for the ladder's\n"
    "dotted-path estimators."
)
code(
    "import subprocess, sys, os\n"
    "\n"
    "from lm_polygraph.estimators import SpilledEnergy\n"
    "from lm_polygraph.stat_calculators import EnergyCalculator\n"
    "from harness.pooled_baseline import PooledBaseline\n"
    "print('in-process OK:', str(SpilledEnergy(variant='spilled', pooling='max')),\n"
    "      '|', str(PooledBaseline(score='log_likelihood', pooling='max')))\n"
    "\n"
    "r = subprocess.run([sys.executable, '-c',\n"
    "                    'from lm_polygraph.utils.factory_estimator import FactoryEstimator;'\n"
    "                    'f=FactoryEstimator();'\n"
    "                    'print(\"subprocess OK:\", f(\"harness.pooled_baseline\",'\n"
    "                    '{\"score\":\"log_likelihood\",\"pooling\":\"max\"}),'\n"
    "                    'f(\"SpilledEnergy\",{\"variant\":\"spilled\",\"pooling\":\"max\"}))'],\n"
    "                   capture_output=True, text=True, env=dict(os.environ))\n"
    "print(r.stdout.strip() or r.stderr.strip()[-2000:])\n"
    "assert r.returncode == 0, ('subprocess cannot import -- check PYTHONPATH (cell 4) '\n"
    "                          'before running anything expensive')"
)

# 9 --------------------------------------------------------------------------
md(
    "## 9. Model provenance\n"
    "\n"
    "Printed before anything loads the model. `--expect` makes a silent model swap a\n"
    "hard failure rather than a footnote. Confirm `model.path`, `dtype` and `device`\n"
    "against the report."
)
code(f"!python harness/provenance.py --config {CFG_BASE} --expect {MODEL}")

# 10 --------------------------------------------------------------------------
md("## 10. Unit tests (CPU, seconds)\n\nCheap proof the install is sane before any long run.")
code("!python -m pytest test/test_spilled_energy.py -q")

# 11 --------------------------------------------------------------------------
md(
    "## 11. Dev run, n=150 — the gates fire here\n"
    "\n"
    "This one short run does four jobs:\n"
    "\n"
    "1. **Accuracy gate** — hard-fails outside 10–90% exact match. Outside that band\n"
    "   PRR is noise and every downstream number is meaningless. The usual cause is\n"
    "   prompt format.\n"
    "2. **Sign check** — a wrong sign shows up as a large *negative* normalized PRR\n"
    "   (~-0.7), which reads as a broken method rather than an inverted score.\n"
    "3. **Answer-span validation** (cell 12).\n"
    "4. **Runtime measurement** (cell 13).\n"
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

# 12 --------------------------------------------------------------------------
md(
    "## 12. Validate the answer-span assumption\n"
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

# 13 --------------------------------------------------------------------------
md(
    "## 13. How long will the real runs take?\n"
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

# 14 --------------------------------------------------------------------------
md(
    "## 14. Run A — primary baseline table, n=1000\n"
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

# 15 --------------------------------------------------------------------------
md(
    "## 15. Run B — ablation ladder, n=1000\n"
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

# 16 --------------------------------------------------------------------------
md(
    "## 16. Are the two tables comparable? — hard gate\n"
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

# 17 --------------------------------------------------------------------------
md(
    "## 17. Run C — attention baselines, batch_size=1, n=300\n"
    "\n"
    "`RAUQ` x2, `CSL` and `AttentionScore` need `output_attentions=True`, which\n"
    "disables transformers' left-padding NaN guard. At `batch_size=1` there is no\n"
    "padding at all, so no attention row is ever fully masked and the fp16 overflow\n"
    "cannot occur — the cost of bs=1 is paid by these four methods rather than by\n"
    "the whole experiment.\n"
    "\n"
    "This config also uses **eager** attention, overriding the model group's sdpa.\n"
    "n=300 because bs=1 is roughly 2x slower per sample."
)
code(
    "cmd = ('python harness/run_baselines.py'\n"
    f"       ' --config {CFG_ATTN}'\n"
    "       f\" --save-dir '{DRIVE}/C_attention_bs1_n300'\"\n"
    "       ' --n-boot 1000'\n"
    f"       ' --expect-model {MODEL}')\n"
    "print(cmd)\n"
    "!{cmd}"
)

# 18 --------------------------------------------------------------------------
md(
    "## 18. Does batch size change the generations? — hard gate\n"
    "\n"
    "With the fix in place `batch_size` must not affect generation at all, so Run C\n"
    "(bs=1, n=300) and Run A (bs=4, n=1000) must agree on their shared 300 samples.\n"
    "This is the property `test_batched_generation_matches_individual` asserts on a\n"
    "stub; here it is verified by hash on the real data.\n"
    "\n"
    "`--allow-prefix` is sound because `Dataset.subsample` uses `np.random.choice`\n"
    "under a fixed seed, which is prefix-stable — the n=300 subsample is exactly the\n"
    "first 300 of the n=1000 one (asserted by `test_subsample_is_prefix_stable`).\n"
    "\n"
    "If this fails, the bs=1 table must NOT be placed beside the primary table."
)
code(
    "cmd = ('python harness/check_run_consistency.py'\n"
    "       f\" --a '{DRIVE}/A_baselines_n1000'\"\n"
    "       f\" --b '{DRIVE}/C_attention_bs1_n300'\"\n"
    "       ' --label-a bs4_primary --label-b bs1_attention --allow-prefix')\n"
    "!{cmd}"
)

# 19 --------------------------------------------------------------------------
md(
    "## 19. The reported tables\n"
    "\n"
    "Primary metric is **normalized PRR@0.5** with bootstrap CIs. Everything on Drive,\n"
    "so tables can be rebuilt offline on CPU with\n"
    "`python harness/run_baselines.py --skip-run --save-dir <dir>`."
)
code(
    "from IPython.display import Markdown, display\n"
    "for tag, label in [('A_baselines_n1000', 'PRIMARY BASELINE TABLE'),\n"
    "                   ('B_ladder_n1000',   'ABLATION LADDER'),\n"
    "                   ('C_attention_bs1_n300',\n"
    "                    'SECONDARY: attention baselines (bs=1, n=300)')]:\n"
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

"""Run a command from a notebook cell and RAISE if it fails.

Colab's ``!command`` never raises: a nonzero exit prints and the notebook happily
continues to the next cell. Every gate in this project was therefore advisory
rather than enforcing -- ``Run all`` would sail straight past a failed accuracy
check, a failed comparability check or a failed energy check and produce tables
regardless. That was discovered when cell 21 invoked
``check_energy_identity.py`` with a wrong flag: both calls printed a usage error
and the run continued to the tables.

``gate()`` runs the command as a subprocess and raises on a nonzero return code,
so ``Run all`` stops where it should. ``required=False`` is for genuinely
informational steps (a runtime projection, a report render) which should not
abort the session.

Exit-code convention used by the harness scripts:
    0  pass
    1  fail
    2  inconclusive -- nothing could actually be checked, which is NOT a pass
"""

import shlex
import subprocess
import sys

MEANING = {0: "pass", 1: "FAIL", 2: "INCONCLUSIVE (nothing was checked)"}


def gate(cmd, required: bool = True, label: str = None):
    """Run ``cmd``; raise RuntimeError on failure when ``required``.

    Parameters:
        cmd: command string, or a list of arguments.
        required: if True (default), a nonzero exit raises.
        label: short name for the message; defaults to the script name.
    """
    if isinstance(cmd, str):
        args, shell = cmd, True
        name = label or next(
            (t for t in shlex.split(cmd) if t.endswith(".py")), cmd.split()[0]
        )
    else:
        args, shell = [str(c) for c in cmd], False
        name = label or next((c for c in args if str(c).endswith(".py")), args[0])

    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(args)}\n")
    rc = subprocess.run(args, shell=shell).returncode

    verdict = MEANING.get(rc, f"exit {rc}")
    print(f"\n[gate] {name}: {verdict}")

    if rc != 0:
        if required:
            raise RuntimeError(
                f"GATE FAILED -- {name} returned {rc} ({verdict}).\n"
                "The notebook is stopping here on purpose. Do not read any table\n"
                "produced after a failed gate: fix the cause and re-run."
            )
        print(f"[gate] {name} is informational; continuing.")
    return rc


def main():
    sys.exit(gate(sys.argv[1:]) if len(sys.argv) > 1 else 0)


if __name__ == "__main__":
    main()

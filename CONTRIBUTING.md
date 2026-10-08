# Contributing

This is research code accompanying a paper. Bug fixes, documentation and clean extensions are welcome.

## Reporting a bug

Please include the operating system, the Python version, the output of
`python reproduction/tools/record_provenance.py` (which captures the package versions, the machine, and the commit),
the exact command, and the full traceback.

For a numerical bug, give the network name and the dispersion, and say whether it reproduces on
Sioux Falls or on the five-node witness — both run in seconds and are far easier to reason about than a
city network.

## Before opening a pull request

```sh
python tests/run_tests.py          # or: python -m pytest tests
ruff check src reproduction tests
```

## Changing the solver

Any change to the loading, the support rule or the equilibrium iteration must leave the reported numbers
alone unless it is meant to change them. The check is direct: solve Sioux Falls, Eastern Massachusetts,
Braess and Anaheim at two dispersions before and after, and compare the flow arrays. When the change is a
refactor they should agree to the last bit, not approximately.

If a number is meant to move, say which and why in the pull request, and re-run the exhibits that read
it. `python reproduction/tools/validate_results.py` will refuse to mix rows written by different solver vintages,
which is what stops a partially re-run result file from being read as a whole one.

## Adding an experiment

Put it in `reproduction/exhibits/`, build it on `reproduction.exhibits.records.Exhibit` so it inherits the argument
parsing, the result sink and the provenance stamping, and add a row to the table in
`reproduction/README.md` saying which exhibit it produces. Keep the sink one JSON object per line, appended
and flushed as the run proceeds, so an interrupted run keeps what it finished.

## Style

Python 3.9 or later, `from __future__ import annotations`, type hints on the public surface, PEP 257
docstrings. Comments explain why the code is shaped the way it is; they do not narrate what a line plainly
does.

## Licence

Contributions are licensed under the project's [MIT licence](LICENSE).

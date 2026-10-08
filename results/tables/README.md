# Generated tables

`generated_tables.tex` is the captured output of

    python reproduction/tools/build_tables.py

which reads `results/reproduce/` and `results/held_out/` and emits the LaTeX table bodies and the
numeric macros the paper declares. It is deposited here so the generated tables are available without
running anything; re-running the command above should reproduce this file byte for byte from the same
records.

The generator emits body rows and macro definitions only. Preambles, column rules and captions live in
the paper source.

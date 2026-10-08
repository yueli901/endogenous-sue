# Documentation

How to install and run anything is in [`reproduction/README.md`](../reproduction/README.md).
These documents answer what the code means rather than how to invoke it.

| document | what it answers |
|---|---|
| [data_sources.md](data_sources.md) | where the benchmark networks come from, at which commit, and what is not redistributed |
| [result_schema.md](result_schema.md) | what every field of a result record means |
| [naming.md](naming.md) | ES-SUE, `endogenous-sue`, `endogenous_sue`: which name belongs where |
| [known_limitations.md](known_limitations.md) | what the method does not do, and which diagnostics are unavailable on which path |
| [face_contract.md](face_contract.md) | the contract a solved tied face satisfies, clauses C1–C8, and what reopening cannot repair |
| [held_out_protocol.md](held_out_protocol.md) | the prospective evaluation, written and frozen before that run |

Two of these are worth reading before any number is quoted. `known_limitations.md` explains why a small
residual is not a certificate. `held_out_protocol.md` fixes, in advance, what each outcome of the
prospective run licenses, so the reading was not chosen to suit the number.

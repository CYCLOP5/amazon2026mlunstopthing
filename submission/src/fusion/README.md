# graph and hybrid fusion

this source tree contains our graph sibling refinement, bounded hybrid, residual fusion and collective-graph work
`upstream_neural/` retains the earlier neural runtime required by these stages
the python environment is pinned by `.python-version`, `pyproject.toml` and `uv.lock`

see the release [pipeline guide](../../docs/pipeline.md) and [model inventory](../../docs/models.md)
the final release uses the collective-graph model together with the frozen policy in `configs/release.json` at the submission-project root

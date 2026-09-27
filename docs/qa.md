# final reviewer guide

the detailed, package-aligned q&a is maintained at [submission/docs/qa.md](../submission/docs/qa.md)
it maps reviewer questions to the exact implementation, recorded configurations and result scope

## reading order

1. [q&a and code map](../submission/docs/qa.md)
2. [full architecture](../submission/docs/arch.md)
3. [azure compute and execution](../submission/docs/compute.md)
4. [training and reconstruction pipeline](../submission/docs/pipeline.md)
5. [results and submission lessons](../submission/docs/results.md)
6. [exact release replay](../submission/docs/reproduce.md)
7. [models and licenses](../submission/docs/models.md)

## local custody

| path | retained material |
| --- | --- |
| `artifacts/final-packages/` | verified copies of the final release zips |
| `artifacts/release-inputs/` | exact matching tsvs and the shared candidate tsv |
| `artifacts/package-assets/` | recorded score inputs for deterministic replay |
| `artifacts/downloads-backup/` | original supplied source, outputs and download snapshots |
| `student_resource/dataset/`, `cache/data/` | original and prepared challenge records |

large local artifacts are intentionally outside git
the source branch and packages contain the code, configuration, dependency and verification records needed to interpret them
see [source-custody evidence](../reports/final-source-custody.json)

the root `src/` tree and historical component notes preserve development history
the integrated `submission/` project is the starting point for the actual final pipeline

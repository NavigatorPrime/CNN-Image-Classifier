# Dataset Split Leakage Audit

## Scope

This Stage 1 correction audited `dataset_64x64.zip` and
`split_manifest.csv` using both SHA-256 of the stored file bytes and SHA-256 of
the decoded RGB pixel payload (including mode and dimensions). Filenames alone
were not used to infer identity.

The UCI description states that the released images were cropped to regions of
interest, but neither the package nor its published metadata provides a
documented parent-image/source identifier or filename-key specification.
Similar filename stems were therefore not treated as proof of a shared source;
doing so would invent relationships not established by the supplied metadata.

## Before correction

| Split | Images |
|---|---:|
| Train | 6,377 |
| Validation | 1,364 |
| Test | 1,373 |
| **Total** | **9,114** |

There were 59 identical-content groups overall. Of those, 31 crossed at least
one split boundary:

- Train–validation: 13 duplicate pairs/groups
- Train–test: 12 duplicate pairs/groups
- Validation–test: 6 duplicate pairs/groups
- Total cross-split duplicate groups: 31

## Corrected deterministic split

Every decoded-pixel duplicate group is now assigned as an indivisible unit.
Assignment is class-aware, uses seed 42, and targets 70% train, 15% validation,
and 15% test as closely as the class totals permit.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| H1 | 3,083 | 661 | 660 | 4,404 |
| H2 | 1,634 | 350 | 350 | 2,334 |
| H3 | 573 | 123 | 123 | 819 |
| H5 | 573 | 123 | 122 | 818 |
| H6 | 517 | 111 | 111 | 739 |
| **Total** | **6,380** | **1,368** | **1,366** | **9,114** |

The resulting percentages are 70.002%, 15.010%, and 14.988%. All 9,114
images are 64×64 RGB, all package members match the manifest, filenames are
globally unique, and all 59 raw-byte and decoded-pixel duplicate groups remain
inside exactly one split.

**CROSS-SPLIT DUPLICATES = 0**

## Result status

The checked-in training outputs were produced on the superseded split. They
remain available for traceability but are historical and must not be presented
as leakage-free final results. All models, optimizer comparisons, final tables,
figures, and reports must be regenerated after retraining on this corrected
split.

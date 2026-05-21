---
name: Reference — IBAMR as the production reference for AMR + Taira–Colonius IBPM
description: The closest production implementation of the scalable AMR + Taira-Colonius direction; consult their design decisions when in doubt.
type: reference
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**IBAMR** (Bhalla, Griffith et al.) is the canonical AMR + immersed-boundary
projection method codebase.  It implements the production-scale version
of the algorithm this repo is now prototyping, on a different software
stack.

- Stack: SAMRAI (AMR) + PETSc (Krylov + matrix-free operators) +
  libMesh (FE-IB for elastic bodies).
- Key paper: **Bhalla, Bale, Griffith, Patankar, *"A unified mathematical
  framework and an adaptive numerical method for fluid-structure
  interaction with rigid, deforming, and elastic bodies,"* JCP 2013**.
  Describes the saddle-point operator + preconditioner structure with
  enough rigour to map directly onto the Trilinos-side implementation
  planned here.
- Repo: github.com/IBAMR/IBAMR.

**When to consult**:

- Designing the matrix-free `Tpetra::Operator` for `S = Q^T B^N Q` —
  IBAMR's `INSStaggeredStokesOperator` is the analogue.
- Choosing the preconditioner for the modified Poisson — IBAMR uses a
  FAC (Fast Adaptive Composite) multigrid Poisson solver as the inner
  preconditioner, very close to what we'd do with AMReX `MLMG` wrapped
  as a `Tpetra::Operator`.
- Reflux / sync solve at C/F interfaces in the projection step — IBAMR
  does an explicit sync after each level projection.
- Treating moving bodies (Lagrangian point regularised-delta
  reassembly each step).

**How to apply**: prefer their algorithmic choices when there's a
question (their code has been validated against many test problems).
Their SAMRAI patterns translate fairly directly to AMReX
(BoxArray ≈ PatchHierarchy, FillPatch ≈ RefineSchedule, etc.).

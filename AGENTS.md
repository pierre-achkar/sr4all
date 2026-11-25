# Repository Instructions

## Scope
These instructions apply to every directory, script, pipeline, and data artifact in this repository.

## Working Principles
- Keep changes narrow, reproducible, and traceable to the stage that owns the behavior.
- Preserve stable record identifiers and provenance across all transformations.
- Prefer resumable, batch-oriented workflows for long-running operations.
- Keep generation, validation, repair, retrieval, and dataset finalization as distinct stages.
- Do not make unrelated refactors while addressing a focused task.

## GitHub Planning and Issue Tracking
- At the start of every session and before starting a new repository task, always review the [SR4ALL project board](https://github.com/users/pierre-achkar/projects/13), [open milestones](https://github.com/pierre-achkar/sr4all/milestones?state=open), and [open issues](https://github.com/pierre-achkar/sr4all/issues?q=is%3Aissue%20is%3Aopen).
- Read relevant issue descriptions, recent comments, acceptance criteria, current board status, milestone membership, and blocked-by dependencies before planning or editing. Use live GitHub state rather than relying only on conversation history; never assume pending human reviews or approvals are complete.
- Keep work aligned with the user's current request and the relevant milestone and issues. Check for related issues before proposing new ones, avoid duplicates, and do not start unrelated backlog work merely because it appears on the board.
- Use authenticated GitHub tooling scoped to this repository and its board. If access fails, explicitly report which checks could not be completed; do not interpret unavailable information as an empty backlog or satisfied prerequisites.

## Research Priorities and Evaluation Planning
- Read relevant [meeting records](meetings/) when interpreting research scope. Distinguish agreed decisions, exploratory proposals, and unresolved questions; do not turn a discussion into an implementation commitment.
- Prioritize the corpus journal track: human adjudication, targeted extraction improvements using resolved labels, and submission preparation with supported validation claims and explicit limitations.
- Keep conference scoping, structure-aware representation, DSPy, and AI-judge experiments separate from the journal's critical path. Do not add an unapproved benchmark, representation redesign, or conference experiment as a submission prerequisite.
- If live issue requirements differ from recorded meeting commitments, surface the discrepancy and obtain scope clarification before expanding work. Do not silently override either source or treat unresolved scope as waived acceptance criteria.
- Before using annotated examples to improve extraction, document their development/reference/evaluation role and which examples remain eligible for evaluation. Do not claim held-out performance on examples used for development.
- Distinguish candidate-value correctness and corpus field completeness from reference-supported recall. Missing extractions or omitted assessments do not establish source-field absence; keep unsupported estimates and unresolved coverage explicit.

## Repository Organization
- `src/` contains code for building, validating, and analyzing the corpus.
- `src/extraction_v2/` contains the methodology extraction, alignment, fact-checking, and repair workflow.
- `src/retrieval/` contains OpenAlex retrieval, filtering, PDF acquisition, and metadata preparation.
- `src/oax/` and `src/norm_queries/` contain OpenAlex query normalization and related retrieval work.
- `src/final_ds/` owns final dataset merging, flattening, and completeness filtering.
- `src/validation/` is flat and Python-only: no nested directories, Markdown, HTML, or generated artifacts.
- `docs/validation/` contains validation documentation and annotator guidelines.
- `tests/` contains automated tests; `tests/fixtures/validation/` contains the synthetic browser example and test inputs.
- `paper/` contains the LaTeX manuscript, bibliography, figures, and manuscript tables.
- `jobs/` contains Slurm entry points for compute-intensive pipeline stages.
- `data/` contains pipeline inputs and outputs. Treat generated datasets as artifacts, not source code.
- `data/validation/annotation/` contains original tasks, the sampling manifest, original packages, and independent annotation exports.
- `data/validation/adjudication/` contains clarification preparation, distributed packages, older builds under `previous_packages/`, and returned clarification reviews under `exports/`.
- `data/validation/analysis/` contains report snapshots and comment-coding inputs; `data/validation/demos/` contains the five-document demo and its package.
- `data/validation/templates/` contains the three version-controlled HTML templates. Preserve their narrow Git-ignore exceptions; other validation data and generated outputs remain ignored.
- `logs/` contains current pipeline and Slurm logs.
- `archive/` contains historical artifacts, including `data_old/`, `logs_old/`, and `validation_toolkit/`. Keep these separate from current inputs and outputs.

## Data Handling
- Do not open large files directly in the terminal, particularly JSON, JSONL, CSV, PDFs, model outputs, or files under `data/` unless their size is known to be small.
- Check file sizes with metadata commands such as `stat` before inspecting artifacts. Read large artifacts only inside a Slurm allocation, using bounded, targeted readers such as `head`, `tail`, `rg`, or `jq` filters.
- Never print an entire large artifact to the terminal or load it into memory merely for inspection.
- Preserve append-only and resumable output behavior where established by a pipeline stage.
- Do not overwrite raw inputs or prior run artifacts unless the task explicitly requires it.
- Keep independent annotation exports separate from clarification reviews and adjudicated decisions. Never substitute AI proposals or majority votes for explicit human decisions.
- Reorganize existing packages and snapshots by moving them intact, preserving package IDs, file hashes, judgments, and historical provenance. Retain older builds and use fresh output directories for new builds.

## Running Code
- Execute Python checks, tests, scripts, and inference in the Slurm-managed Apptainer environment, not directly on the login node.
- Check live capacity with `sinfo -a -N -o "%P %N %T %G %C"` before allocating resources.
- Use CPU-only allocations for tests and analysis; request GPUs and pass `--nv` only for workloads that require them. For example:

	```bash
	srun --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
	  apptainer exec --fakeroot --overlay .overlay/ --bind /storage/remote/DevStorage \
	  --env PYTHONDONTWRITEBYTECODE=1 \
	  /opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif \
	  python3 -m pytest tests/test_analyze_validation.py tests/test_adjudication_packages.py tests/test_sample_validation.py
	```

- For interactive GPU work, use:

	```bash
	srun --gres=gpu:A100:1 --pty apptainer shell --fakeroot --overlay .overlay/ --bind /storage/remote/DevStorage --nv /opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif
	```

- Run commands from the repository root after entering the container.
- Use the appropriate script in `jobs/` for production or long-running stages rather than manually reproducing job setup.
- Keep environment paths, mounts, overlays, and model settings consistent with the relevant job script.
- `.overlay/` and `.overlay-vllm/` serve different jobs; v2 fact-checking also uses `.factcheck_venv/`. Do not remove or relocate these resources based solely on which overlay was used most recently.
- Set `PYTHONDONTWRITEBYTECODE=1` during validation checks to avoid creating nested bytecode caches in the Python-only validation source directory.

## Validation and Quality
- Validate the smallest affected stage after code changes, then expand validation only when warranted.
- Enforce schema and contract violations explicitly; do not silently invent missing evidence or metadata.
- Preserve evidence-bearing fields and use existing validation or repair mechanisms rather than ad hoc post-processing.
- Record meaningful failures with enough context to resume or audit the run.

## Change Management
- Follow established file naming and stage sequencing conventions.
- Update documentation when a pipeline contract, execution requirement, or artifact layout changes.
- Avoid adding new abstractions or pipeline stages unless they resolve a clear ownership or reproducibility problem.

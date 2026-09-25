# Monkey-Parma-ReachWalk

Does channel-level LFP power in macaque premotor/motor cortex, during freely moving
behaviour, organise by **action goal** (grasp hook vs. grasp floor) or by
**kinematics/posture** (grasp floor vs. step)? Same recording dataset as Lanzarini et
al., 2025, *Science* 387:214-220 (Bonini lab, University of Parma), re-analysed at the
single-LFP-channel level. See `CLAUDE.md` section 1 for the full hypotheses and
`HANDOFF.md` for the current task list.

## Data

Data live **outside** this repository, on:

```
C:\Users\tommy\OneDrive - Scuola Superiore Sant'Anna\Monkeys Parma
```

Paths are resolved in `src/config.py`; never hardcode a data path elsewhere. See
`CLAUDE.md` section 2 for the directory layout, the session registry
(`session_metadata.csv`) and the event CSV schema.

## Environment

Use the conda env `MonkeyReachWalk`:

```bash
conda activate MonkeyReachWalk
pip install -r requirements.txt
```

## Pipeline

Run in order (entry points in `src/` and `utils/`; notebooks in `notebooks/` are thin
drivers over them):

| Stage | Entry point | Input | Output |
| --- | --- | --- | --- |
| 0. Conversion (MATLAB) | `utils/from_matlab_to_binary_conversion.m` | `raw_matlab/` | `raw_binary/` |
| 1. LFP extraction | `src.preprocessing.extract_and_save_lfp` | `raw_binary/*.bin` | `interim/.../lfp_1000Hz` |
| 2. Artefact marking | `utils.artifact_inspection.inspect_artifacts` | LFP + events | `processed/.../bad_trials_<event>.csv` |
| 3. Time-frequency | `src.preprocessing.extract_multitaper_epochs` | LFP + events + bad trials | `processed/.../epoched_multitaper_<event>_<fs>Hz_<win>ms.npz` |
| 4. Statistics | `src.statistical_analysis.analyze_event_modulation`, `analyze_selectivity` | `.npz` | `processed/.../*_results.csv` |
| 5. Visualisation | `src.plot` | `.npz` | figures in `results/figures/` |

Stage 0 is only needed for a new session (see `CLAUDE.md` section 2). Stage 1 takes
hours per session: never launch a full-pipeline re-run without asking first.

## More

- `CLAUDE.md`: guidance for working in this repository (dataset, pipeline, signal and
  statistical conventions, code style) — read this first.
- `HANDOFF.md`: the task list and communication channel between the planning session
  and implementation work in this repo.

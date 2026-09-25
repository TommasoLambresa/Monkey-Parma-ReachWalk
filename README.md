# Monkey-Parma-ReachWalk

Does channel-level LFP power in macaque premotor/motor cortex, during freely moving
behaviour, organise by **action goal** (grasp hook vs. grasp floor) or by
**kinematics/posture** (grasp floor vs. step)? Same recording dataset as Lanzarini et
al., 2025, *Science* 387:214-220 (Bonini lab, University of Parma), re-analysed at the
single-LFP-channel level. 

## Data

Paths are resolved in `src/config.py`; never hardcode a data path elsewhere.

## Pipeline preprocessing

Run in order (entry points in `src/` and `utils/`; notebooks in `notebooks/` are thin
drivers over them):

| Stage                  | Entry point                                     | Input                     | Output                                                          |
| ---------------------- | ----------------------------------------------- | ------------------------- | --------------------------------------------------------------- |
| 0. Conversion (MATLAB) | `utils/from_matlab_to_binary_conversion.m`    | `raw_matlab/`           | `raw_binary/`                                                 |
| 1. LFP extraction      | `src.preprocessing.extract_and_save_lfp`      | `raw_binary/*.bin`      | `interim/.../lfp_1000Hz`                                      |
| 2. Artefact marking    | `utils.artifact_inspection.inspect_artifacts` | LFP + events              | `processed/.../bad_trials_<event>.csv`                        |
| 3. Time-frequency      | `src.preprocessing.extract_multitaper_epochs` | LFP + events + bad trials | `processed/.../epoched_multitaper_<event>_<fs>Hz_<win>ms.npz` |

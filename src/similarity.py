import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rsatoolbox.data import Dataset
from rsatoolbox.data.noise import cov_from_residuals
from rsatoolbox.rdm import calc_rdm
from statsmodels.stats.multitest import multipletests

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SESSION_METADATA, EVENT_SUFFIXES, FREQ_BANDS,
    EPOCH_T_PRE, SIMILARITY_PARAMS, MULTITAPER_PARAMS,
)
from src.io import load_multitaper_epochs

# Integer codes of the three classes, used for the labels throughout this module.
HOOK, FLOOR, STEP = 0, 1, 2
CLASS_NAMES = {HOOK: 'hook', FLOOR: 'floor', STEP: 'first_step'}
# Pair statistics: name -> (class, class).
PAIR_STATISTICS = {'d2_SH': (STEP, HOOK), 'd2_SF': (STEP, FLOOR), 'd2_HF': (HOOK, FLOOR)}
# Difference statistics: name -> (minuend, subtrahend) among the pair statistics.
DELTA_STATISTICS = {'delta_SH_minus_SF': ('d2_SH', 'd2_SF'), 'delta_SF_minus_HF': ('d2_SF', 'd2_HF')}
STATISTICS = list(PAIR_STATISTICS) + list(DELTA_STATISTICS)


def _cross_validated_d2(points: np.ndarray, labels: np.ndarray, session_ids: np.ndarray,
                        rng: np.random.Generator, n_repetitions: int) -> tuple[dict, float]:
    """
    Cross-validated squared Mahalanobis distance (crossnobis, computed by rsatoolbox)
    between the centroids of the classes present in 'labels', for the pairs of
    PAIR_STATISTICS.

    Plain words: the trials of a class form a cloud of points (one axis per channel). The
    noise cloud is how trials scatter around their own class centroid; its inverse (the
    precision matrix) is the ruler of the Mahalanobis distance. Each class is split into two
    independent halves of trials, and the difference between two class centroids in half 1
    is multiplied channel by channel, through the ruler, with the same difference in half 2.
    If the two classes have the same map, the two differences are unrelated and the product
    averages to zero, whatever the trial counts.

    Parameters
    ----------
    points : (n_trials, n_channels) array
        Window-mean band power of every trial, in session-z units.
    labels : (n_trials,) int array
        Class code of every trial (HOOK, FLOOR, STEP). A class absent from 'labels' gives
        NaN for the pairs that contain it.
    session_ids : (n_trials,) int array
        Session of every trial. Halves are drawn within each class x session group, so both
        halves have the same session mix.
    rng : numpy Generator
        Source of the random half-splits.
    n_repetitions : int
        Number of random splits; the result is their mean.

    Returns
    -------
    d2 : dict
        PAIR_STATISTICS name -> squared distance per channel (dimensionless: squared noise
        standard deviations). Expected value is the true squared distance, 0 if the two
        classes have the same map; it can be negative by noise.
    shrinkage : float
        Weight in [0, 1] given to the diagonal in the shrunk noise covariance. Near 1 the
        distance reduces to a Euclidean distance on channels scaled by their own noise; near
        0 the full cloud shape is used.
    """
    n_trials, n_channels = points.shape
    present = [c for c in (HOOK, FLOOR, STEP) if np.any(labels == c)]

    # Noise cloud: residuals around the centroid of each trial's class, n_trials - n_classes
    # degrees of freedom. 'shrinkage_diag' (Schaefer & Strimmer 2005) keeps the channel
    # variances and shrinks the off-diagonal covariances by a common factor (1 - shrinkage).
    centroids = np.zeros((3, n_channels))
    for c in present:
        centroids[c] = points[labels == c].mean(axis=0)
    residuals = points - centroids[labels]
    dof = n_trials - len(present)
    shrunk_cov = cov_from_residuals(residuals, dof=dof, method='shrinkage_diag')
    full_cov = cov_from_residuals(residuals, dof=dof, method='full')
    # The shrinkage weight is not returned by rsatoolbox; it is the factor that scales the
    # largest off-diagonal covariance of the sample covariance down to the shrunk one.
    off_diagonal = full_cov - np.diag(np.diag(full_cov))
    largest = np.unravel_index(np.argmax(np.abs(off_diagonal)), off_diagonal.shape)
    shrinkage = 1.0 - shrunk_cov[largest] / full_cov[largest]
    precision = np.linalg.inv(shrunk_cov)

    # Stratified random halves, one per repetition: within each class x session group a
    # random permutation decides which trials go to the second half; an odd trial goes to a
    # random half.
    second_half = np.zeros((n_repetitions, n_trials), dtype=bool)
    group_keys = labels * (session_ids.max() + 1) + session_ids
    for key in np.unique(group_keys):
        members = np.flatnonzero(group_keys == key)
        n_members = len(members)
        ranks = rng.random((n_repetitions, n_members)).argsort(axis=1).argsort(axis=1)
        first_half_size = n_members // 2 + (n_members % 2) * rng.integers(0, 2, n_repetitions)
        second_half[:, members] = ranks >= first_half_size[:, None]

    # rsatoolbox returns the matrix of distances between the classes present, sorted by code.
    rdm_index = {c: i for i, c in enumerate(present)}
    d2_per_repetition = {name: [] for name in PAIR_STATISTICS}
    for repetition in range(n_repetitions):
        halves = second_half[repetition]
        if any(not (np.any(halves & (labels == c)) and np.any(~halves & (labels == c))) for c in present):
            # A class with an empty half cannot be cross-validated: the repetition is skipped.
            continue
        dataset = Dataset(
            points,
            obs_descriptors={'class': labels, 'half': halves.astype(int)},
        )
        rdm = calc_rdm(
            dataset, method='crossnobis', descriptor='class', noise=precision, cv_descriptor='half',
        ).get_matrices()[0]
        for name, (class_a, class_b) in PAIR_STATISTICS.items():
            if class_a in rdm_index and class_b in rdm_index:
                d2_per_repetition[name].append(rdm[rdm_index[class_a], rdm_index[class_b]])

    d2 = {
        name: float(np.mean(values)) if values else np.nan
        for name, values in d2_per_repetition.items()
    }
    return d2, float(shrinkage)


def compute_class_distances(subject: str) -> dict:
    """
    Measures how far apart the class maps of first steps, hook grasps and floor grasps are,
    per frequency band, with the cross-validated Mahalanobis distance, and tests the
    distances by label permutation. Sessions of the subject are pooled. Results are also
    saved to PROCESSED_DATA_DIR/<subject>/class_distances.csv.

    Geometry, in plain words. For one band, average the band power of every channel over
    [-EPOCH_T_PRE, 0): a trial becomes one point in a space with one axis per channel. The
    trials of a class form a cloud; its centre (centroid) is the class map. The Mahalanobis
    distance between two centroids is weighted by the scatter of trials around their own
    class centroid: a shift along a direction where trials of the same class already
    scatter a lot counts little, a shift along a direction where they barely scatter counts
    a lot. Predictions (S = first step, H = hook grasp, F = floor grasp):
    H_goal if Floor is closer to Hook than to Steps (d2_SF > d2_HF), H_kin if Floor is
    closer to Steps than to Hook (d2_SF < d2_HF), non-selective if no pair is separated.

    Trial selection: contralateral hand only. Hook and Floor are the contralateral grasps of
    action_class 'grasp_hook' / 'grasp_floor'. Step is the first step of each walk (earliest
    StepTime of its WalkNumber in the full event table, whatever the hand), kept only if it
    is contralateral; a walk whose first contact is ipsilateral contributes nothing, and if
    the true first step was dropped before the multitaper file was written the walk
    contributes nothing. No caps, no subsampling. All sessions of the subject in
    SESSION_METADATA are pooled; subjects are never pooled. A class with fewer than
    SIMILARITY_PARAMS['min_trials_per_class'] pooled trials is not testable: every
    statistic involving it is NaN and a message is printed (a class below the threshold is
    also left out of the noise cloud).

    Noise cloud. Subtract from each trial the centroid of its class (centroids from the
    pooled trials of the subject, n_trials - n_classes degrees of freedom) and estimate the
    covariance of the residuals with the 'shrinkage_diag' estimator of rsatoolbox (Schaefer
    & Strimmer 2005): with fewer trials than channels the shape of the cloud is known only
    along a few directions, and shrinkage fills in the rest by keeping each channel's own
    variance and pulling the covariances between channels toward zero. The inverse of the
    shrunk covariance (the precision matrix) is the ruler of the Mahalanobis distance; no
    further weighting is applied.

    Statistic: crossnobis, computed by rsatoolbox.rdm.calc_rdm. d2_AB = (1/C) * (m_A1 - m_B1)
    P (m_A2 - m_B2), with P the precision matrix, m_X1, m_X2 the centroids of class X over two
    random halves of its trials (stratified by session) and C the number of channels. The
    value is averaged over SIMILARITY_PARAMS['n_repetitions'] random splits. Expected value: the true squared distance per channel; 0 if the two
    classes have the same map. Unlike the non-cross-validated ||m_A - m_B||^2 / C, whose
    expected value grows by noise_A / n_A + noise_B / n_B, it is unbiased whatever the trial
    counts; its variance still grows as counts shrink, so counts are reported with every
    result.

    Inference (null hypothesis, no distributional assumption): labels are permuted within
    session, so every permuted class keeps the session mix of the real one. The noise cloud
    and its shrinkage are recomputed for every permutation, and each null uses the same
    number of repetitions as the observed value. p = (1 + count) / (1 + n_permutations).
        d2_SH, d2_SF, d2_HF : H0 the two classes have the same map (labels exchangeable);
            one-sided, null >= observed. This is the explicit test of the non-selective
            outcome.
        delta_SH_minus_SF = d2_SH - d2_SF : positive means Floor is closer to Steps than Hook
            is. H0 Hook and Floor exchangeable; two-sided, |null| >= |observed|.
        delta_SF_minus_HF = d2_SF - d2_HF : positive supports H_goal, negative H_kin. H0 Hook
            and Steps exchangeable; two-sided.
    Holm correction across the five bands, separately for each statistic (raw and corrected
    p reported). Beta and gamma are the primary bands; delta, theta and alpha are not
    separable at the 2 Hz Rayleigh resolution (3 Hz half-bandwidth) of the 0.5 s window.

    Caveats:
        - Pooling assumes the same channel records the same activity across the sessions of
          a subject (chronic arrays, session-level normalisation). It is not tested here.
        - Uneven class composition across sessions (e.g. Router, contralateral, before
          artefact rejection: hook 19 vs 5, floor 5 vs 10) lets a day difference leak into a
          class difference. The within-session permutation puts the same leak into the null
          and the pooled residuals down-weight it, but neither removes it entirely.
        - The cloud shape is estimated in 128 dimensions from a few tens of trials; the
          shrinkage weight says how much of it is used. The shrinkage estimator assumes
          independent trials. The noise cloud is estimated on the same trials the distance is
          computed on, so with fewer trials than channels the distance has a small positive
          bias (simulated identical classes, 71 trials, 128 channels: about +0.03 to +0.06);
          the permutation nulls carry the same bias, so the tests are not affected, but the
          absolute d2 values are slightly inflated, by an amount that depends on the counts.
        - All classes are aligned to hand contact, so [-EPOCH_T_PRE, 0) contains the swing of
          the first step or the reach: late preparation plus the first part of the movement.
          For the first step, the early part of the window reflects what the monkey was doing
          before walking, which is not controlled.
        - The 0.5 s multitaper window lets post-contact signal into the last ~275 ms of the
          window.
        - Gamma (40-200 Hz) is broadband and contains line-noise harmonics (no notch filter)
          and possibly impact or muscle artefacts. The noise ruler does not remove an artefact
          specific to one class.
    References: Natraj et al. 2022 (Neuron), Mahalanobis distance between time-averaged
    channel maps (here cross-validated and with one cloud shape shared by all pairs);
    Walther et al. 2016 (NeuroImage), cross-validated Mahalanobis distance; Diedrichsen,
    Provost & Zareamoghaddam 2016 (arXiv:1607.01371), noise covariance from residuals with
    shrinkage toward the diagonal; Schaefer & Strimmer 2005 (Statistical Applications in
    Genetics and Molecular Biology), shrinkage estimator; Nili et al. 2014 (PLoS
    Computational Biology), representational similarity analysis toolbox (rsatoolbox
    implements the crossnobis distance and the covariance estimators).

    Requires normalization == 'session' in every file (a per-trial z-score sets every trial's
    window mean to zero, so all centroids would coincide); the two files of a session must
    share robust, mu and sigma; all sessions must have the same number of channels. Each
    session stays in its own session-z units.

    Parameters
    ----------
    subject : str
        Subject name (e.g. 'Router').

    Returns
    -------
    results : dict
        band name -> dict with:
            'observed' : statistic name -> value (d2 values are dimensionless).
            'null' : statistic name -> (n_permutations,) array.
            'p_raw', 'p_holm' : statistic name -> p-value.
            'lambda' : shrinkage weight of the noise covariance for the observed labels.
            'counts' : DataFrame, trials per class (rows) x session (columns).
            'n_repetitions', 'n_permutations' : int.
        Statistics are 'd2_SH', 'd2_SF', 'd2_HF', 'delta_SH_minus_SF', 'delta_SF_minus_HF'.
    """
    n_repetitions = SIMILARITY_PARAMS['n_repetitions']
    n_permutations = SIMILARITY_PARAMS['n_permutations']
    min_trials = SIMILARITY_PARAMS['min_trials_per_class']
    seed = SIMILARITY_PARAMS['seed']
    n_jobs = SIMILARITY_PARAMS['n_jobs']

    sessions = SESSION_METADATA.index[SESSION_METADATA['Subject'] == subject].tolist()
    if not sessions:
        raise ValueError(f"No session of subject {subject!r} in SESSION_METADATA.")

    # 1. Window-mean band power of the selected trials of every session.
    band_points = {band: [] for band in FREQ_BANDS}
    label_blocks, session_blocks = [], []
    counts = pd.DataFrame(0, index=[CLASS_NAMES[c] for c in (HOOK, FLOOR, STEP)], columns=sessions)
    n_channels_ref = None
    print(f"[{subject}] Trial selection (contralateral hand, first step of each walk):")
    for session_index, session in enumerate(sessions):
        steps = load_multitaper_epochs(subject, session, 'steps')
        grasp = load_multitaper_epochs(subject, session, 'grasp')

        for event_type, data in (('steps', steps), ('grasp', grasp)):
            if str(data['normalization']) != 'session':
                raise ValueError(
                    f"{session} ({event_type}) has normalization={str(data['normalization'])!r}; "
                    f"'session' is required (a per-trial z-score sets every trial's window "
                    f"mean to zero, so all centroids would coincide)."
                )
        if not (bool(steps['robust']) == bool(grasp['robust'])
                and np.array_equal(steps['mu'], grasp['mu'])
                and np.array_equal(steps['sigma'], grasp['sigma'])):
            raise ValueError(
                f"{session}: the steps and grasp files do not share robust/mu/sigma."
            )
        n_channels = steps['mt_tensor'].shape[3]
        if n_channels_ref is None:
            n_channels_ref = n_channels
        elif n_channels != n_channels_ref:
            raise ValueError(
                f"{session} has {n_channels} channels, expected {n_channels_ref} (same as "
                f"the other sessions of {subject})."
            )

        steps_trials = steps['trials']
        keep_steps = (steps_trials['is_first_in_walk'] & steps_trials['is_contralateral']).values
        grasp_trials = grasp['trials']
        keep_hook = (grasp_trials['is_contralateral'] & (grasp_trials['action_class'] == 'grasp_hook')).values
        keep_floor = (grasp_trials['is_contralateral'] & (grasp_trials['action_class'] == 'grasp_floor')).values

        steps_csv = RAW_DATA_DIR / subject / session / "Events" / f"{session}{EVENT_SUFFIXES['steps']}"
        n_walks = int(pd.read_csv(steps_csv)['WalkNumber'].dropna().nunique())
        print(
            f"  {session}: {n_walks} walks, {int(steps_trials['is_first_in_walk'].sum())} "
            f"first steps kept by the multitaper extraction, {int(keep_steps.sum())} "
            f"after the hand filter."
        )

        counts.loc['hook', session] = int(keep_hook.sum())
        counts.loc['floor', session] = int(keep_floor.sum())
        counts.loc['first_step', session] = int(keep_steps.sum())

        for data, selections in ((grasp, [(HOOK, keep_hook), (FLOOR, keep_floor)]),
                                 (steps, [(STEP, keep_steps)])):
            band_masks = {
                band: (data['freqs'] >= low) & (data['freqs'] <= high)
                for band, (low, high) in FREQ_BANDS.items()
            }
            # Window [-EPOCH_T_PRE, 0), with half a sample of tolerance on the float time axis.
            time_tolerance = 0.5 / MULTITAPER_PARAMS['target_fs']
            window_samples = np.flatnonzero(
                (data['times_s'] >= -EPOCH_T_PRE - time_tolerance) & (data['times_s'] < -time_tolerance)
            )
            window = slice(window_samples[0], window_samples[-1] + 1)
            for class_code, keep in selections:
                if not keep.any():
                    continue
                selected = data['mt_tensor'][keep][:, :, window]
                for band, band_mask in band_masks.items():
                    band_points[band].append(
                        selected[:, band_mask].mean(axis=(1, 2), dtype=np.float64)
                    )
                label_blocks.append(np.full(int(keep.sum()), class_code))
                session_blocks.append(np.full(int(keep.sum()), session_index))
            del data['mt_tensor']

    print(counts.to_string())
    labels = np.concatenate(label_blocks)
    session_ids = np.concatenate(session_blocks)

    # 2. Testable classes.
    class_totals = {c: int((labels == c).sum()) for c in (HOOK, FLOOR, STEP)}
    testable = {c for c, n in class_totals.items() if n >= min_trials}
    for c, n in class_totals.items():
        if c not in testable:
            print(
                f"[{subject}] Class '{CLASS_NAMES[c]}' has {n} pooled trials (< {min_trials}): "
                f"not testable; every statistic involving it is NaN."
            )
    keep_trials = np.isin(labels, list(testable))
    labels, session_ids = labels[keep_trials], session_ids[keep_trials]

    pair_ready = {
        name: classes[0] in testable and classes[1] in testable
        for name, classes in PAIR_STATISTICS.items()
    }
    all_ready = len(testable) == 3

    # 3. Observed statistics and permutation nulls, per band.
    # Each scheme swaps the labels of two classes within session. Its permutations feed the
    # pair test of those two classes and one difference test: Hook/Floor exchangeable for
    # delta_SH_minus_SF, Hook/Steps exchangeable for delta_SF_minus_HF.
    schemes = [
        (HOOK, FLOOR, 'd2_HF', 'delta_SH_minus_SF'),
        (HOOK, STEP, 'd2_SH', 'delta_SF_minus_HF'),
        (FLOOR, STEP, 'd2_SF', None),
    ]
    results = {}
    for band_index, band in enumerate(FREQ_BANDS):
        print(f"[{subject}] {band}: observed distance and {n_permutations} permutations")
        points = np.concatenate(band_points[band])[keep_trials]
        rng = np.random.default_rng([seed, band_index, len(schemes)])

        if len(testable) >= 2:
            d2_obs, shrinkage = _cross_validated_d2(points, labels, session_ids, rng, n_repetitions)
        else:
            d2_obs, shrinkage = {name: np.nan for name in PAIR_STATISTICS}, np.nan
        observed = dict(d2_obs)
        for delta_name, (minuend, subtrahend) in DELTA_STATISTICS.items():
            observed[delta_name] = d2_obs[minuend] - d2_obs[subtrahend]
        null = {name: np.full(n_permutations, np.nan) for name in STATISTICS}

        def permuted_d2(seed_sequence, swappable):
            # One permutation: labels shuffled among the 'swappable' trials within session.
            permutation_rng = np.random.default_rng(seed_sequence)
            permuted = labels.copy()
            for session_index in np.unique(session_ids):
                idx = np.flatnonzero(swappable & (session_ids == session_index))
                permuted[idx] = permutation_rng.permutation(labels[idx])
            return _cross_validated_d2(points, permuted, session_ids, permutation_rng, n_repetitions)[0]

        for scheme_index, (class_a, class_b, pair_name, delta_name) in enumerate(schemes):
            if class_a not in testable or class_b not in testable:
                continue
            swappable = np.isin(labels, [class_a, class_b])
            # One independent random stream per permutation, so the result does not depend
            # on how the permutations are spread over the workers.
            seed_sequences = np.random.SeedSequence([seed, band_index, scheme_index]).spawn(n_permutations)
            d2_nulls = Parallel(n_jobs=n_jobs)(
                delayed(permuted_d2)(seed_sequence, swappable) for seed_sequence in seed_sequences
            )
            null[pair_name] = np.array([d2_perm[pair_name] for d2_perm in d2_nulls])
            if delta_name is not None:
                minuend, subtrahend = DELTA_STATISTICS[delta_name]
                null[delta_name] = np.array([d2_perm[minuend] - d2_perm[subtrahend] for d2_perm in d2_nulls])

        p_raw = {}
        for name in STATISTICS:
            ready = pair_ready[name] if name in pair_ready else all_ready
            if not ready:
                p_raw[name] = np.nan
            elif name.startswith('d2'):
                p_raw[name] = (1 + np.sum(null[name] >= observed[name])) / (1 + n_permutations)
            else:
                p_raw[name] = (1 + np.sum(np.abs(null[name]) >= abs(observed[name]))) / (1 + n_permutations)

        results[band] = {
            'observed': observed, 'null': null, 'p_raw': p_raw, 'p_holm': {},
            'lambda': shrinkage, 'counts': counts.copy(),
            'n_repetitions': n_repetitions, 'n_permutations': n_permutations,
        }

    # 4. Holm correction across bands, separately for each statistic.
    bands = list(results)
    for name in STATISTICS:
        raw = np.array([results[b]['p_raw'][name] for b in bands])
        corrected = np.full(len(bands), np.nan)
        valid = ~np.isnan(raw)
        if valid.any():
            corrected[valid] = multipletests(raw[valid], method='holm')[1]
        for b, p in zip(bands, corrected):
            results[b]['p_holm'][name] = float(p)

    # 5. CSV: one row per band x statistic.
    totals = counts.sum(axis=1)
    rows = [
        {
            'subject': subject, 'band': band, 'statistic': name,
            'observed': results[band]['observed'][name],
            'p_raw': results[band]['p_raw'][name],
            'p_holm': results[band]['p_holm'][name],
            'n_hook': int(totals['hook']), 'n_floor': int(totals['floor']),
            'n_first_step': int(totals['first_step']),
            'n_sessions': len(sessions), 'lambda': results[band]['lambda'],
            'n_repetitions': n_repetitions, 'n_permutations': n_permutations,
        }
        for band in bands for name in STATISTICS
    ]
    pd.DataFrame(rows).to_csv(PROCESSED_DATA_DIR / subject / 'class_distances.csv', index=False)
    return results

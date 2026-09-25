import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, ttest_1samp
from tqdm import tqdm
from collections import Counter
import warnings
from src.config import (PROCESSED_DATA_DIR, FREQ_BANDS, MULTITAPER_PARAMS, STATISTICAL_PARAMS, EVENT_SUFFIXES)
from src.io import load_multitaper_epochs

def _get_p_val(res_df: pd.DataFrame, source_name: str) -> float:
    """Helper to extract uncorrected p-value if available, else uncorrected."""
    row = res_df[res_df['Source'] == source_name]
    if row.empty or 'p_unc' not in row.columns:
        return np.nan
    return row['p_unc'].values[0]

def _check_pairwise_diff_tukey(df_base: pd.DataFrame, n_bins: int, alpha: float) -> tuple:
    """Performs Tukey-Kramer HSD bin-by-bin to evaluate pairwise differences between events."""
    diff_sh = diff_sf = diff_hf = False
    for b in range(n_bins):
        df_b = df_base[df_base['Bin'] == b]
        # Ensure variance exists and all 3 events are present in the bin for Tukey
        if df_b['Power'].nunique() > 1 and df_b['Event'].nunique() == 3:
            try:
                pt = pg.pairwise_tukey(data=df_b, dv='Power', between='Event')
                for _, row in pt.iterrows():
                    if row['p_tukey'] < alpha:
                        pair = {row['A'], row['B']}
                        if {'steps', 'grasp_hook'} == pair: diff_sh = True
                        elif {'steps', 'grasp_floor'} == pair: diff_sf = True
                        elif {'grasp_hook', 'grasp_floor'} == pair: diff_hf = True
            except Exception:
                pass
    return diff_sh, diff_sf, diff_hf

def _load_merged_session(subject: str, session: str, hand: str | None = 'contra'):
    """
    Loads every event type of a session through load_multitaper_epochs (T-07) and merges
    them into one tensor and one trials table, so the 3 action classes (steps, grasp_hook,
    grasp_floor) can be compared together.

    hand : {'contra', 'ipsi', None}, default 'contra'
        Keeps only trials whose Hand is contralateral / ipsilateral to the implanted
        hemisphere, or all trials if None. Replaces the old substring `label_filter`.

    Raises ValueError if the merged event types were normalised differently (T-04b):
    every event type of a session must share the same mu/sigma, otherwise a leftover
    per-event reference mismatch would look like a small, constant power offset between
    classes in the between-class tests these functions run.

    Returns (mt_tensor, trials, freqs, times_s), or (None, None, None, None) if no .npz
    file was found for any event type of this session.
    """
    if hand not in ('contra', 'ipsi', None):
        raise ValueError(f"hand must be 'contra', 'ipsi' or None, got {hand!r}")

    tensors, trials_list = [], []
    freqs = times_s = None
    ref_normalization = ref_robust = ref_mu = ref_sigma = None

    for event_type in EVENT_SUFFIXES:
        try:
            data = load_multitaper_epochs(subject, session, event_type)
        except FileNotFoundError:
            continue

        if freqs is None:
            freqs, times_s = data['freqs'], data['times_s']

        file_normalization = str(data['normalization'])
        file_robust = bool(data['robust'])
        file_mu, file_sigma = data['mu'], data['sigma']
        if ref_normalization is None:
            ref_normalization, ref_robust, ref_mu, ref_sigma = (
                file_normalization, file_robust, file_mu, file_sigma
            )
        elif not (
            file_normalization == ref_normalization
            and file_robust == ref_robust
            and np.array_equal(file_mu, ref_mu)
            and np.array_equal(file_sigma, ref_sigma)
        ):
            raise ValueError(
                f"The '{event_type}' .npz of {subject}/{session} was normalised "
                f"differently from another event type of the same session "
                f"(normalization/robust/mu/sigma do not match). Re-run stage 3 "
                f"(extract_multitaper_epochs) for every event type of this session so "
                f"they share the same session-level reference."
            )

        tensors.append(data['mt_tensor'])
        trials_list.append(data['trials'])

    if not tensors:
        return None, None, None, None

    mt_tensor = np.concatenate(tensors, axis=0)
    trials = pd.concat(trials_list, ignore_index=True)

    if hand is not None:
        keep = trials['is_contralateral'] if hand == 'contra' else ~trials['is_contralateral']
        mt_tensor = mt_tensor[keep.values]
        trials = trials[keep.values].reset_index(drop=True)

    return mt_tensor, trials, freqs, times_s

def analyze_event_modulation(subject: str, session: str, hand: str | None = 'contra') -> pd.DataFrame:
    """
    Evaluates event-specific temporal modulation for each frequency band using 1-way RM ANOVA.
    Returns a DataFrame with channels, raw p-values, FDR-corrected p-values, and modulation flags.
    """
    bands_dict = FREQ_BANDS
    target_fs = MULTITAPER_PARAMS['target_fs']
    bin_size_ms = STATISTICAL_PARAMS['bin_size_ms']
    alpha = STATISTICAL_PARAMS['alpha']
    out_folder = PROCESSED_DATA_DIR / subject / session

    # 1. LOAD, JOIN AND MERGE EVERY EVENT TYPE (T-07)
    mt_tensor, trials, freqs, times_s = _load_merged_session(subject, session, hand=hand)
    if mt_tensor is None:
        print(f"[ERROR] No multitaper .npz files found for {subject}/{session}")
        return pd.DataFrame()

    labels = trials['action_class'].values

    unique, counts = np.unique(labels, return_counts=True)
    print(f"[INFO] Trials: {dict(zip(unique, counts))}")

    num_trials, num_freqs, num_times, num_channels = mt_tensor.shape
    n_samples_per_bin = int((bin_size_ms / 1000.0) * target_fs)
    n_bins = num_times // n_samples_per_bin
    csv_records = []

    # 2. ANALYSIS LOOP (per band)
    for band_name, target_band in bands_dict.items():
        print(f"\n{'='*60}")
        print(f"  {band_name.upper()} band ({target_band[0]}-{target_band[1]} Hz)")
        print(f"{'='*60}")
        
        band_mask = (freqs >= target_band[0]) & (freqs <= target_band[1])
        power_band = np.mean(mt_tensor[:, band_mask, :, :], axis=1)
        power_band_trunc = power_band[:, :n_bins * n_samples_per_bin, :]
        power_binned = power_band_trunc.reshape(num_trials, n_bins, n_samples_per_bin, num_channels).mean(axis=2)
 
        df_base = pd.DataFrame({
            'Trial': np.repeat(np.arange(num_trials), n_bins),
            'Event': np.repeat(labels, n_bins),
            'Bin':   np.tile(np.arange(n_bins), num_trials)
        })
        
        mask_steps = df_base['Event'] == 'steps'
        mask_hook  = df_base['Event'] == 'grasp_hook'
        mask_floor = df_base['Event'] == 'grasp_floor'

        p_steps_raw, p_hook_raw, p_floor_raw = [], [], []
 
        for ch in tqdm(range(num_channels), desc="Channels", leave=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_base['Power'] = power_binned[:, :, ch].flatten()
                
                df_s = df_base[mask_steps]
                df_h = df_base[mask_hook]
                df_f = df_base[mask_floor]
                
                # 1-way RM ANOVA per event (correction=False avoids Singular Matrix errors on small N)
                try: 
                    res_s = pg.rm_anova(dv='Power', within='Bin', subject='Trial', data=df_s, detailed=True, correction=False)
                    p_s = _get_p_val(res_s, 'Bin')
                except Exception: p_s = np.nan
                
                try: 
                    res_h = pg.rm_anova(dv='Power', within='Bin', subject='Trial', data=df_h, detailed=True, correction=False)
                    p_h = _get_p_val(res_h, 'Bin')
                except Exception: p_h = np.nan
                
                try: 
                    res_f = pg.rm_anova(dv='Power', within='Bin', subject='Trial', data=df_f, detailed=True, correction=False)
                    p_f = _get_p_val(res_f, 'Bin')
                except Exception: p_f = np.nan
                
                p_steps_raw.append(p_s)
                p_hook_raw.append(p_h)
                p_floor_raw.append(p_f)
                
        # 3. FDR CORRECTION ACROSS CHANNELS
        _, p_steps_fdr, _, _ = multipletests(np.nan_to_num(p_steps_raw, nan=1.0), alpha=alpha, method='fdr_bh')
        _, p_hook_fdr, _, _  = multipletests(np.nan_to_num(p_hook_raw, nan=1.0), alpha=alpha, method='fdr_bh')
        _, p_floor_fdr, _, _ = multipletests(np.nan_to_num(p_floor_raw, nan=1.0), alpha=alpha, method='fdr_bh')
        
        # 4. RECORD RESULTS
        for ch in range(num_channels):
            is_mod_steps = p_steps_fdr[ch] < alpha
            is_mod_hook  = p_hook_fdr[ch] < alpha
            is_mod_floor = p_floor_fdr[ch] < alpha
            
            csv_records.append({
                'Band': band_name,
                'Channel': ch,
                'p_raw_steps': p_steps_raw[ch],
                'p_fdr_steps': p_steps_fdr[ch],
                'is_modulated_steps': is_mod_steps,
                'p_raw_hook': p_hook_raw[ch],
                'p_fdr_hook': p_hook_fdr[ch],
                'is_modulated_hook': is_mod_hook,
                'p_raw_floor': p_floor_raw[ch],
                'p_fdr_floor': p_floor_fdr[ch],
                'is_modulated_floor': is_mod_floor
            })
            
        # Console Summary
        print(f"\n  Modulated Channels (FDR < {alpha}):")
        print(f"    - Steps: {np.sum(np.array(p_steps_fdr) < alpha)}")
        print(f"    - Grasp Hook: {np.sum(np.array(p_hook_fdr) < alpha)}")
        print(f"    - Grasp Floor: {np.sum(np.array(p_floor_fdr) < alpha)}")
   
    # 5. SAVE MASTER CSV
    out_path = out_folder / "event_modulation_results.csv"
    
    if csv_records:
        df_out = pd.DataFrame(csv_records)
        df_out.to_csv(out_path, index=False)
        print(f"\n[INFO] Event modulation results saved to:\n{out_path}")
        return df_out
    else:
        print(f"\n[WARNING] No records found.")
        return pd.DataFrame()
    
def analyze_selectivity(subject: str, session: str, hand: str | None = 'contra') -> pd.DataFrame:
    """
    Strict Mixed ANOVA approach for 3 motor actions.
    """
    bands_dict = FREQ_BANDS
    target_fs = MULTITAPER_PARAMS['target_fs']
    bin_size_ms = STATISTICAL_PARAMS['bin_size_ms']
    alpha = STATISTICAL_PARAMS['alpha']
    out_folder = PROCESSED_DATA_DIR / subject / session

    # 1. LOAD, JOIN AND MERGE EVERY EVENT TYPE (T-07)
    mt_tensor, trials, freqs, times_s = _load_merged_session(subject, session, hand=hand)
    if mt_tensor is None:
        print(f"[ERROR] No multitaper .npz files found for {subject}/{session}")
        return pd.DataFrame()

    labels = trials['action_class'].values

    unique, counts = np.unique(labels, return_counts=True)
    print(f"[INFO] Trials: {dict(zip(unique, counts))}")

    num_trials, num_freqs, num_times, num_channels = mt_tensor.shape
    n_samples_per_bin = int((bin_size_ms / 1000.0) * target_fs)
    n_bins = num_times // n_samples_per_bin
    n_bins_800ms = int(800 / bin_size_ms)
    csv_records = []

    # 2. ANALYSIS LOOP (per band)
    for band_name, target_band in bands_dict.items():
        print(f"\n{'='*60}")
        print(f"  {band_name.upper()} band ({target_band[0]}-{target_band[1]} Hz)")
        print(f"{'='*60}")
        
        band_mask = (freqs >= target_band[0]) & (freqs <= target_band[1])
        power_band = np.mean(mt_tensor[:, band_mask, :, :], axis=1)
        power_band_trunc = power_band[:, :n_bins * n_samples_per_bin, :]
        power_binned = power_band_trunc.reshape(num_trials, n_bins, n_samples_per_bin, num_channels).mean(axis=2)
 
        # Keep only the premovement phase
        power_binned = power_binned[:, :n_bins_800ms, :]
        df_base = pd.DataFrame({
            'Trial': np.repeat(np.arange(num_trials), n_bins_800ms),
            'Event': np.repeat(labels, n_bins_800ms),
            'Bin':   np.tile(np.arange(n_bins_800ms), num_trials)
        })
        
        mask_steps = df_base['Event'] == 'steps'
        mask_hook  = df_base['Event'] == 'grasp_hook'
        mask_floor = df_base['Event'] == 'grasp_floor'

        # Tracking variables
        p_int_raw, p_bin_raw = [], []
        channel_results = []
 
        for ch in tqdm(range(num_channels), desc="Channels", leave=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_base['Power'] = power_binned[:, :, ch].flatten()
                
                # STEP A: Global Mixed ANOVA (Main & Interaction)
                try:
                    res_mixed = pg.mixed_anova(
                        dv='Power', within='Bin', between='Event', 
                        subject='Trial', data=df_base
                    )
                    p_interaction = _get_p_val(res_mixed, 'Interaction')
                    p_main_bin    = _get_p_val(res_mixed, 'Bin')
                except Exception:
                    p_interaction = np.nan
                    p_main_bin    = np.nan

                # STEP B: Pairwise differences between events (Post-Hoc via Tukey-Kramer)
                pair_sh_diff, pair_sf_diff, pair_hf_diff = _check_pairwise_diff_tukey(df_base, n_bins_800ms, alpha)
                
                # Append raw results
                p_int_raw.append(p_interaction)
                p_bin_raw.append(p_main_bin)
                
                channel_results.append({
                    'pair_sh_diff': pair_sh_diff, 
                    'pair_sf_diff': pair_sf_diff, 
                    'pair_hf_diff': pair_hf_diff,
                })
        
        # 3. FDR CORRECTION
        _, p_int_fdr, _, _ = multipletests(np.nan_to_num(p_int_raw, nan=1.0), alpha=alpha, method='fdr_bh')
        _, p_bin_fdr, _, _ = multipletests(np.nan_to_num(p_bin_raw, nan=1.0), alpha=alpha, method='fdr_bh')
        
        # 4. RIGOROUS CLASSIFICATION (Relative Tuning)
        for ch in range(num_channels):
            r = channel_results[ch]
            diff_SH, diff_SF, diff_HF = r['pair_sh_diff'], r['pair_sf_diff'], r['pair_hf_diff']
 
            if p_int_fdr[ch] >= alpha:
                # Interaction NS: Tasks do not statistically differ in temporal profile.
                category = 'motor_aspecific' if p_bin_fdr[ch] < alpha else 'non_informative'
            else:
                # Interaction Sig: Channel discriminates! Assign based strictly on pairwise contrast logic.
                if diff_SH and diff_SF and not diff_HF:
                    category = 'steps_specific'  # Steps diverges from both Hook and Floor (which are similar)
                elif diff_SH and diff_HF and not diff_SF:
                    category = 'hook_specific'   # Hook diverges from both Steps and Floor (which are similar)
                elif diff_SF and diff_HF and not diff_SH:
                    category = 'floor_specific'  # Floor diverges from both Steps and Hook (which are similar)
                elif diff_SH and diff_SF and diff_HF:
                    category = 'motor_specific' # All 3 actions are statistically distinct from each other
                elif diff_SH and not diff_SF and not diff_HF:
                    category = 'mixed_steps_hook_diff' # Incomplete separation (only S and H differ)
                elif diff_SF and not diff_SH and not diff_HF:
                    category = 'mixed_steps_floor_diff'
                elif diff_HF and not diff_SH and not diff_SF:
                    category = 'mixed_hook_floor_diff'
                else:
                    category = 'ambiguous' # Interaction sig, but Tukey-Kramer post-hocs are too conservative to catch the specific bins

            csv_records.append({
                'Band': band_name,
                'Channel': ch,
                'p_interaction_raw': p_int_raw[ch],
                'p_interaction_fdr': p_int_fdr[ch],
                'p_main_bin_raw': p_bin_raw[ch],
                'p_main_bin_fdr': p_bin_fdr[ch],
                'pair_steps_hook_diff':  diff_SH,
                'pair_steps_floor_diff': diff_SF,
                'pair_hook_floor_diff':  diff_HF,
                'category': category,
            })
 
        # Print band summary
        band_records = csv_records[-num_channels:]
        cats = Counter([r['category'] for r in band_records])

        print(f"\n  Interaction significant (FDR): {np.sum(p_int_fdr < alpha)}")
        print(f"  Categories:")
        for cat, count in sorted(cats.items()):
            print(f"    - {cat}: {count}")
   
    # 5. SAVE MASTER CSV
    out_path = out_folder / "selectivity_results.csv"
    
    if csv_records:
        df_out = pd.DataFrame(csv_records)
        df_out.to_csv(out_path, index=False)
        print(f"\n[INFO] Selectivity results saved to:\n{out_path}")
        return df_out
    else:
        print(f"\n[WARNING] No records found.")
        return pd.DataFrame()
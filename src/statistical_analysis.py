import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, ttest_1samp
from tqdm import tqdm
from collections import Counter
import warnings
from src.config import (PROCESSED_DATA_DIR, FREQ_BANDS, MULTITAPER_PARAMS, STATISTICAL_PARAMS)

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

def analyze_selectivity(subject: str, session: str, label_filter: str = None) -> pd.DataFrame:
    """
    Strict Mixed ANOVA approach for 3 motor actions.
    """
    bands_dict = FREQ_BANDS
    target_fs = MULTITAPER_PARAMS['target_fs']
    window_s = MULTITAPER_PARAMS['window_taper_s']
    bin_size_ms = STATISTICAL_PARAMS['bin_size_ms']
    alpha = STATISTICAL_PARAMS['alpha']
    out_folder = PROCESSED_DATA_DIR / subject / session
    
    # 1. LOAD AND MERGE ALL MULTITAPER FILES
    file_pattern = f"epoched_multitaper_*_{int(target_fs)}Hz_{int(window_s*1000)}ms.npz"
    npz_files = list(out_folder.glob(file_pattern))
    
    if not npz_files:
        print(f"[ERROR] No files found matching: {file_pattern}")
        return pd.DataFrame()
        
    print(f"[INFO] Found {len(npz_files)} .npz files. Merging...")
    
    all_tensors, all_labels = [], []
    freqs = None
    
    for file_path in npz_files:
        with np.load(file_path, allow_pickle=True) as data:
            all_tensors.append(data['mt_tensor'])
            all_labels.append(data['labels'])
            if freqs is None:
                freqs = data['freqs']
                
    mt_tensor = np.concatenate(all_tensors, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # 2. FILTER AND RE-LABEL INTO 3 ACTIONS
    if label_filter is not None:
        filter_mask = np.array([label_filter in str(l) for l in labels])
        mt_tensor = mt_tensor[filter_mask]
        labels = labels[filter_mask]
    
    new_labels = []
    keep_mask = []
    for lbl in labels:
        lbl_str = str(lbl).lower()
        if 'step' in lbl_str:
            new_labels.append('steps')
            keep_mask.append(True)
        elif 'hook' in lbl_str:
            new_labels.append('grasp_hook')
            keep_mask.append(True)
        elif 'floor' in lbl_str and 'step' not in lbl_str:
            new_labels.append('grasp_floor')
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            
    keep_mask = np.array(keep_mask)
    mt_tensor = mt_tensor[keep_mask]
    labels = np.array(new_labels)
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[INFO] Trials: {dict(zip(unique, counts))}")
    
    num_trials, num_freqs, num_times, num_channels = mt_tensor.shape
    n_samples_per_bin = int((bin_size_ms / 1000.0) * target_fs)
    n_bins = num_times // n_samples_per_bin
    csv_records = []
    
    # 3. ANALYSIS LOOP (per band)
    for band_name, target_band in bands_dict.items():
        print(f"\n{'='*60}")
        print(f"  {band_name.upper()} band ({target_band[0]}-{target_band[1]} Hz)")
        print(f"{'='*60}")
        
        band_mask = (freqs >= target_band[0]) & (freqs <= target_band[1])
        power_band = np.mean(mt_tensor[:, band_mask, :, :], axis=1)
        power_band_trunc = power_band[:, :n_bins * n_samples_per_bin, :]
        power_binned = power_band_trunc.reshape(num_trials, n_bins, n_samples_per_bin, num_channels).mean(axis=2)
 
        # Pre-build base DataFrame
        df_base = pd.DataFrame({
            'Trial': np.repeat(np.arange(num_trials), n_bins),
            'Event': np.repeat(labels, n_bins),
            'Bin':   np.tile(np.arange(n_bins), num_trials)
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
                pair_sh_diff, pair_sf_diff, pair_hf_diff = _check_pairwise_diff_tukey(df_base, n_bins, alpha)
                
                # Append raw results
                p_int_raw.append(p_interaction)
                p_bin_raw.append(p_main_bin)
                
                channel_results.append({
                    'pair_sh_diff': pair_sh_diff, 
                    'pair_sf_diff': pair_sf_diff, 
                    'pair_hf_diff': pair_hf_diff,
                })
        
        # 4. FDR CORRECTION
        _, p_int_fdr, _, _ = multipletests(np.nan_to_num(p_int_raw, nan=1.0), alpha=alpha, method='fdr_bh')
        _, p_bin_fdr, _, _ = multipletests(np.nan_to_num(p_bin_raw, nan=1.0), alpha=alpha, method='fdr_bh')
        
        # 5. RIGOROUS CLASSIFICATION (Relative Tuning)
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
   
    # 6. SAVE MASTER CSV
    out_path = out_folder / "selectivity_results.csv"
    
    if csv_records:
        df_out = pd.DataFrame(csv_records)
        df_out.to_csv(out_path, index=False)
        print(f"\n[INFO] Selectivity results saved to:\n{out_path}")
        return df_out
    else:
        print(f"\n[WARNING] No records found.")
        return pd.DataFrame()
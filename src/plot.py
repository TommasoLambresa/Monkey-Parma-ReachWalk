import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from src.io import load_multitaper_epochs
from src.config import FREQ_BANDS, EPOCH_T_PRE, SESSION_METADATA, SIMILARITY_PARAMS

def plot_interactive_multitaper(subject: str, session: str, event_type: str, hand: str | None = 'contra') -> None:
    """
    Renders an interactive Time-Frequency Representation (Spectrogram) of the Multitaper data.
    Uses a scrollable list for channel selection.

    hand : {'contra', 'ipsi', None}, default 'contra'
        Keeps only trials whose Hand is contralateral / ipsilateral to the implanted
        hemisphere, or all trials if None.
    """

    try:
        multitaper_dict = load_multitaper_epochs(subject, session, event_type)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    mt_tensor = multitaper_dict['mt_tensor']
    trials = multitaper_dict['trials']
    freqs = multitaper_dict['freqs']
    time_vector = multitaper_dict['times_s']

    if hand is not None:
        keep = trials['is_contralateral'] if hand == 'contra' else ~trials['is_contralateral']
        mt_tensor = mt_tensor[keep.values]
        trials = trials[keep.values].reset_index(drop=True)

    num_trials, num_freqs, num_times, num_channels = mt_tensor.shape

    if num_trials == 0:
        print(f"No trial found for hand='{hand}'.")
        return

    # Create a scrollable list (Select widget) for channels
    channel_options = [(f"Ch {i}", i) for i in range(num_channels)]
    channel_selector = widgets.Select(
        options=channel_options,
        value=0,
        description='Ch:',
        rows=20,
        layout=widgets.Layout(width='150px', min_width='150px')
    )
    
    plot_output = widgets.Output()

    def update_plot(change) -> None:
        channel_idx = change.new if change is not None else channel_selector.value
        
        with plot_output:
            plot_output.clear_output(wait=True)
            
            # 3. Draw a SINGLE plot (no more loops or exploding grids)
            fig, ax = plt.subplots(figsize=(8, 5))

            # Average of all extracted trials for that channel
            mean_power = np.mean(mt_tensor[:, :, :, channel_idx], axis=0)

            # 4. Scaling by the maximum absolute value
            max_val = np.max(np.abs(mean_power))
            scaled_power = mean_power / max_val

            im = ax.pcolormesh(
                time_vector, 
                freqs, 
                scaled_power, 
                cmap='turbo', 
                vmin=-0.5, 
                vmax=0.5, 
                shading='gouraud'
            )

            # Plot aesthetics
            ax.axvline(x=0.0, color='black', linestyle='-', linewidth=1.5)
            ax.set_ylim(0, 100)
            ax.set_xlim(-0.8, 0.5)
            # Dynamic title
            filter_str = f" | Hand: {hand}" if hand else " | All Trials"
            ax.set_title(f"Averaged Trials (N={num_trials})", fontweight='bold', fontsize=11)
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time [s]")

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized power (a.u.)", fontsize=11)

            fig.suptitle(f"{subject} / {session} | Event: {event_type}{filter_str} | Ch: {channel_idx}", fontsize=14)
            plt.tight_layout()
            plt.show()

    channel_selector.observe(update_plot, names='value')
    update_plot(None)
    
    display(widgets.HBox([channel_selector, plot_output]))

def plot_population_modulation(subject: str, session: str, band: str, hand: str | None = 'contra',
                             offset: str | None = 'common') -> None:
    """
    Generates 3 heatmaps (Steps, Grasp Hook, Grasp Floor), each sorted independently by its
    own modulation score. The Steps panel uses only the first step of each walk (walk
    initiation), not ongoing locomotion.

    hand : {'contra', 'ipsi', None}, default 'contra'
        Keeps only trials whose Hand is contralateral / ipsilateral to the implanted
        hemisphere, or all trials if None. For steps, the hand filter is applied after
        selecting the first step of each walk, so walks whose first contact is ipsilateral
        contribute nothing.
    offset : {'common', 'condition', None}, default 'common'
        Plain language: removes the large, nearly constant level that the session normalisation
        leaves in the band power, so only the change over time is left and the colour scale is
        not squeezed. With m_c(t) the trial-averaged band power of channel c (session-z units,
        n_channels x n_times):
        'common'    : subtracts from m_c(t) an offset o_c, the mean of the pre-movement window
                      [-EPOCH_T_PRE, 0] s averages of the conditions that have trials, so level
                      differences between Steps, Hook and Floor survive. Caveat: the three
                      centred traces of a channel sum to zero in the window (when all three
                      classes have trials), so any between-class "mirror" in this plot is
                      partly imposed by the centring and must not be read as opposite
                      activation. Use plot_class_distances and
                      plot_pairwise_difference_heatmaps for that question.
        'condition' : single-trial normalisation. Every trial is z-scored on its own, for each
                      frequency and channel, along the whole epoch [-EPOCH_T_PRE, +EPOCH_T_POST]
                      s (mean 0, standard deviation 1 over time), then averaged over the band
                      frequencies and across the trials of the condition. A z-score is
                      unchanged by the session shift and scale, so the result equals that of
                      extract_multitaper_epochs(normalization='trial'). Level and amplitude
                      differences between trials, and between conditions, are removed. Because
                      every trial has epoch mean 0, the pre-movement window mean of a trial is
                      minus its post-event part: the Fac/Supp score below tells whether the
                      window lies above or below the rest of the epoch, not whether it is
                      modulated against a baseline. Trial-averaged traces have amplitude below 1
                      because trials with different time courses partly cancel.
        None        : no offset removal.
    The modulation score and the row order are unchanged: window mean of the scaled traces.
    """

    if offset not in ('common', 'condition', None):
        raise ValueError(f"offset must be 'common', 'condition' or None, got {offset!r}.")

    # --- 1. DATA LOADING ---
    try:
        mt_dict_steps = load_multitaper_epochs(subject, session, 'steps')
        mt_dict_grasp = load_multitaper_epochs(subject, session, 'grasp')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    tensor_steps = mt_dict_steps['mt_tensor']
    trials_steps = mt_dict_steps['trials']

    tensor_grasp = mt_dict_grasp['mt_tensor']
    trials_grasp = mt_dict_grasp['trials']

    # Steps: first step of each walk only, then the hand filter if specified
    keep_steps = trials_steps['is_first_in_walk'].values
    if hand is not None:
        keep_steps = keep_steps & (trials_steps['is_contralateral'].values == (hand == 'contra'))
    tensor_steps = tensor_steps[keep_steps]

    if hand is not None:
        keep_grasp = trials_grasp['is_contralateral'] if hand == 'contra' else ~trials_grasp['is_contralateral']
        tensor_grasp = tensor_grasp[keep_grasp.values]
        trials_grasp = trials_grasp[keep_grasp.values].reset_index(drop=True)

    mask_hook = (trials_grasp['action_class'] == 'grasp_hook').values
    mask_floor = (trials_grasp['action_class'] == 'grasp_floor').values

    tensor_hook = tensor_grasp[mask_hook]
    tensor_floor = tensor_grasp[mask_floor]

    freqs = mt_dict_steps['freqs']
    time_vector = mt_dict_steps['times_s']
    num_channels = tensor_steps.shape[3]
    num_times = tensor_steps.shape[2]

    # Operational masks
    band_mask = (freqs >= FREQ_BANDS[band][0]) & (freqs <= FREQ_BANDS[band][1])
    time_mod_mask = (time_vector >= -EPOCH_T_PRE) & (time_vector <= 0.0)
    
    # Helper: band power averaged over the band frequencies and across trials -> (Channels, Time).
    # With zscore_trials, each trial is z-scored per frequency and channel along time before the
    # band average. Trials are processed one at a time to keep memory low.
    def get_band_mean(tensor, zscore_trials=False):
        if tensor.shape[0] == 0: return np.zeros((num_channels, num_times))
        per_trial = np.empty((tensor.shape[0], num_times, num_channels))
        for i, trial in enumerate(tensor):
            band_power = trial[band_mask].astype(np.float64)  # (band freqs, times, channels)
            if zscore_trials:
                trial_std = band_power.std(axis=1, keepdims=True)
                trial_std = np.where(trial_std == 0, 1.0, trial_std)
                band_power = (band_power - band_power.mean(axis=1, keepdims=True)) / trial_std
            per_trial[i] = band_power.mean(axis=0)
        return per_trial.mean(axis=0).T

    zscore_trials = offset == 'condition'
    z_steps = get_band_mean(tensor_steps, zscore_trials)
    z_hook = get_band_mean(tensor_hook, zscore_trials)
    z_floor = get_band_mean(tensor_floor, zscore_trials)

    # Common offset: subtract each channel's mean level in the pre-movement window, averaged
    # over the conditions that have trials. A condition without trials stays at zero.
    if offset == 'common':
        z_all = [z_steps, z_hook, z_floor]
        has_trials = [t.shape[0] > 0 for t in (tensor_steps, tensor_hook, tensor_floor)]
        win_means = [z[:, time_mod_mask].mean(axis=1) for z in z_all]
        common = np.mean([w for w, ok in zip(win_means, has_trials) if ok], axis=0)
        z_steps, z_hook, z_floor = [z - common[:, None] if ok else z for z, ok in zip(z_all, has_trials)]

    # --- 2. GLOBAL FRACTIONAL NORMALIZATION ---
    z_norm_steps = np.zeros_like(z_steps)
    z_norm_hook = np.zeros_like(z_hook)
    z_norm_floor = np.zeros_like(z_floor)
    
    for ch in range(num_channels):
        max_val = max(
            np.max(np.abs(z_steps[ch])),
            np.max(np.abs(z_hook[ch])),
            np.max(np.abs(z_floor[ch]))
        )
        if max_val == 0: max_val = 1.0 # Prevents division by zero 
        
        z_norm_steps[ch] = z_steps[ch] / max_val
        z_norm_hook[ch] = z_hook[ch] / max_val
        z_norm_floor[ch] = z_floor[ch] / max_val

    # --- 3. PLOT CONFIGURATION AND SORTING ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    conditions = [
        ("Steps", z_norm_steps),
        ("Grasp Hook", z_norm_hook),
        ("Grasp Floor", z_norm_floor)
    ]

    for ax, (cond_name, z_norm_cond) in zip(axes, conditions):

        # Calculate Modulation Score (in the pre-event window [-EPOCH_T_PRE, 0.0])
        scores = np.mean(z_norm_cond[:, time_mod_mask], axis=1)

        # Classification
        fac_idx = np.where(scores > 0)[0]
        sup_idx = np.where(scores <= 0)[0]

        # Each panel is sorted independently
        fac_idx_sorted = fac_idx[np.argsort(scores[fac_idx])[::-1]]
        sup_idx_sorted = sup_idx[np.argsort(np.abs(scores[sup_idx]))]
        final_order = np.concatenate([fac_idx_sorted, sup_idx_sorted])

        z_sorted = z_norm_cond[final_order, :]
        
        # Plot Heatmap (Extent fixes the Y-axis to match row 0 at the top)
        im = ax.imshow(
            z_sorted, 
            aspect='auto', 
            cmap='RdBu_r', 
            vmin=-1.0, 
            vmax=1.0,
            extent=[time_vector[0], time_vector[-1], num_channels, 0],
            interpolation='none'
        )
        
        # Overlay Mean Traces (On a secondary transparent Y-axis)
        ax_traces = ax.twinx()
        trace_all = np.mean(z_norm_cond, axis=0)
        
        ax_traces.plot(time_vector, trace_all, color='black', linewidth=2.5, zorder=3)
        
        # if len(fac_idx) > 0:
        #     trace_fac = np.mean(z_norm_cond[fac_idx, :], axis=0)
        #     ax_traces.plot(time_vector, trace_fac, color='red', linewidth=1.5, zorder=2)
        # if len(sup_idx) > 0:
        #     trace_sup = np.mean(z_norm_cond[sup_idx, :], axis=0)
        #     ax_traces.plot(time_vector, trace_sup, color='blue', linewidth=1.5, zorder=2)
            
        ax_traces.set_ylim(-0.7, 0.7) # Customizable scale depending on how high your curves go

        if ax != axes[-1]: 
            ax_traces.set_yticks([])
        else:
            ax_traces.set_ylabel("Avg Normalized Power", fontsize=11, rotation=270, labelpad=15)
            
        # Aesthetic Details
        ax.axvline(x=0.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(f"{cond_name}\n(Fac: {len(fac_idx)} | Supp: {len(sup_idx)})", fontweight='bold', pad=10)
        ax.set_xlabel("Time [s]")
        
        if ax == axes[0]:
            ax.set_ylabel("Channel # (Sorted by Magnitude)")
            
    # Global shared colorbar for the three figures
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.08)
    cbar.set_label(f"Normalized {band.capitalize()} Power (a.u.)", fontsize=11)
    
    plt.suptitle(f"{subject} / {session} | Population {band.capitalize()} Power", fontsize=16, fontweight='bold')
    plt.show()


def plot_class_distances(results: dict, subject: str, band: str) -> None:
    """
    Shows the cross-validated Mahalanobis distances between the class maps of one subject
    and band, and the two tests that compare them. Plain words: panel (a) asks whether each
    pair of classes is farther apart than label shuffling would produce; panels (b) and (c)
    ask which pair is closer.

    results : dict
        Output of src.similarity.compute_class_distances for this subject.
    band : str
        Key of FREQ_BANDS.

    (a) The three squared distances per channel (S = first step, H = hook grasp, F = floor
        grasp), each with the 95th percentile of its permutation null (labels of the two
        classes shuffled within session).
    (b), (c) Null histograms of delta_SH_minus_SF (positive: Floor closer to Steps than Hook
        is) and delta_SF_minus_HF (positive: Floor closer to Hook than to Steps, H_goal;
        negative: H_kin), with the observed value marked. Holm-corrected p across the five
        bands is in the panel titles.
    The title reports trial counts per class, the number of sessions and the Ledoit-Wolf
    weight lambda.
    """
    band_results = results[band]
    totals = band_results['counts'].sum(axis=1)
    observed, null = band_results['observed'], band_results['null']

    fig, (ax_d2, ax_sh_sf, ax_sf_hf) = plt.subplots(1, 3, figsize=(15, 4.5))

    pair_names = ['d2_SH', 'd2_SF', 'd2_HF']
    pair_labels = ['Steps-Hook', 'Steps-Floor', 'Hook-Floor']
    positions = np.arange(len(pair_names))
    ax_d2.bar(positions, [observed[n] for n in pair_names], color='0.55', width=0.6, label='observed')
    for position, name in zip(positions, pair_names):
        if not np.all(np.isnan(null[name])):
            ax_d2.hlines(np.nanpercentile(null[name], 95), position - 0.35, position + 0.35,
                         color='crimson', linewidth=2.5,
                         label='95th percentile of null' if position == 0 else None)
    ax_d2.axhline(0, color='black', linewidth=0.8)
    ax_d2.set_xticks(positions)
    ax_d2.set_xticklabels([f"{label}\np_holm={band_results['p_holm'][n]:.3f}"
                           for label, n in zip(pair_labels, pair_names)])
    ax_d2.set_ylabel("Squared distance per channel (noise SD$^2$)")
    ax_d2.set_title("(a) Separation of each pair")
    ax_d2.legend(frameon=False, fontsize=9)

    for ax, name, panel, meaning in (
        (ax_sh_sf, 'delta_SH_minus_SF', '(b)', 'positive: Floor closer to Steps than Hook is'),
        (ax_sf_hf, 'delta_SF_minus_HF', '(c)', 'positive: H_goal, negative: H_kin'),
    ):
        if not np.all(np.isnan(null[name])):
            ax.hist(null[name][~np.isnan(null[name])], bins=40, color='0.75')
        ax.axvline(observed[name], color='crimson', linewidth=2.5)
        ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_xlabel(f"{name}  ({meaning})", fontsize=9)
        ax.set_ylabel("Permutations")
        ax.set_title(f"{panel} {name}\nobserved={observed[name]:.4f}, p_holm={band_results['p_holm'][name]:.3f}")

    fig.suptitle(
        f"{subject} | {band.capitalize()} | n hook={totals['hook']}, floor={totals['floor']}, "
        f"first step={totals['first_step']} | {band_results['counts'].shape[1]} sessions | "
        f"lambda={band_results['lambda']:.2f}",
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    plt.show()


def plot_pairwise_difference_heatmaps(subject: str, band: str) -> None:
    """
    Draws the difference between class-mean maps as channels x time heatmaps, pooled across
    the sessions of the subject: Steps - Hook, Steps - Floor, Hook - Floor. Descriptive
    only: if Floor is closer to Steps, the Steps - Floor panel is paler than Steps - Hook.

    Trial selection is the one of src.similarity.compute_class_distances: contralateral
    hand only, Hook and Floor grasps, Steps = first step of each walk. Units are session-z
    (no per-channel scaling); one symmetric colour limit, taken from the data, is shared by
    the three panels. The dotted and dashed vertical lines bound the window
    [-EPOCH_T_PRE, 0] used by the distance analysis.

    Channel order comes from a random half of the trials (seeded, stratified by class x
    session): the window mean of Steps - Hook, largest first. The panels are drawn from the
    other half, so the order does not select the channels that look different in the data
    shown.
    """
    class_names = ['steps', 'hook', 'floor']
    band_power = {name: [] for name in class_names}
    session_of_trial = {name: [] for name in class_names}
    times_s = None
    sessions = SESSION_METADATA.index[SESSION_METADATA['Subject'] == subject]
    for session_index, session in enumerate(sessions):
        loaded = {event_type: load_multitaper_epochs(subject, session, event_type)
                  for event_type in ('steps', 'grasp')}
        for event_type, data in loaded.items():
            if str(data['normalization']) != 'session':
                raise ValueError(f"{session} ({event_type}) is not session-normalised.")
        if times_s is None:
            times_s = loaded['steps']['times_s']
        elif not np.array_equal(times_s, loaded['steps']['times_s']):
            raise ValueError(f"{session} has a different time axis from the other sessions.")

        steps_trials = loaded['steps']['trials']
        grasp_trials = loaded['grasp']['trials']
        selections = {
            'steps': (loaded['steps'], (steps_trials['is_first_in_walk'] & steps_trials['is_contralateral']).values),
            'hook': (loaded['grasp'], (grasp_trials['is_contralateral'] & (grasp_trials['action_class'] == 'grasp_hook')).values),
            'floor': (loaded['grasp'], (grasp_trials['is_contralateral'] & (grasp_trials['action_class'] == 'grasp_floor')).values),
        }
        for name, (data, keep) in selections.items():
            if keep.any():
                band_mask = (data['freqs'] >= FREQ_BANDS[band][0]) & (data['freqs'] <= FREQ_BANDS[band][1])
                band_power[name].append(
                    data['mt_tensor'][keep][:, band_mask].mean(axis=1, dtype=np.float64)
                )
                session_of_trial[name].append(np.full(int(keep.sum()), session_index))
        del loaded, selections, data

    min_trials = SIMILARITY_PARAMS['min_trials_per_class']
    for name in class_names:
        n_trials = sum(len(a) for a in session_of_trial[name])
        if n_trials < min_trials:
            print(f"[{subject}] Class '{name}' has {n_trials} pooled trials (< {min_trials}): no heatmaps.")
            return
        band_power[name] = np.concatenate(band_power[name])  # (n_trials, n_times, n_channels)
        session_of_trial[name] = np.concatenate(session_of_trial[name])

    # A random half of the trials of each class x session group picks the channel order.
    rng = np.random.default_rng(SIMILARITY_PARAMS['seed'])
    ordering_half = {}
    for name in class_names:
        ordering = np.zeros(len(session_of_trial[name]), dtype=bool)
        for session_index in np.unique(session_of_trial[name]):
            members = np.flatnonzero(session_of_trial[name] == session_index)
            n_ordering = (len(members) + rng.integers(0, 2)) // 2
            ordering[rng.permutation(members)[:n_ordering]] = True
        if ordering.all() or not ordering.any():
            print(f"[{subject}] Class '{name}' cannot be split into two halves: no heatmaps.")
            return
        ordering_half[name] = ordering

    window = (times_s >= -EPOCH_T_PRE - 1e-9) & (times_s < -1e-9)
    mean_ordering = {n: band_power[n][ordering_half[n]].mean(axis=0) for n in class_names}
    channel_order = np.argsort(-(mean_ordering['steps'] - mean_ordering['hook'])[window].mean(axis=0))
    mean_shown = {n: band_power[n][~ordering_half[n]].mean(axis=0) for n in class_names}

    panels = [
        ("Steps - Hook", mean_shown['steps'] - mean_shown['hook']),
        ("Steps - Floor", mean_shown['steps'] - mean_shown['floor']),
        ("Hook - Floor", mean_shown['hook'] - mean_shown['floor']),
    ]
    color_limit = max(np.abs(difference).max() for _, difference in panels)
    n_channels = panels[0][1].shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, (title, difference) in zip(axes, panels):
        image = ax.imshow(
            difference[:, channel_order].T, aspect='auto', cmap='RdBu_r',
            vmin=-color_limit, vmax=color_limit,
            extent=[times_s[0], times_s[-1], n_channels, 0], interpolation='none',
        )
        ax.axvline(-EPOCH_T_PRE, color='black', linestyle=':', linewidth=1.5)
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Time [s]")
    axes[0].set_ylabel("Channel # (ordered on the other half of the trials)")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03)
    cbar.set_label(f"Difference of mean {band} power (session-z)")
    n_shown = {n: int((~ordering_half[n]).sum()) for n in class_names}
    fig.suptitle(
        f"{subject} | {band.capitalize()} | pooled sessions, contralateral hand | trials shown: "
        f"steps={n_shown['steps']}, hook={n_shown['hook']}, floor={n_shown['floor']}",
        fontsize=13, fontweight='bold',
    )
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from src.io import load_multitaper_epochs
from src.config import FREQ_BANDS
from utils.get_channel_geometry_mapping import get_channel_geometry_mapping

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

            # 4. Scaling (Division by zero check removed as requested)
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

def plot_population_heatmaps(subject: str, session: str, band: str, hand: str | None = 'contra', order: bool = False) -> None:
    """
    Generates 3 heatmaps (Steps, Grasp Hook, Grasp Floor) with independent sorting based on modulation.

    hand : {'contra', 'ipsi', None}, default 'contra'
        Keeps only trials whose Hand is contralateral / ipsilateral to the implanted
        hemisphere, or all trials if None.
    """

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

    # Apply the hand filter if specified
    if hand is not None:
        keep_steps = trials_steps['is_contralateral'] if hand == 'contra' else ~trials_steps['is_contralateral']
        tensor_steps = tensor_steps[keep_steps.values]

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
    time_mod_mask = (time_vector >= -0.8) & (time_vector <= 0.0)
    
    # Helper: Extracts Band power and computes across-trial mean -> Output: (Channels, Time)
    def get_band_mean(tensor):
        if tensor.shape[0] == 0: return np.zeros((num_channels, num_times))
        band_power = np.mean(tensor[:, band_mask, :, :], axis=1)
        return np.mean(band_power, axis=0).T 

    z_steps = get_band_mean(tensor_steps)
    z_hook = get_band_mean(tensor_hook)
    z_floor = get_band_mean(tensor_floor)
    
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

    steps_order = None
    for ax, (cond_name, z_norm_cond) in zip(axes, conditions):
        
        # Calculate Modulation Score (in the pre-event window [-0.8, 0.0])
        scores = np.mean(z_norm_cond[:, time_mod_mask], axis=1)
        
        # Classification 
        fac_idx = np.where(scores > 0)[0]
        sup_idx = np.where(scores <= 0)[0]
        
        # Apply shared sorting if flag "order" (steps based) is True otherwise sort independently
        if order and steps_order is not None:
            final_order = steps_order
        else:
            # Independent sorting
            fac_idx_sorted = fac_idx[np.argsort(scores[fac_idx])[::-1]]
            sup_idx_sorted = sup_idx[np.argsort(np.abs(scores[sup_idx]))]
            final_order = np.concatenate([fac_idx_sorted, sup_idx_sorted])
            
            # Save the Steps order for the next iterations
            if order and cond_name == "Steps":
                steps_order = final_order
        
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
        
        if len(fac_idx) > 0:
            trace_fac = np.mean(z_norm_cond[fac_idx, :], axis=0)
            ax_traces.plot(time_vector, trace_fac, color='red', linewidth=1.5, zorder=2)
        if len(sup_idx) > 0:
            trace_sup = np.mean(z_norm_cond[sup_idx, :], axis=0)
            ax_traces.plot(time_vector, trace_sup, color='blue', linewidth=1.5, zorder=2)
            
        ax_traces.set_ylim(-0.4, 0.4) # Customizable scale depending on how high your curves go

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


def _band_time_map(mt_tensor: np.ndarray, freqs: np.ndarray, band: str, chunk: int = 16) -> np.ndarray:
    """
    Trial-averaged power of one frequency band, shape (n_times, n_channels).

    The mean over trials is accumulated in chunks, so the (trials, band freqs, times, channels)
    slice is never materialised in full (the steps tensors can be several GB).
    """
    low, high = FREQ_BANDS[band]
    band_mask = (freqs >= low) & (freqs <= high)
    if not band_mask.any():
        raise ValueError(f"No frequency of the multitaper grid falls inside the {band} band {low}-{high} Hz.")
    n_trials = mt_tensor.shape[0]
    acc = np.zeros(mt_tensor.shape[2:], dtype=np.float64)  # (n_times, n_channels)
    for start in range(0, n_trials, chunk):
        block = mt_tensor[start:start + chunk][:, band_mask]        # (chunk, n_band_freqs, T, C)
        acc += block.mean(axis=1, dtype=np.float64).sum(axis=0)
    return acc / n_trials

def plot_video_multitaper(
    subject: str,
    session: str,
    event_type: str,
    band: str = 'gamma',
    hand: str | None = 'contra',
    action_class: str | None = None,
    vlim: float | None = None,
    interval_ms: int = 50,
) -> None:
    """
    Interactive "video" of the band power across the whole epoch (-EPOCH_T_PRE ... +EPOCH_T_POST s,
    t=0 = contact), one frame per multitaper time sample.

    The 128 channels are drawn as four 8x4 grids (2x2 layout: top-left, top-right, bottom-left,
    bottom-right), one cell per electrode, placed according to utils.get_channel_geometry_mapping
    (same physical layout as probe_layout.json). Array a owns the channels k with k % 4 == a,
    as in utils.artifact_inspection.

    Plotted values: the values stored in the .npz tensor (already normalised in
    extract_multitaper_epochs), averaged first across the selected trials (one value per freq,
    time, channel) and then across the frequencies of `band`. No further normalisation or baseline
    subtraction is applied here; `vlim` only sets the colour scale. Below the grids, the mean of
    each array over its channels is plotted over time, with a cursor that follows the current frame.

    Controls: Play/Pause, a time slider and a band selector.

    Parameters
    ----------
    event_type : {'steps', 'grasp'}
        Which .npz to load.
    band : key of FREQ_BANDS, initial band shown (can be changed from the GUI).
    hand : {'contra', 'ipsi', None}, default 'contra'
        Keeps only trials whose Hand is contralateral / ipsilateral to the implanted
        hemisphere, or all trials if None.
    action_class : {None, 'steps', 'grasp_hook', 'grasp_floor'}
        Further restricts the trials, e.g. 'grasp_hook' when event_type='grasp'.
    vlim : float or None
        Symmetric colour limit (+/- vlim). If None, the 99th percentile of |mean power|
        of the displayed band is used, recomputed when the band changes.
    interval_ms : delay between frames during Play.
    """
    if band not in FREQ_BANDS:
        raise ValueError(f"Unknown band {band!r}, expected one of {list(FREQ_BANDS)}")

    try:
        multitaper_dict = load_multitaper_epochs(subject, session, event_type)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    mt_tensor = multitaper_dict['mt_tensor']
    trials = multitaper_dict['trials']
    freqs = multitaper_dict['freqs']
    time_vector = multitaper_dict['times_s']

    keep = np.ones(len(trials), dtype=bool)
    if hand is not None:
        keep &= trials['is_contralateral'].values if hand == 'contra' else ~trials['is_contralateral'].values
    if action_class is not None:
        keep &= (trials['action_class'] == action_class).values
    mt_tensor = mt_tensor[keep]

    num_trials, _, num_times, num_channels = mt_tensor.shape
    if num_trials == 0:
        print(f"No trial found for hand='{hand}', action_class='{action_class}'.")
        return
    if num_channels != 128:
        print(f"Expected 128 channels for the 2x2 array layout, found {num_channels}.")
        return

    # Physical layout: 4 arrays of 8x4 electrodes. grid_channels[a][r, c] = channel index.
    mapping = get_channel_geometry_mapping()  # (16, 8)
    array_names = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    grid_channels = []
    for a in range(4):
        row0, col0 = (a // 2) * 8, (a % 2) * 4
        grid = mapping[row0:row0 + 8, col0:col0 + 4]
        assert (grid % 4 == a).all(), "Channel mapping inconsistent with the k % 4 array convention"
        grid_channels.append(grid)

    sample_period = float(np.median(np.diff(time_vector)))
    zero_index = int(np.argmin(np.abs(time_vector)))

    # Band maps are computed lazily and cached: (n_times, n_channels) per band.
    band_maps = {}
    def get_map(band_name: str) -> np.ndarray:
        if band_name not in band_maps:
            band_maps[band_name] = _band_time_map(mt_tensor, freqs, band_name)
        return band_maps[band_name]

    def color_limit(band_name: str) -> float:
        if vlim is not None:
            return float(vlim)
        limit = float(np.percentile(np.abs(get_map(band_name)), 99))
        return limit if limit > 0 else 1.0

    state = {'band': band}
    power_map = get_map(band)
    limit = color_limit(band)

    # --- Figure: 2x2 electrode grids + mean-trace panel ---
    fig = plt.figure(figsize=(7, 7), layout='constrained')
    grid_spec = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.3])
    heat_axes = [fig.add_subplot(grid_spec[a // 2, a % 2]) for a in range(4)]
    trace_ax = fig.add_subplot(grid_spec[2, :])

    images = []
    for a, ax in enumerate(heat_axes):
        im = ax.imshow(
            power_map[0][grid_channels[a]], cmap='RdBu_r', vmin=-limit, vmax=limit,
            aspect='auto', interpolation='nearest'
        )
        images.append(im)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(images[0], ax=heat_axes, fraction=0.04, pad=0.02)
    cbar.set_label(f"Normalized {band.capitalize()} power (z)", fontsize=10)

    trace_lines = []
    array_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for a in range(4):
        line, = trace_ax.plot(time_vector, power_map[:, grid_channels[a].ravel()].mean(axis=1),
                              color=array_colors[a], linewidth=1.2, label=array_names[a])
        trace_lines.append(line)
    global_line, = trace_ax.plot(time_vector, power_map.mean(axis=1), color='black', linewidth=2.2, label='All 128')
    trace_ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
    cursor = trace_ax.axvline(time_vector[0], color='red', linewidth=1.5)
    trace_ax.set_xlim(time_vector[0], time_vector[-1])
    trace_ax.set_xlabel("Time [s]  (0 = contact)")
    trace_ax.set_ylabel("Mean power (z)")
    trace_ax.legend(ncol=5, fontsize=8, frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.0))

    def refresh_trace_limits() -> None:
        values = np.concatenate([l.get_ydata() for l in trace_lines + [global_line]])
        margin = 0.1 * (values.max() - values.min() + 1e-9)
        trace_ax.set_ylim(values.min() - margin, values.max() + margin)
    refresh_trace_limits()

    hand_str = f" | Hand: {hand}" if hand else " | All hands"
    class_str = f" | {action_class}" if action_class else ""
    suptitle = fig.suptitle("", fontsize=10, fontweight='bold')

    def draw_frame(time_idx: int) -> None:
        """Pure drawing step (no widget access), also used by FuncAnimation."""
        current_map = band_maps[state['band']]
        for a, im in enumerate(images):
            im.set_data(current_map[time_idx][grid_channels[a]])
        t = time_vector[time_idx]
        cursor.set_xdata([t, t])
        contact = "  ← contact" if time_idx == zero_index else ""
        suptitle.set_text(
            f"{subject} / {session} | {event_type}{class_str}{hand_str} | N={num_trials} | "
            f"{state['band'].capitalize()} | t = {t:+.3f} s{contact}"
        )

    def apply_band(band_name: str) -> None:
        state['band'] = band_name
        current_map = get_map(band_name)
        current_limit = color_limit(band_name)
        for im in images:
            im.set_clim(-current_limit, current_limit)
        for a, line in enumerate(trace_lines):
            line.set_ydata(current_map[:, grid_channels[a].ravel()].mean(axis=1))
        global_line.set_ydata(current_map.mean(axis=1))
        refresh_trace_limits()
        cbar.set_label(f"Normalized {band_name.capitalize()} power (z)", fontsize=10)

    # --- GUI widgets ---
    play = widgets.Play(min=0, max=num_times - 1, step=1, value=0, interval=interval_ms,
                        description="Play", show_repeat=True)
    slider = widgets.IntSlider(min=0, max=num_times - 1, step=1, value=0, description='Time:',
                               readout=False, continuous_update=True,
                               layout=widgets.Layout(width='55%'))
    time_label = widgets.Label(value="")
    band_selector = widgets.Dropdown(options=list(FREQ_BANDS.keys()), value=band, description='Band:',
                                     layout=widgets.Layout(width='160px'))
    widgets.jslink((play, 'value'), (slider, 'value'))

    def on_time_change(change=None) -> None:
        idx = int(slider.value)
        draw_frame(idx)
        time_label.value = f" t = {time_vector[idx]:+.3f} s  (frame {idx + 1}/{num_times})"
        fig.canvas.draw_idle()

    def on_band_change(change) -> None:
        time_label.value = f" Computing {change.new} map..."
        apply_band(change.new)
        on_time_change()

    slider.observe(on_time_change, names='value')
    band_selector.observe(on_band_change, names='value')

    display(widgets.VBox([widgets.HBox([play, slider, band_selector]), time_label]))
    on_time_change()
    plt.show()

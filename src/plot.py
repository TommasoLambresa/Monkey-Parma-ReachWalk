import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from src.io import load_multitaper_epochs
from src.config import EPOCH_T_POST, EPOCH_T_PRE, FREQ_BANDS

def plot_interactive_multitaper(subject: str, session: str, event_type: str, label_filter: str = None) -> None:
    """
    Renders an interactive Time-Frequency Representation (Spectrogram) of the Multitaper data.
    Uses a scrollable list for channel selection. Displays a grid comparing all unique labels.
    """
    
    try:
        multitaper_dict = load_multitaper_epochs(subject, session, event_type)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    mt_tensor = multitaper_dict['mt_tensor']  
    labels = multitaper_dict['labels']
    freqs = multitaper_dict['freqs']
    
    if label_filter is not None:
        filter_mask = np.array([label_filter in str(l) for l in labels])
        mt_tensor = mt_tensor[filter_mask]
        labels = labels[filter_mask]

    num_trials, num_freqs, num_times, num_channels = mt_tensor.shape
    
    time_vector = np.linspace(-EPOCH_T_PRE, EPOCH_T_POST, num_times)

    if num_trials == 0:
        print(f"No trial found for '{label_filter}'.")
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
            for band_lines in FREQ_BANDS.values():
                ax.axhline(y=band_lines[0], color='black', linestyle='--', linewidth=1.0)
                ax.axhline(y=band_lines[1], color='black', linestyle='--', linewidth=1.0)
            ax.set_ylim(0, 100)
            ax.set_xlim(-0.8, 0.5)
            # Dynamic title
            filter_str = f" | Filter: '{label_filter}'" if label_filter else " | All Trials"
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

def plot_population_heatmaps(subject: str, session: str, label_filter: str = None) -> None:
    """
    Generates 3 heatmaps (Steps, Grasp Hook, Grasp Floor) with independent sorting based on modulation.
    """
    
    # --- 1. DATA LOADING ---
    try:
        mt_dict_steps = load_multitaper_epochs(subject, session, 'steps')
        mt_dict_grasp = load_multitaper_epochs(subject, session, 'grasp')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    tensor_steps = mt_dict_steps['mt_tensor']
    labels_steps = mt_dict_steps['labels']
    
    tensor_grasp = mt_dict_grasp['mt_tensor']
    labels_grasp = mt_dict_grasp['labels']

    # Apply label filter if specified
    if label_filter is not None:
        filter_mask_steps = np.array([label_filter in str(l) for l in labels_steps])
        tensor_steps = tensor_steps[filter_mask_steps]
        labels_steps = labels_steps[filter_mask_steps]
        
        filter_mask_grasp = np.array([label_filter in str(l) for l in labels_grasp])
        tensor_grasp = tensor_grasp[filter_mask_grasp]
        labels_grasp = labels_grasp[filter_mask_grasp]

    mask_hook = np.array(['hook' in str(l).lower() for l in labels_grasp])
    mask_floor = np.array(['floor' in str(l).lower() for l in labels_grasp])
    
    tensor_hook = tensor_grasp[mask_hook]
    tensor_floor = tensor_grasp[mask_floor]
    
    freqs = mt_dict_steps['freqs']
    num_channels = tensor_steps.shape[3]
    num_times = tensor_steps.shape[2]
    
    time_vector = np.linspace(-EPOCH_T_PRE, EPOCH_T_POST, num_times)
    
    # Operational masks
    gamma_mask = (freqs >= FREQ_BANDS['gamma'][0]) & (freqs <= FREQ_BANDS['gamma'][1])
    time_mod_mask = (time_vector >= -0.8) & (time_vector <= 0.0)
    
    # Helper: Extracts Gamma power and computes across-trial mean -> Output: (Channels, Time)
    def get_gamma_mean(tensor):
        if tensor.shape[0] == 0: return np.zeros((num_channels, num_times))
        gamma_power = np.mean(tensor[:, gamma_mask, :, :], axis=1)
        return np.mean(gamma_power, axis=0).T 

    z_steps = get_gamma_mean(tensor_steps)
    z_hook = get_gamma_mean(tensor_hook)
    z_floor = get_gamma_mean(tensor_floor)
    
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    conditions = [
        ("Steps", z_norm_steps),
        ("Grasp Hook", z_norm_hook),
        ("Grasp Floor", z_norm_floor)
    ]
    
    for ax, (cond_name, z_norm_cond) in zip(axes, conditions):
        
        # Calculate Modulation Score (in the pre-event window [-0.8, 0.0])
        scores = np.mean(z_norm_cond[:, time_mod_mask], axis=1)
        
        # Classification 
        fac_idx = np.where(scores > 0)[0]
        sup_idx = np.where(scores <= 0)[0]
        
        # Independent sorting
        fac_idx_sorted = fac_idx[np.argsort(scores[fac_idx])[::-1]]     # From strong positive down to near zero
        sup_idx_sorted = sup_idx[np.argsort(np.abs(scores[sup_idx]))]   # From near zero down to strong negative
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
    cbar.set_label("Normalized Gamma Power (a.u.)", fontsize=11)
    
    plt.suptitle(f"{subject} / {session} | Population Gamma Power", fontsize=16, fontweight='bold')
    plt.show()
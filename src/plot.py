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


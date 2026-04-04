import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import ipywidgets as widgets
from IPython.display import display
from src.io import load_envelopes, load_epochs
from src.config import EPOCH_T_POST, EPOCH_T_PRE

def plot_interactive_envelopes(
    subject: str,
    session: str,
    fs: float = 100.0,
    time_window: Optional[Tuple[float, float]] = (0.0, 10.0)
) -> None:
    """
    Renders an interactive time-frequency heatmap of the envelopes.
    Uses a scrollable list for channel selection placed next to the plot.
    """
    envelopes = load_envelopes(subject, session)
    bands = list(envelopes.keys())
    num_channels = envelopes[bands[0]].shape[1]
    num_bands = len(bands)
    # Determine sample indices based on the requested time window
    if time_window is not None:
        start_sample = int(time_window[0] * fs)
        end_sample = int(time_window[1] * fs)
    else:
        start_sample = 0
        end_sample = envelopes[bands[0]].shape[0]
        
    # Add + 1 to end_sample to generate the rightmost boundary required by shading='flat'
    time_vector = np.arange(start_sample, end_sample) / fs
    
    # Create a scrollable list (Select widget) for channels
    channel_options = [(f"{i}", i) for i in range(num_channels)]
    channel_selector = widgets.Select(
        options=channel_options,
        value=0,
        description='Ch:',
        rows=20,  # Number of visible rows determines the height and scrollability
        layout=widgets.Layout(width='150px')
    )
    
    # Output widget to host the plot
    plot_output = widgets.Output()
    
    def update_plot(change) -> None:
        # Get the new channel index from the event or fallback to the current value
        channel_idx = change.new if change is not None else channel_selector.value
        
        with plot_output:
            plot_output.clear_output(wait=True)
            
            fig, axes = plt.subplots(nrows=num_bands, ncols=1, figsize=(10, num_bands), sharex=True)
            if num_bands == 1:
                axes = [axes]
            for ax, band in zip(axes, bands):
                # Extract temporal trace for the specific band and channel
                trace = envelopes[band][start_sample:end_sample, channel_idx]
                # Define Y-axis limits with a 10% margin
                ymin, ymax = np.min(trace), np.max(trace)
                margin = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
                ymin, ymax = ymin - margin, ymax + margin
                # Plot the colormap background reflecting the amplitude
                ax.imshow(
                    trace[np.newaxis, :], 
                    aspect='auto', 
                    cmap='viridis', 
                    extent=[time_vector[0], time_vector[-1], ymin, ymax],
                    alpha=0.7,  # Add transparency to make the line clearly visible
                    origin='lower'
                )
                
                # Plot the line plot
                ax.plot(time_vector, trace, color='black', linewidth=1.2)
                ax.set_ylim(ymin, ymax)
                ax.set_ylabel(f"{band}\n[$\\mu$V]")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            axes[-1].set_xlabel("Time [s]")
            axes[-1].set_xlim(time_vector[0], time_vector[-1])
            fig.suptitle(f"{subject} / {session} - Channel {channel_idx}", fontsize=14, y=0.98)
            fig.tight_layout()
            plt.show()
    # Bind the selector to the update function
    channel_selector.observe(update_plot, names='value')
    update_plot(None)
    display(widgets.HBox([channel_selector, plot_output]))

def plot_interactive_epochs(subject: str, session: str, event_type: str) -> None:
    """
    Renders an interactive plot of trial-averaged epochs across all frequency bands.
    Uses a scrollable list for channel selection placed next to the plot.
    """
    # Expected shape for each band: (num_trials, num_samples, num_channels)
    epochs_dict = load_epochs(subject, session, event_type)
    labels = epochs_dict.pop("labels")
    bands = list(epochs_dict.keys())
    unique_labels = np.unique(labels)
    
    num_trials, num_samples, num_channels = epochs_dict[bands[0]].shape
    time_vector = np.linspace(-EPOCH_T_PRE, EPOCH_T_POST, num_samples)
    
    # Create a scrollable list (Select widget) for channels
    channel_options = [(f"{i}", i) for i in range(num_channels)]
    channel_selector = widgets.Select(
        options=channel_options,
        value=0,
        description='Ch:',
        rows=20,
        layout=widgets.Layout(width='150px')
    )
    visibility_state = {}
    # Output widget to host the plot
    plot_output = widgets.Output()
    
    def update_plot(change) -> None:
        # Get the new channel index from the event or fallback to the current value
        channel_idx = change.new if change is not None else channel_selector.value
        
        with plot_output:
            plot_output.clear_output(wait=True)
            
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17, 8))
            axes = axes.flatten()                
            artists_dict = {}
            for i, ax in enumerate(axes):
                # Hide unused subplots if bands are less than 16
                if i >= len(bands):
                    ax.set_visible(False)
                    continue
                    
                band = bands[i]
                
                for col_idx, label in enumerate(unique_labels):
                    # Filter trials corresponding to the current label
                    trial_mask = (labels == label)
                    num_trials_label = np.sum(trial_mask)
                    
                    if num_trials_label == 0:
                        continue

                    legend_label = f"{label} (N={num_trials_label})"
                    if legend_label not in artists_dict:
                        artists_dict[legend_label] = []

                    # Initialize state if not present
                    if legend_label not in visibility_state:
                        visibility_state[legend_label] = True
                    is_visible = visibility_state[legend_label]

                    # Extract temporal traces for specific band, filtered trials, and specific channel
                    channel_data = epochs_dict[band][trial_mask, :, channel_idx]
                    
                    # Compute mean and standard error of the mean (SEM)
                    mean_trace = np.mean(channel_data, axis=0)
                    std_trace = np.std(channel_data, axis=0)
                    sem_trace = std_trace / np.sqrt(num_trials_label)
                    
                    # Plot mean line and shaded SEM area applying the persisted visibility
                    line, = ax.plot(time_vector, mean_trace, linewidth=1.5, label=legend_label, visible=is_visible)
                    fill = ax.fill_between(
                        time_vector, 
                        mean_trace - sem_trace, 
                        mean_trace + sem_trace, 
                        alpha=0.2, 
                        edgecolor='none',
                        visible=is_visible
                    )
                    
                    # Store artists to toggle visibility later
                    artists_dict[legend_label].extend([line, fill])
                
                # Mark event onset
                ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title(f"{band}")
                ax.set_ylabel("[$\\mu$V]")
                ax.set_xlabel("Time [s]")
                ax.set_xlim(time_vector[0], time_vector[-1])
                
            # Extract handles and labels from the first subplot
            handles, leg_labels = axes[0].get_legend_handles_labels()
            
            # Repurpose the last axis of the 4x4 grid for the legend
            legend_ax = axes[-1]
            legend_ax.set_visible(True)
            legend_ax.axis('off')
            leg = legend_ax.legend(
                handles, 
                leg_labels, 
                loc='center', 
                frameon=False, 
                fontsize='small', 
                ncol=1
            )
            
            # Make legend lines clickable
            for leg_line, label_key in zip(leg.get_lines(), leg_labels):
                leg_line.set_picker(True)
                leg_line.set_pickradius(5)
                leg_line._associated_artists = artists_dict[label_key]
                leg_line._label_key = label_key  # Store key for state update
                leg_line.set_alpha(1.0 if visibility_state[label_key] else 0.2)

            def on_pick(event) -> None:
                leg_line = event.artist
                label_key = leg_line._label_key
                
                # Toggle and save visibility state
                visibility_state[label_key] = not visibility_state[label_key]
                is_visible = visibility_state[label_key]
                
                # Update legend alpha to indicate state
                leg_line.set_alpha(1.0 if is_visible else 0.2)
                
                # Toggle visibility of associated plot lines and fills
                for artist in leg_line._associated_artists:
                    artist.set_visible(is_visible)
                    
                fig.canvas.draw_idle()

            # Connect the click event to the figure
            fig.canvas.mpl_connect('pick_event', on_pick)

            # Connect the click event to the figure
            fig.canvas.mpl_connect('pick_event', on_pick)          
            fig.suptitle(f"{subject} / {session} | Event: {event_type} - Channel {channel_idx}", fontsize=14, y=1.02)
            fig.tight_layout()
            plt.show()

    # Bind the selector to the update function
    channel_selector.observe(update_plot, names='value')
    update_plot(None)
    display(widgets.HBox([channel_selector, plot_output]))

def plot_spatiotemporal_video(subject: str, session: str, event_type: str, label_filter: str) -> None:
    """
    Renders an interactive player showing the spatiotemporal activation of 128 channels 
    (arranged in a 2x2 grid of 8x4 matrices) across 8 frequency bands.
    Trials are filtered by checking if `label_filter` is a substring of the trial label.
    """
    epochs_dict = load_epochs(subject, session, event_type)
    labels = epochs_dict.pop("labels")
    bands = list(epochs_dict.keys())
    
    # Filter trials based on substring
    trial_mask = np.array([label_filter in str(lbl) for lbl in labels])
    num_trials = np.sum(trial_mask)
    
    if num_trials == 0:
        print(f"No trial found for '{label_filter}'.")
        return
        
    num_samples = epochs_dict[bands[0]].shape[1]
    
    # Compute mean across filtered trials for each band
    # Expected shape after mean: (num_samples, 128)
    mean_data = {band: np.mean(epochs_dict[band][trial_mask, :, :], axis=0) for band in bands}
    
    # Determine global min and max across time for consistent colormapping per band
    vlims = {band: (np.percentile(mean_data[band], 10), np.percentile(mean_data[band], 90)) for band in bands}
    
    # Setup the figure
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    axes = axes.flatten()
    
    im_artists = []
    
    for i, ax in enumerate(axes):
        if i >= len(bands):
            ax.set_visible(False)
            continue
            
        band = bands[i]
        
        # Initialize an empty 16x8 matrix for the 4 arrays
        im = ax.imshow(
            np.zeros((16, 8)), 
            aspect='auto', 
            cmap='viridis', 
            vmin=vlims[band][0], 
            vmax=vlims[band][1], 
            origin='upper'
        )
        ax.set_title(f"{band}")
        ax.axis('off')
        
        # Draw separators for the 2x2 macro-grid
        ax.axhline(7.5, color='white', linewidth=3)
        ax.axvline(3.5, color='white', linewidth=3)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('[$\\mu$V]', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        im_artists.append((band, im))
        
    fig.suptitle(f"{subject} / {session} | Event: {event_type} | Filter: '{label_filter}' (N={num_trials})", fontsize=14)
    fig.tight_layout()
    
    # Create interactive widgets
    play = widgets.Play(
        value=0,
        min=0,
        max=num_samples - 1,
        step=1,
        interval=50, # Update interval in milliseconds (20 fps)
        description="Press play"
    )
    time_slider = widgets.IntSlider(min=0, max=num_samples - 1, step=1, description='Sample:')
    widgets.jslink((play, 'value'), (time_slider, 'value'))
    
    def update(change) -> None:
        t = change.new if change is not None else time_slider.value
        
        for band, im in im_artists:
            data_t = mean_data[band][t, :] # Shape: (128,)
            
            # Map the 128 channels into the 16x8 grid
            grid = np.zeros((16, 8))
            # Array 1: Top-Left
            grid[0:8, 0:4] = data_t[0:32].reshape(8, 4)
            # Array 2: Top-Right
            grid[0:8, 4:8] = data_t[32:64].reshape(8, 4)
            # Array 3: Bottom-Left
            grid[8:16, 0:4] = data_t[64:96].reshape(8, 4)
            # Array 4: Bottom-Right
            grid[8:16, 4:8] = data_t[96:128].reshape(8, 4)
            
            im.set_data(grid)
            
        fig.canvas.draw_idle()

    # Bind and display
    time_slider.observe(update, names='value')
    update(None) # Trigger first frame
    
    display(widgets.HBox([play, time_slider]))
    plt.show()
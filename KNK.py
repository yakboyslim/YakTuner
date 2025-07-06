"""
Knock (KNK) Analysis and Ignition Timing Correction Module

This module analyzes engine logs for knock events to recommend ignition timing
corrections. It processes log data to identify when and where knock occurs,
calculates a statistically-driven correction map, and displays the results
visually and in a table.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import tkinter as tk
from tkinter import simpledialog, Toplevel
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pandastable import Table

# Set the matplotlib backend for tkinter compatibility
matplotlib.use('TkAgg')


# --- Helper Classes ---

class ColoredTable(Table):
    """A pandastable Table subclass that colors cells based on correction value."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.rowselectedcolor = None  # Disable default row selection highlighting

    def color_cells(self, correction_array):
        """Color cells green for advance, red for retard, based on the correction values."""
        self.resetColors()
        if correction_array.shape != self.model.df.shape:
            print("Warning: Shape mismatch between correction array and table. Cannot color cells.")
            return

        for r in range(correction_array.shape[0]):
            for c in range(correction_array.shape[1]):
                value = correction_array[r, c]
                if value > 0:
                    self.setRowColors(rows=[r], cols=[c], clr='#90EE90')  # Light Green
                elif value < 0:
                    self.setRowColors(rows=[r], cols=[c], clr='#FFB6C1')  # Light Red
        self.redraw()


# --- Helper Functions ---

def _get_knock_parameters():
    """Shows dialogs to get user inputs for knock tuning parameters."""
    root = tk.Tk()
    root.withdraw()

    params = {
        'max_adv': float(simpledialog.askstring("Knock Inputs", "Maximum advance if no knock seen:", initialvalue="0.75")),
        'confidence': 1 - float(simpledialog.askstring("Knock Inputs", "Confidence Required to confirm knock:", initialvalue="0.75")),
        'map_num': simpledialog.askinteger("Select Map", "Select a SP Map 1-5 (0 is no SP Map):", minvalue=0, maxvalue=5)
    }
    return params


def _prepare_knock_data(log):
    """Creates derived columns and identifies knock events in the log data."""
    log['MAP'] = log['MAP'] * 10

    # --- Identify single cylinder knock outliers for statistical analysis ---
    all_cyl_knock = log[['KNK1', 'KNK2', 'KNK3', 'KNK4']].to_numpy()
    log['KNKAVG'] = np.mean(all_cyl_knock, axis=1)

    min_cyl_knock = np.min(all_cyl_knock, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores = stats.zscore(all_cyl_knock, axis=1)
        # Scale z-score by the ratio of the cylinder's knock to the minimum knock value.
        # This amplifies the score for the cylinder that is knocking most relative to others.
        outlier_scores = z_scores * all_cyl_knock / min_cyl_knock[:, np.newaxis]

    outlier_scores = np.nan_to_num(outlier_scores)
    rows_with_outlier = np.where(np.any(outlier_scores > 0, axis=1))[0]
    if len(rows_with_outlier) > 0:
        outlier_cyl_indices = np.argmax(outlier_scores[rows_with_outlier], axis=1)
        log.loc[rows_with_outlier, 'singlecyl'] = outlier_cyl_indices + 1
    log['singlecyl'] = log['singlecyl'].fillna(0)

    # --- Vectorized knock event detection ---
    # A knock event is detected when a knock counter decreases, signifying that the
    # ECU has retarded timing and reset the counter.
    knock_cols = ['KNK1', 'KNK2', 'KNK3', 'KNK4']
    knock_decreased = log[knock_cols].diff() < 0
    log['knkoccurred'] = knock_decreased.any(axis=1)

    # --- Identify source of knock for plotting ---
    # This is more direct than the z-score method for event-based plotting
    knock_events_mask = log['knkoccurred']
    num_knocking_cyls = knock_decreased[knock_events_mask].sum(axis=1)

    # Find the index of the single knocking cylinder (returns 0-3)
    single_knock_cyl_idx = np.argmax(knock_decreased[knock_events_mask].to_numpy(), axis=1)

    # Initialize source cylinder column
    log['knock_source_cyl'] = np.nan

    # Where 1 cylinder knocked, set it to cyl number (1-4)
    log.loc[knock_events_mask & (num_knocking_cyls == 1), 'knock_source_cyl'] = single_knock_cyl_idx[num_knocking_cyls == 1] + 1

    # Where >1 cylinder knocked, set it to 0 ("Multiple")
    log.loc[knock_events_mask & (num_knocking_cyls > 1), 'knock_source_cyl'] = 0

    return log


def _create_bins(log, igxaxis, igyaxis):
    """Discretizes log data into bins based on ignition map axes."""
    xedges = [0] + [(igxaxis[i] + igxaxis[i + 1]) / 2 for i in range(len(igxaxis) - 1)] + [float('inf')]
    yedges = [0] + [(igyaxis[i] + igyaxis[i + 1]) / 2 for i in range(len(igyaxis) - 1)] + [float('inf')]

    log['X'] = pd.cut(log['RPM'], bins=xedges, labels=False)
    log['Y'] = pd.cut(log['MAF'], bins=yedges, labels=False)
    return log


def _calculate_knock_correction(log, igxaxis, igyaxis, params):
    """Calculates the recommended ignition correction map based on knock data."""
    num_x, num_y = len(igxaxis), len(igyaxis)
    correction_map = np.zeros((num_y, num_x))

    for i in range(num_x):
        for j in range(num_y):
            cell_data = log[(log['X'] == i) & (log['Y'] == j)]
            if len(cell_data) > 3:
                knock_events = cell_data[cell_data['knkoccurred']]
                mean_knock_retard = knock_events['KNKAVG'].mean() if not knock_events.empty else 0.0

                mean_cell_knock, std_dev_cell_knock = cell_data['KNKAVG'].mean(), cell_data['KNKAVG'].std()
                low_ci, high_ci = norm.interval(params['confidence'], loc=mean_cell_knock, scale=std_dev_cell_knock if std_dev_cell_knock > 0 else 1e-9)

                if high_ci < 0:
                    correction_map[j, i] = (high_ci + mean_knock_retard) / 2
                elif high_ci > 0 and igxaxis[i] > 2500 and igyaxis[j] > 700:
                    if std_dev_cell_knock > 0:
                        z = (params['max_adv'] - mean_knock_retard) / std_dev_cell_knock
                        correction_map[j, i] = params['max_adv'] * (1 - norm.cdf(z))

    correction_map = np.nan_to_num(correction_map)
    correction_map = np.minimum(correction_map, params['max_adv'])

    # --- ECU Value Quantization ---
    # Rounds the correction to the nearest value the ECU can represent.
    # 5.333... = 16/3, 2.666... = 8/3.
    intermediate = np.ceil(correction_map * (16 / 3)) / (16 / 3)
    final_correction = np.round(intermediate * (8 / 3)) / (8 / 3)

    return final_correction


def _plot_knock_events(log, igxaxis, igyaxis):
    """Displays a scatter plot of the knock events."""
    knock_events = log[log['knkoccurred']].copy()
    if knock_events.empty:
        print("No knock events to plot.")
        return

    plt.figure(figsize=(12, 8))

    # --- Define custom colormap for 5 discrete categories ---
    # 0: Multiple, 1: Cyl 1, 2: Cyl 2, 3: Cyl 3, 4: Cyl 4
    colors = ['grey', 'red', 'blue', 'green', 'purple']
    cmap = ListedColormap(colors)

    # Define the boundaries for the color mapping.
    # For values 0,1,2,3,4 we need 6 boundaries from -0.5 to 4.5.
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    scatter = plt.scatter(
        knock_events['RPM'], knock_events['MAF'],
        s=abs(knock_events['KNKAVG']) * 100,
        c=knock_events['knock_source_cyl'],
        cmap=cmap,
        norm=norm
    )

    # --- Configure the colorbar with descriptive labels ---
    cbar = plt.colorbar(scatter, label='Knock Source')
    # Set the ticks to be in the middle of each color segment
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Multiple', 'Cyl 1', 'Cyl 2', 'Cyl 3', 'Cyl 4'])

    plt.gca().invert_yaxis()
    plt.xlabel('RPM')
    plt.ylabel('MAF')
    plt.title('Knock Events by Cylinder and Magnitude')
    plt.grid(True)
    plt.xticks(igxaxis, rotation=45)
    plt.yticks(igyaxis)
    plt.tight_layout()
    plt.show(block=True)


def _display_knock_results(result_df, correction_array):
    """Creates a Toplevel window to display the final results in a colored table."""
    window = Toplevel()
    window.title("SP IGN Correction Table")
    window.geometry("800x600")

    frame = tk.Frame(window)
    frame.pack(fill='both', expand=True, padx=5, pady=5)

    table = ColoredTable(frame, dataframe=result_df, showtoolbar=True, showstatusbar=True)
    table.editable = False
    table.show()
    table.color_cells(correction_array)

    window.mainloop()


# --- Main Function ---

def KNK(log, igxaxis, igyaxis, IGNmaps):
    """
    Analyzes engine logs for knock events and recommends ignition timing corrections.
    """
    # 1. Get user-defined parameters for the analysis.
    params = _get_knock_parameters()

    # 2. Select the base ignition map to apply corrections to.
    if params['map_num'] == 0:
        base_ignition_map = np.zeros((len(igyaxis), len(igxaxis)))
    else:
        base_ignition_map = IGNmaps[params['map_num'] - 1]

    # 3. Process log data to identify knock events and add derived values.
    log = _prepare_knock_data(log)

    # 4. Discretize log data into bins corresponding to the ignition map cells.
    log = _create_bins(log, igxaxis, igyaxis)

    # 5. Plot the detected knock events for visual inspection.
    _plot_knock_events(log, igxaxis, igyaxis)

    # 6. Calculate the recommended ignition timing correction map.
    correction_map = _calculate_knock_correction(log, igxaxis, igyaxis, params)

    # 7. Create the final recommended ignition map.
    recommended_map = base_ignition_map + correction_map

    # 8. Prepare results for display.
    xlabels = [str(x) for x in igxaxis]
    ylabels = [str(y) for y in igyaxis]
    result_df = pd.DataFrame(recommended_map, columns=xlabels, index=ylabels)

    # 9. Display the final table with colored cells indicating changes.
    _display_knock_results(result_df, correction_map)

    return result_df
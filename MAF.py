"""
Mass Airflow (MAF) Correction Tuning Module

This module analyzes engine logs to calculate and recommend adjustments to the
four primary MAF correction tables (IDX0-IDX3). It processes fuel trim data
to determine the required additive MAF correction, fits a 3D surface to this
data, and applies a confidence-based algorithm to generate new table values.

The results are presented in a 2x2 grid of tables and optional interactive
3D plots for detailed visual analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate
import tkinter as tk
from tkinter import simpledialog, messagebox, Toplevel, Frame
from pandastable import Table
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- Helper Classes ---

class ColoredTable(Table):
    """A pandastable Table subclass that colors cells based on correction value."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.rowselectedcolor = None  # Disable default row selection highlighting

    def color_cells(self, change_array):
        """Colors cells green for positive changes, red for negative."""
        self.resetColors()
        if change_array.shape != self.model.df.shape:
            return

        for r in range(change_array.shape[0]):
            for c in range(change_array.shape[1]):
                value = change_array[r, c]
                if value > 0.001:  # Use a small threshold to avoid coloring negligible changes
                    self.setRowColors(rows=[r], cols=[c], clr='#90EE90')  # Light Green
                elif value < -0.001:
                    self.setRowColors(rows=[r], cols=[c], clr='#FFB6C1')  # Light Red
        self.redraw()

# --- Helper Functions ---

def _get_maf_parameters():
    """Shows dialogs to get user inputs for MAF tuning."""
    params = {
        'confidence': 1 - float(simpledialog.askstring("MAF Inputs", "Confidence required to make change:", initialvalue="0.25")),
        'show_3d_plot': messagebox.askyesno(
            "3D Visualization",
            "Would you like to visualize the results in a 3D plot?\n(This can help in understanding the changes)"
        )
    }
    return params

def _prepare_maf_data(log, logvars):
    """Adds derived columns, filters log data, and warns about missing variables."""
    log['MAP'] = log['MAP'] * 10

    if "OILTEMP" in logvars:
        log = log[log['OILTEMP'] > 180]

    if 'LAM_DIF' not in logvars:
        log['LAM_DIF'] = 1/log['LAMBDA_SP'] - 1/log['LAMBDA']
        messagebox.showwarning('Recommendation', 'Recommend logging LAM DIF. Using calculated value, but may introduce inaccuracy.')

    log['LTFT'] = log["FAC_MFF_ADD"] if "FAC_MFF_ADD" in logvars else log['STFT']
    log['STFT'] = log['FAC_LAM_OUT'] if 'FAC_LAM_OUT' in logvars else log['LTFT']

    log['ADD_MAF'] = log['STFT'] + log['LTFT'] - log['LAM_DIF']

    if 'MAF_COR' in logvars:
        log['ADD_MAF'] = log['ADD_MAF'] + log['MAF_COR']
    else:
        messagebox.showwarning('Recommendation', 'Recommend logging MAF_COR for increased accuracy.')

    log = log[log['state_lam'] == 1]
    return log

def _create_bins(log, mafxaxis, mafyaxis):
    """Discretizes log data into bins based on MAF map axes."""
    xedges = [0] + [(mafxaxis[i] + mafxaxis[i + 1]) / 2 for i in range(len(mafxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mafyaxis[i] + mafyaxis[i + 1]) / 2 for i in range(len(mafyaxis) - 1)] + [np.inf]

    log['X'] = pd.cut(log['RPM'], bins=xedges, labels=False)
    log['Y'] = pd.cut(log['MAP'], bins=yedges, labels=False)
    return log

def _fit_surface_maf(log_data, mafxaxis, mafyaxis):
    """Fits a 3D surface to the MAF correction data using griddata."""
    if log_data.empty or len(log_data) < 3:
        return np.zeros((len(mafyaxis), len(mafxaxis)))

    points = log_data[['RPM', 'MAP']].values
    values = log_data['ADD_MAF'].values
    grid_x, grid_y = np.meshgrid(mafxaxis, mafyaxis)

    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    return np.nan_to_num(fitted_surface)

def _calculate_maf_correction(log_data, blend_surface, old_table, mafxaxis, mafyaxis, confidence):
    """Applies confidence interval logic to determine the final correction table."""
    new_table = old_table.copy()
    changed_mask = np.zeros_like(old_table, dtype=bool)
    max_count = 100.0
    interp_factor = 0.25

    for i in range(len(mafxaxis)):
        for j in range(len(mafyaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            count = len(cell_data)

            if count > 3:
                mean, std_dev = stats.norm.fit(cell_data['ADD_MAF'])
                low_ci, high_ci = stats.norm.interval(confidence, loc=mean, scale=std_dev if std_dev > 0 else 1e-9)

                current_val = old_table[j, i]
                change_made = False
                new_val = current_val

                if low_ci > current_val:
                    new_val = (blend_surface[j, i] * interp_factor + low_ci * (1 - interp_factor))
                    change_made = True
                elif high_ci < current_val:
                    new_val = (blend_surface[j, i] * interp_factor + high_ci * (1 - interp_factor))
                    change_made = True

                if change_made:
                    # Weight the change by the number of data points, capped at max_count
                    weight = min(count, max_count) / max_count
                    change_amount = (new_val - current_val) * weight
                    new_table[j, i] = current_val + change_amount
                    changed_mask[j, i] = True

    # Quantize the final table to the ECU's resolution (5.12 = 256 / 50)
    recommended_table = np.round(new_table * 5.12) / 5.12
    # Recalculate the final change amount after quantization
    final_change = recommended_table - old_table
    return recommended_table, final_change, changed_mask

def _plot_3d_maf_surface(title, mafxaxis, mafyaxis, old_map, new_map, log_data, changed_mask):
    """Creates an interactive 3D plot to visualize and compare MAF surfaces."""
    if log_data.empty:
        return

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(mafxaxis, mafyaxis)

    ax.scatter(log_data['RPM'], log_data['MAP'], log_data['ADD_MAF'], c='red', marker='o', label='Raw ADD_MAF Data', s=15)
    ax.plot_wireframe(X, Y, old_map, color='gray', alpha=0.7, label='Original Map')
    ax.plot_surface(X, Y, new_map, cmap='viridis', alpha=0.6, label='Recommended Map')

    changed_y_indices, changed_x_indices = np.where(changed_mask)
    if changed_y_indices.size > 0:
        x_coords = mafxaxis[changed_x_indices]
        y_coords = mafyaxis[changed_y_indices]
        z_coords = new_map[changed_y_indices, changed_x_indices] + 0.01  # Z-offset
        ax.scatter(x_coords, y_coords, z_coords, c='magenta', marker='X', s=60, label='Changed Cells', depthshade=False)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Engine Speed (RPM)', fontsize=12)
    ax.set_ylabel('Manifold Absolute Pressure (MAP)', fontsize=12)
    ax.set_zlabel('Additive MAF Correction', fontsize=12)
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Original Map'),
        Patch(facecolor=plt.cm.viridis(0.5), edgecolor='k', label='Recommended Map'),
        Line2D([0], [0], marker='o', color='w', label='Raw ADD_MAF Data', markerfacecolor='r', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Changed Cells', markerfacecolor='magenta', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show(block=True)

def _display_maf_results(results, changes):
    """Creates a Toplevel window with a 2x2 grid to display the MAF tables."""
    window = Toplevel()
    window.title("MAF Table Recommendations")

    frames = [Frame(window) for _ in range(4)]
    grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, frame in enumerate(frames):
        row, col = grid_positions[i]
        window.grid_rowconfigure(row, weight=1)
        window.grid_columnconfigure(col, weight=1)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        tk.Label(frame, text=f"IDX{i} Correction Table", font=("Arial", 10, "bold")).pack(pady=5)
        table_frame = Frame(frame)
        table_frame.pack(fill='both', expand=True)

        pt = ColoredTable(table_frame, dataframe=results[f"IDX{i}"], showtoolbar=True, showstatusbar=True)
        pt.editable = False
        pt.show()
        pt.color_cells(changes[f"IDX{i}"])

# --- Main Function ---

def MAF_tune(log, mafxaxis, mafyaxis, maftables, combmodes, logvars):
    """Main orchestrator for the MAF tuning process."""
    params = _get_maf_parameters()
    log = _prepare_maf_data(log, logvars)
    log = _create_bins(log, mafxaxis, mafyaxis)

    results = {}
    changes = {}

    for idx in range(4):
        current_table = maftables[idx]
        idx_modes = np.where(combmodes == idx)[0]
        log_filtered = log[log['CMB'].isin(idx_modes)].copy()
        log_filtered.dropna(subset=['RPM', 'MAP', 'ADD_MAF'], inplace=True)

        # Fit a 3D surface to the filtered data
        blend_surface = _fit_surface_maf(log_filtered, mafxaxis, mafyaxis)

        # Calculate the final recommended table based on confidence logic
        recommended_table, final_change, changed_mask = _calculate_maf_correction(
            log_filtered, blend_surface, current_table, mafxaxis, mafyaxis, params['confidence']
        )

        # Store results
        xlabels = [str(x) for x in mafxaxis]
        ylabels = [str(y) for y in mafyaxis]
        results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)
        changes[f'IDX{idx}'] = final_change

        # Optionally visualize the results in 3D
        if params['show_3d_plot']:
            _plot_3d_maf_surface(
                title=f"IDX{idx} MAF Correction (Changes Marked)",
                mafxaxis=mafxaxis,
                mafyaxis=mafyaxis,
                old_map=current_table,
                new_map=recommended_table,
                log_data=log_filtered,
                changed_mask=changed_mask
            )

    _display_maf_results(results, changes)
    return results
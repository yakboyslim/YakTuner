"""
Low-Pressure Fuel Pump (LPFP) PWM Tuning Module

This module analyzes engine logs to calculate and recommend adjustments to the
LPFP PWM control tables (lpfppwm and lpfppwm4wd). It processes log data for
the desired fuel pump duty cycle, fits a 3D surface to this data, and applies
a confidence-based algorithm to generate a new, optimized table.

The results are presented in a table and an optional interactive 3D plot for
detailed visual analysis.
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
from utils import ColoredTable, plot_3d_surface


# --- Helper Classes ---

class ColoredTable(Table):
    """A pandastable Table subclass that colors cells based on change value."""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.rowselectedcolor = None  # Disable default row selection highlighting

    def color_cells(self, new_array, old_array):
        """Colors cells green for higher values, red for lower."""
        self.resetColors()
        if new_array.shape != self.model.df.shape or old_array.shape != self.model.df.shape:
            return

        for r in range(new_array.shape[0]):
            for c in range(new_array.shape[1]):
                diff = new_array[r, c] - old_array[r, c]
                if diff > 0.1:  # Use a small threshold for PWM %
                    self.setRowColors(rows=[r], cols=[c], clr='#90EE90')  # Light Green
                elif diff < -0.1:
                    self.setRowColors(rows=[r], cols=[c], clr='#FFB6C1')  # Light Red
        self.redraw()


# --- Helper Functions ---

def _get_lpfp_parameters():
    """Shows dialogs to get user inputs for LPFP tuning."""
    params = {
        'confidence': 1 - float(simpledialog.askstring("LPFP Inputs", "Confidence required to make change:",
                                                      initialvalue="0.50")),
        'show_3d_plot': messagebox.askyesno(
            "3D Visualization",
            "Would you like to visualize the results in a 3D plot?"
        )
    }
    return params


def _prepare_lpfp_data(log, logvars):
    """
    Filters log data for relevant LPFP operating conditions.
    This version filters for data where the LPFP pressure is either close to the
    setpoint or holding steady.
    """
    # Add RPM and LPFP_FP to the required variables list.
    required_vars = ['FF_SP', 'LPFP_FP_SP', 'LPFP_PWM', 'RPM', 'LPFP_FP']
    for var in required_vars:
        if var not in logvars:
            messagebox.showerror("Missing Variable", f"The required log variable '{var}' was not found.")
            return pd.DataFrame()  # Return empty DataFrame if essential data is missing

    # --- Convert FF_SP from per-stroke to per-minute ---
    # This calculation is based on: [Fuel Flow SP] * 2 * [Engine Speed] / 1000
    log['FF_SP'] = log['FF_SP'] * 2 * log['RPM'] / 1000

    if "OILTEMP" in logvars:
        log = log[log['OILTEMP'] > 180]

    # Filter for rows where the pump is actively being controlled and in closed loop
    log = log[log['LPFP_PWM'] > 0].copy()
    log = log[log['state_lam'] == 1]
    log.dropna(subset=required_vars, inplace=True)

    # --- New Filtering Logic ---
    # Calculate the difference between actual and setpoint fuel pressure.
    log['LPFP_DELTA'] = log['LPFP_FP'] - log['LPFP_FP_SP']

    # Calculate the rate of change of this delta to determine stability.
    log['LPFP_DELTA_CHANGE'] = log['LPFP_DELTA'].diff().abs()

    # Define the conditions for keeping a data point:
    # 1. The delta is small (actual pressure is close to the target).
    is_small_delta = log['LPFP_DELTA'].abs() < 10
    # 2. The delta is steady (not changing rapidly).
    is_steady_delta = log['LPFP_DELTA_CHANGE'] < 1.0  # A threshold of 1 unit per timestep is a good starting point.

    # Combine the conditions. We want data that is EITHER small OR steady.
    # The first row will have a NaN for the change, so we fill it with False to exclude it.
    final_mask = is_small_delta | is_steady_delta
    log = log[final_mask.fillna(False)]

    if log.empty:
        messagebox.showwarning("LPFP Tune", "No data points met the criteria (small or steady LPFP delta).")

    return log


def _create_bins(log, xaxis, yaxis):
    """Discretizes log data into bins based on LPFP map axes."""
    xedges = [0] + [(xaxis[i] + xaxis[i + 1]) / 2 for i in range(len(xaxis) - 1)] + [np.inf]
    yedges = [0] + [(yaxis[i] + yaxis[i + 1]) / 2 for i in range(len(yaxis) - 1)] + [np.inf]

    log['X'] = pd.cut(log['FF_SP'], bins=xedges, labels=False)
    log['Y'] = pd.cut(log['LPFP_FP_SP'], bins=yedges, labels=False)
    return log


def _fit_surface_lpfp(log_data, xaxis, yaxis):
    """
    Fits a 3D surface to the LPFP PWM data using griddata.
    This version first aggregates the data to significantly improve performance
    while maintaining spatial accuracy.
    """
    if log_data.empty or len(log_data) < 3:
        return np.zeros((len(yaxis), len(xaxis)))

    # --- Performance/Accuracy Improvement: Aggregate data before fitting ---
    # Instead of fitting all raw points, fit the mean of each grid cell's
    # coordinates and value. This is fast and spatially accurate.
    agg_data = log_data.groupby(['X', 'Y']).agg({
        'FF_SP': 'mean',
        'LPFP_FP_SP': 'mean',
        'LPFP_PWM': 'mean'
    }).reset_index()

    if agg_data.empty:
        return np.zeros((len(yaxis), len(xaxis)))

    # Use the mean coordinates of the points within each cell for interpolation
    points = agg_data[['FF_SP', 'LPFP_FP_SP']].values
    values = agg_data['LPFP_PWM'].values

    grid_x, grid_y = np.meshgrid(xaxis, yaxis)

    # Interpolate using the much smaller set of aggregated points
    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill any remaining NaNs with the nearest value
    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        # Ensure we have valid points to use for nearest-neighbor fill
        if len(points) > 0:
            nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
            fitted_surface[nan_mask] = nearest_fill

    return np.nan_to_num(fitted_surface)


def _calculate_lpfp_correction(log_data, blend_surface, old_table, xaxis, yaxis, confidence):
    """Applies confidence interval logic to determine the final correction table."""
    new_table = old_table.copy()
    changed_mask = np.zeros_like(old_table, dtype=bool)

    for i in range(len(xaxis)):
        for j in range(len(yaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            count = len(cell_data)

            if count > 3:
                mean, std_dev = stats.norm.fit(cell_data['LPFP_PWM'])
                low_ci, high_ci = stats.norm.interval(confidence, loc=mean, scale=std_dev if std_dev > 0 else 1e-9)

                current_val = old_table[j, i]

                # Update if the old value is outside the confidence interval of the new data.
                if not (low_ci <= current_val <= high_ci):
                    # The new value is a simple blend of the fitted surface and the cell's mean
                    new_val = (blend_surface[j, i] + mean) / 2
                    new_table[j, i] = new_val
                    changed_mask[j, i] = True

    # Round the final table to a reasonable precision for PWM %
    recommended_table = np.round(new_table, 2)
    return recommended_table, changed_mask


def _plot_3d_lpfp_surface(title, xaxis, yaxis, old_map, new_map, log_data, changed_mask):
    """
    Creates an interactive 3D plot to visualize and compare LPFP surfaces.
    The raw data is aggregated by cell and shown as mean points with std dev error bars
    for improved performance and clarity.
    """
    if log_data.empty:
        return

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(xaxis, yaxis)

    # --- Data Aggregation for Performance ---
    # Group data by cell to calculate mean and std dev for plotting
    agg_data = log_data.groupby(['X', 'Y'])['LPFP_PWM'].agg(['mean', 'std']).reset_index().fillna(0)

    # Plot aggregated data points and their error bars
    for _, row in agg_data.iterrows():
        x_idx, y_idx = int(row['X']), int(row['Y'])
        if x_idx < len(xaxis) and y_idx < len(yaxis):
            x_coord = xaxis[x_idx]
            y_coord = yaxis[y_idx]
            mean_val = row['mean']
            std_val = row['std']

            # Plot the mean point
            ax.scatter(x_coord, y_coord, mean_val, c='red', marker='o', s=20)
            # Plot the vertical error bar for standard deviation
            ax.plot([x_coord, x_coord], [y_coord, y_coord], [mean_val - std_val, mean_val + std_val],
                    marker="_", color='red', alpha=0.8)

    # --- Original Plotting Logic (Surfaces and Markers) ---
    ax.plot_wireframe(X, Y, old_map, color='gray', alpha=0.7, label='Original Map')
    ax.plot_surface(X, Y, new_map, cmap='viridis', alpha=0.6, label='Recommended Map')

    changed_y_indices, changed_x_indices = np.where(changed_mask)
    if changed_y_indices.size > 0:
        x_coords = xaxis[changed_x_indices]
        y_coords = yaxis[changed_y_indices]
        z_coords = new_map[changed_y_indices, changed_x_indices] + 0.5  # Z-offset for visibility
        ax.scatter(x_coords, y_coords, z_coords, c='magenta', marker='X', s=60, label='Changed Cells',
                   depthshade=False)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Fuel Flow SP (Calculated per minute)', fontsize=12)
    ax.set_ylabel('LPFP Fuel Pressure SP (LPFP_FP_SP)', fontsize=12)
    ax.set_zlabel('LPFP Duty Cycle (%)', fontsize=12)
    ax.invert_yaxis()

    # --- Updated Legend ---
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Original Map'),
        Patch(facecolor=plt.cm.viridis(0.5), edgecolor='k', label='Recommended Map'),
        Line2D([0], [0], marker='o', color='w', label='Mean Log Data', markerfacecolor='r', markersize=8),
        Line2D([0], [0], marker='_', color='r', label='Std. Dev. of Log Data', markersize=8, markeredgewidth=2),
        Line2D([0], [0], marker='X', color='w', label='Changed Cells', markerfacecolor='magenta', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show(block=True)


def _display_lpfp_results(result_df, old_array):
    """Creates a Toplevel window to display the LPFP table."""
    window = Toplevel()
    window.title("LPFP PWM Table Recommendation")
    window.geometry("800x600")

    frame = Frame(window)
    frame.pack(fill='both', expand=True, padx=5, pady=5)

    tk.Label(frame, text="Recommended LPFP PWM Table", font=("Arial", 10, "bold")).pack(pady=5)
    table_frame = Frame(frame)
    table_frame.pack(fill='both', expand=True)

    pt = ColoredTable(table_frame, dataframe=result_df, showtoolbar=True, showstatusbar=True)
    pt.editable = False
    pt.show()
    pt.color_cells(result_df.to_numpy(), old_array)


# --- Main Function ---

def LPFP_tune(log, xaxis, yaxis, old_table, logvars):
    """Main orchestrator for the LPFP tuning process."""
    params = _get_lpfp_parameters()
    log = _prepare_lpfp_data(log, logvars)
    if log.empty:
        messagebox.showinfo("LPFP Tune", "No valid LPFP data found in logs. Skipping tune.")
        return None

    log = _create_bins(log, xaxis, yaxis)

    # Fit a 3D surface to the filtered data
    blend_surface = _fit_surface_lpfp(log, xaxis, yaxis)

    # Calculate the final recommended table based on confidence logic
    recommended_table, changed_mask = _calculate_lpfp_correction(
        log, blend_surface, old_table, xaxis, yaxis, params['confidence']
    )

    # Optionally visualize the results in 3D
    if params['show_3d_plot']:
        _plot_3d_lpfp_surface(
            title="LPFP PWM Correction (Changes Marked)",
            xaxis=xaxis,
            yaxis=yaxis,
            old_map=old_table,
            new_map=recommended_table,
            log_data=log,
            changed_mask=changed_mask
        )

    # Prepare and display results
    xlabels = [str(x) for x in xaxis]
    ylabels = [str(y) for y in yaxis]
    result_df = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    _display_lpfp_results(result_df, old_table)
    return result_df

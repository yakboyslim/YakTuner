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

# --- Helper Functions ---

def _get_lpfp_parameters():
    """Returns hardcoded parameters for LPFP tuning."""
    params = {
        'confidence': 0.7,
        'show_3d_plot': True
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

    points = log_data[['FF_SP', 'LPFP_FP_SP']].values
    values = log_data['LPFP_PWM'].values

    if len(points) < 4:
        print("LPFP Tuner Warning: Not enough valid data points in the log to create a surface. Skipping LPFP tuning.")
        return None  # Return None to signal that no surface could be created.

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

def _display_lpfp_results(result_df, old_array, parent):
    """Creates a Toplevel window to display the LPFP table."""
    window = Toplevel(parent)
    window.title("LPFP PWM Table Recommendation")
    window.geometry("800x600")

    frame = Frame(window)
    frame.pack(fill='both', expand=True, padx=5, pady=5)

    tk.Label(frame, text="Recommended LPFP PWM Table", font=("Arial", 10, "bold")).pack(pady=5)
    table_frame = Frame(frame)
    table_frame.pack(fill='both', expand=True)

    # This now uses the imported ColoredTable class
    pt = ColoredTable(table_frame, dataframe=result_df, showtoolbar=True, showstatusbar=True)
    pt.editable = False
    pt.show()
    pt.color_cells(result_df.to_numpy(), old_array)


# --- Main Function ---

def LPFP_tune(log, xaxis, yaxis, old_table, logvars, parent):
    """Main orchestrator for the LPFP tuning process."""
    print(" -> Initializing LPFP analysis...")
    params = _get_lpfp_parameters()

    print(" -> Preparing LPFP data from logs...")
    log = _prepare_lpfp_data(log, logvars)
    if log.empty:
        messagebox.showinfo("LPFP Tune", "No valid LPFP data found in logs. Skipping tune.")
        return None

    print(" -> Creating data bins from LPFP axes...")
    log = _create_bins(log, xaxis, yaxis)


    print(" -> Calculating correction map...")
    # Fit a 3D surface to the filtered data
    blend_surface = _fit_surface_lpfp(log, xaxis, yaxis)

    if blend_surface is None:
        return None

    # Calculate the final recommended table based on confidence logic
    recommended_table, changed_mask = _calculate_lpfp_correction(
        log, blend_surface, old_table, xaxis, yaxis, params['confidence']
    )

    # --- REFACTORED: Call the generic 3D plot function from utils.py ---
    if params['show_3d_plot']:
        print(" -> Plotting 3D surfaces for comparison...")
        plot_3d_surface(
            title="LPFP PWM Correction (Changes Marked)",
            xaxis=xaxis,
            yaxis=yaxis,
            old_map=old_table,
            new_map=recommended_table,
            log_data=log,
            changed_mask=changed_mask,
            x_label='Fuel Flow SP (Calculated per minute)',
            y_label='LPFP Fuel Pressure SP (LPFP_FP_SP)',
            z_label='LPFP Duty Cycle (%)',
            data_col_name='LPFP_PWM'
        )

    print(" -> Preparing final results as DataFrames...")
    # Prepare and display results
    xlabels = [str(x) for x in xaxis]
    ylabels = [str(y) for y in yaxis]
    result_df = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    print(" -> Displaying final results table...")
    _display_lpfp_results(result_df, old_table, parent)
    return result_df
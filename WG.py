"""
Wastegate (WG) Tuning Module for YAKtuner

This module analyzes engine log data to recommend adjustments to the wastegate base tables.
Key functionalities include:
- Filtering log data to isolate relevant operating conditions.
- Calculating the required wastegate duty cycle ('WGNEED') to correct for boost errors.
- Fitting a 3D surface to the scattered log data points (EFF, IFF, WGNEED) for both
  VVL0 and VVL1 camshaft profiles using scipy's griddata for robust interpolation.
- Applying a confidence interval to determine whether the newly fitted values are a
  statistically significant improvement over the existing table values.
- Calculating a recommended temperature compensation slope for the WG base table.
- Displaying the results in an interactive, color-coded table for user review.
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

# --- Helper Functions ---

def _get_wg_tuning_parameters():
    """Returns parameters for WG tuning, some hardcoded, some from user input."""
    # The root window creation is unnecessary and can cause issues.
    # simpledialog will attach to the application's main window.
    params = {
        'fudge': 0.71,  # Hardcoded as per request
        'minboost': 0,
        'show_3d_plot': True  # Hardcoded as per request
    }
    return params

def _prepare_and_filter_log(log, params, logvars, WGlogic, tempcomp, tempcompaxis):
    """Adds derived columns and filters the log data to isolate relevant tuning conditions."""
    # Determine SWG/FF logic and apply temperature correction
    if WGlogic:
        log['EFF'] = log['RPM']
        log['IFF'] = log['PUTSP'] * 10
        interp_func = interpolate.interp1d(tempcompaxis, tempcomp, kind='linear', fill_value='extrapolate')
        tempcorr = interp_func(log['AMBTEMP'])
    else:
        tempcorr = 0

    # Create derived values for analysis
    log['deltaPUT'] = log['PUT'] - log['PUTSP']
    log['WGNEED_uncorrected'] = log['WG_Final'] - log['deltaPUT'] * params['fudge']
    log['WGNEED'] = log['WGNEED_uncorrected'] - tempcorr

    # Filter log data to valid conditions
    if 'I_INH' in logvars:
        log = log[log['I_INH'] <= 0]
    else:
        minpedal = float(simpledialog.askstring("WG Inputs", "Recommend logging PUT I Inhibit.\nChoose minimum pedal to use:", initialvalue="50"))
        log = log[log['Pedal'] >= minpedal]

    if 'DV' in logvars:
        log = log[log['DV'] <= 50]
    else:
        messagebox.showwarning('Recommendation', 'Recommend logging DV position. Otherwise, DV may impact accuracy.')

    if 'BOOST' in logvars:
        log = log[log['BOOST'] >= params['minboost']]
    else:
        messagebox.showwarning('Recommendation', 'Recommend logging boost. Otherwise, logs are not trimmed for min boost.')

    # --- New Filtering Logic for PUT Delta ---
    # Calculate the rate of change of deltaPUT to determine stability.
    log['deltaPUT_CHANGE'] = log['deltaPUT'].diff().abs()

    # Define the conditions for keeping a data point:
    # 1. The delta is small (actual pressure is close to the target).
    is_small_delta = log['deltaPUT'].abs() < 10
    # 2. The delta is steady (not changing rapidly).
    is_steady_delta = log['deltaPUT_CHANGE'] < 1.0  # A threshold of 1 kPa per timestep is a good starting point.

    # Combine the conditions. We want data that is EITHER small OR steady.
    # The first row will have a NaN for the change, so we fill it with False to exclude it.
    final_mask = is_small_delta | is_steady_delta
    log = log[final_mask.fillna(False)]

    if log.empty:
        messagebox.showwarning("WG Tune", "No data points met the criteria (small or steady PUT delta).")

    # Final filter for WG duty cycle range
    log = log[log['WG_Final'] <= 98]
    return log

def _create_bins_and_labels(log_df, wgxaxis, wgyaxis):
    """Creates bin edges from axes and assigns each log entry to a grid cell (X, Y)."""
    wgxedges = np.zeros(len(wgxaxis) + 1)
    wgxedges[0] = wgxaxis[0]
    wgxedges[-1] = wgxaxis[-1] + 2
    for i in range(len(wgxaxis) - 1):
        wgxedges[i + 1] = (wgxaxis[i] + wgxaxis[i + 1]) / 2

    wgyedges = np.zeros(len(wgyaxis) + 1)
    wgyedges[0] = wgyaxis[0]
    wgyedges[-1] = wgyaxis[-1] + 2
    for i in range(len(wgyaxis) - 1):
        wgyedges[i + 1] = (wgyaxis[i] + wgyaxis[i + 1]) / 2

    log_df['X'] = pd.cut(log_df['EFF'], wgxedges, labels=False)
    log_df['Y'] = pd.cut(log_df['IFF'], wgyedges, labels=False)
    return log_df

def _plot_wg_data(log_VVL0, log_VVL1, wgxaxis, wgyaxis, WGlogic):
    """Displays a scatter plot of the filtered log data."""
    plt.figure(figsize=(12, 8))
    plt.scatter(log_VVL1['EFF'], log_VVL1['IFF'], s=abs(log_VVL1['WGNEED']), c=log_VVL1['deltaPUT'], marker='x', cmap='RdBu', label='VVL1')
    plt.scatter(log_VVL0['EFF'], log_VVL0['IFF'], s=abs(log_VVL0['WGNEED']), c=log_VVL0['deltaPUT'], marker='o', cmap='RdBu', label='VVL0', alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label('PUT - PUT SP (kPa)')
    plt.gca().invert_yaxis()
    if WGlogic:
        plt.xlabel('RPM')
        plt.ylabel('PUT SP')
    else:
        plt.xlabel('Engine Efficiency (EFF)')
        plt.ylabel('Intake Flow Factor (IFF)')
    plt.title('Wastegate Duty Cycle Need vs. Operating Point')
    plt.grid(True)
    plt.xticks(wgxaxis, rotation=45)
    plt.yticks(wgyaxis)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

def _fit_surface(log_data, wgxaxis, wgyaxis):
    """
    Fits a 3D surface to the provided log data using scipy.interpolate.griddata.
    This replaces the previous 1D row/column pwlf fitting method.
    """
    if log_data.empty or len(log_data) < 3:
        return np.zeros((len(wgyaxis), len(wgxaxis)))

    points = log_data[['EFF', 'IFF']].values
    values = log_data['WGNEED'].values
    grid_x, grid_y = np.meshgrid(wgxaxis, wgyaxis)

    # Interpolate using a linear method, which is analogous to piecewise linear.
    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill any NaNs (points outside the convex hull of the data) using nearest neighbor.
    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    # If all points were outside the hull, return a zeroed array.
    if np.all(np.isnan(fitted_surface)):
        return np.zeros((len(wgyaxis), len(wgxaxis)))

    # Return the surface scaled to a 0-1 range, consistent with the old tables.
    return fitted_surface / 100.0

def _calculate_final_recommendations(log_data, blend, old_table, wgxaxis, wgyaxis, calculate_temp_coef=False):
    """
    Calculates the final recommended WG table by comparing the new fit with the old
    table and applying confidence intervals to each cell.
    """
    final_table = old_table.copy()
    total_coef, count_coef = 0, 0

    for i in range(len(wgxaxis)):
        for j in range(len(wgyaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            count = len(cell_data)

            # Only update cells with sufficient data points.
            if count > 3:
                # Fit a normal distribution to the WGNEED data in the cell.
                mean, std_dev = stats.norm.fit(cell_data['WGNEED'])
                # Get a 70% confidence interval (hardcoded).
                low_ci, high_ci = stats.norm.interval(0.7, loc=mean, scale=std_dev)

                # The old table values are fractions; scale them to compare with WGNEED.
                current_val_scaled = old_table[j, i] * 100.0

                # Decide whether to update the cell.
                # Update if the old value is outside the confidence interval of the new data.
                if np.isnan(current_val_scaled) or not (low_ci <= current_val_scaled <= high_ci):
                    # Average the fitted surface value with the cell's mean value.
                    new_val = (blend[j, i] + (mean / 100.0)) / 2
                    final_table[j, i] = new_val
                # Otherwise, keep the old value.

            # Optional: Calculate temperature compensation coefficient for VVL0.
            if calculate_temp_coef:
                temp_log = cell_data[cell_data['BOOST'] >= 8]
                if len(temp_log) > 2 and (temp_log['AMBTEMP'].max() > temp_log['AMBTEMP'].min() + 15):
                    coef = np.polyfit(temp_log['AMBTEMP'], temp_log['WGNEED_uncorrected'] / 100, 1)[0]
                    total_coef += coef * len(temp_log)
                    count_coef += len(temp_log)

    # Round the final table to the resolution of the ECU.
    final_table = np.round(final_table * 16384) / 16384

    if calculate_temp_coef:
        avg_coef = total_coef / count_coef if count_coef > 0 else "Not enough data"
        return final_table, avg_coef
    else:
        return final_table

def _display_results_table(Res_1, Res_0, oldWG1, oldWG0, temp_comp_results_df, parent):
    """Creates a Toplevel window to display the final results in colored tables."""
    W1 = Toplevel(parent)
    W1.title("WG Table Recommendations")
    W1.minsize(600, 800)

    # --- VVL1 Table ---
    vvl1_main_frame = Frame(W1)
    vvl1_main_frame.pack(fill='both', expand=True, padx=5, pady=5)
    Label1 = tk.Label(vvl1_main_frame, text="VVL1 Recommended Table (Green=Higher, Red=Lower)", font=("Arial", 10, "bold"))
    Label1.pack(side='top', fill='x', pady=(0, 5))
    table_frame1 = Frame(vvl1_main_frame)
    table_frame1.pack(fill='both', expand=True)

    # This now uses the imported ColoredTable class
    pt1 = ColoredTable(table_frame1, dataframe=Res_1, showtoolbar=True, showstatusbar=True)
    pt1.editable = False
    pt1.show()
    pt1.color_cells(Res_1.to_numpy(), oldWG1)

    # --- VVL0 Table ---
    vvl0_main_frame = Frame(W1)
    vvl0_main_frame.pack(fill='both', expand=True, padx=5, pady=5)
    Label0 = tk.Label(vvl0_main_frame, text="VVL0 Recommended Table", font=("Arial", 10, "bold"))
    Label0.pack(side='top', fill='x', pady=(0, 5))
    table_frame0 = Frame(vvl0_main_frame)
    table_frame0.pack(fill='both', expand=True)

    # This now uses the imported ColoredTable class
    pt0 = ColoredTable(table_frame0, dataframe=Res_0, showtoolbar=True, showstatusbar=True)
    pt0.editable = False
    pt0.show()
    pt0.color_cells(Res_0.to_numpy(), oldWG0)

    # --- Temperature Compensation Table ---
    # This new section displays the recommended temp comp table if one was generated.
    if temp_comp_results_df is not None:
        temp_comp_frame = Frame(W1)
        temp_comp_frame.pack(fill='both', expand=True, padx=5, pady=5)

        temp_label = tk.Label(temp_comp_frame, text="Recommended Temperature Compensation", font=("Arial", 10, "bold"))
        temp_label.pack(side='top', fill='x', pady=(0, 5))

        table_frame_temp = Frame(temp_comp_frame)
        table_frame_temp.pack(fill='both', expand=True)

        # Use a standard pandastable, as coloring isn't needed here.
        pt_temp = Table(table_frame_temp, dataframe=temp_comp_results_df, showtoolbar=True, showstatusbar=True)
        pt_temp.editable = False
        pt_temp.show()
    else:
        # If no recommendation could be made, inform the user.
        messagebox.showinfo(
            "Temperature Compensation",
            "Not enough data to calculate a recommended temperature compensation slope.",
            parent=W1
        )

#Main Function

def WG_tune(log, wgxaxis, wgyaxis, oldWG0, oldWG1, logvars, plot, WGlogic, tempcomp, tempcompaxis, parent):
    """
    Main orchestrator for the WG tuning process.
    """
    print(" -> Initializing WG analysis...")
    params = _get_wg_tuning_parameters()

    print(" -> Preparing and filtering log data...")
    log = _prepare_and_filter_log(log, params, logvars, WGlogic, tempcomp, tempcompaxis)

    print(" -> Creating data bins from WG axes...")
    log = _create_bins_and_labels(log, wgxaxis, wgyaxis)

    print(" -> Separating data for VVL0 and VVL1...")
    log_VVL1 = log[log['VVL'] == 1].copy()
    log_VVL0 = log[log['VVL'] == 0].copy()

    if plot:
        print(" -> Plotting raw WG data for visual inspection...")
        _plot_wg_data(log_VVL0, log_VVL1, wgxaxis, wgyaxis, WGlogic)

    print(" -> Fitting 3D surface for VVL1...")
    blend1 = _fit_surface(log_VVL1, wgxaxis, wgyaxis)

    print(" -> Fitting 3D surface for VVL0...")
    blend0 = _fit_surface(log_VVL0, wgxaxis, wgyaxis)

    print(" -> Calculating final recommendations for VVL1...")
    final_table_1 = _calculate_final_recommendations(log_VVL1, blend1, oldWG1, wgxaxis, wgyaxis)

    print(" -> Calculating final recommendations for VVL0 and Temp Comp...")
    final_table_0, avg_coef = _calculate_final_recommendations(log_VVL0, blend0, oldWG0, wgxaxis, wgyaxis, calculate_temp_coef=True)

    if params['show_3d_plot']:
        print(" -> Plotting 3D surfaces for comparison...")
        changed_mask_1 = final_table_1 != oldWG1
        changed_mask_0 = final_table_0 != oldWG0

        if WGlogic:
            x_label, y_label = 'Engine Speed (RPM)', 'PUT SP (Axis scaled by 10)'
        else:
            x_label, y_label = 'Engine Efficiency (EFF)', 'Intake Flow Factor (IFF)'

        plot_3d_surface(
            title="VVL1 3D Comparison (Changes Marked)",
            xaxis=wgxaxis, yaxis=wgyaxis,
            old_map=oldWG1 * 100.0,
            new_map=final_table_1 * 100.0,
            log_data=log_VVL1, changed_mask=changed_mask_1,
            x_label=x_label, y_label=y_label,
            z_label='Wastegate Duty Cycle (%)',
            data_col_name='WGNEED'
        )
        plot_3d_surface(
            title="VVL0 3D Comparison (Changes Marked)",
            xaxis=wgxaxis, yaxis=wgyaxis,
            old_map=oldWG0 * 100.0,
            new_map=final_table_0 * 100.0,
            log_data=log_VVL0, changed_mask=changed_mask_0,
            x_label=x_label, y_label=y_label,
            z_label='Wastegate Duty Cycle (%)',
            data_col_name='WGNEED'
        )

    print(" -> Preparing temperature compensation results...")
    temp_comp_results_df = None
    if isinstance(avg_coef, (int, float)):
        _slope, original_intercept = np.polyfit(tempcompaxis, tempcomp, 1)
        new_tempcomp = (avg_coef * tempcompaxis) + original_intercept
        temp_df = pd.DataFrame({
            'Temperature': tempcompaxis,
            'Original Comp': tempcomp,
            'Recommended Comp': new_tempcomp
        })
        temp_comp_results_df = temp_df.set_index('Temperature').T.round(4)

    print(" -> Preparing final results as DataFrames...")
    exhlabels = [str(x) for x in wgxaxis]
    intlabels = [str(x) for x in wgyaxis]
    Res_1 = pd.DataFrame(final_table_1, columns=exhlabels, index=intlabels)
    Res_0 = pd.DataFrame(final_table_0, columns=exhlabels, index=intlabels)

    print(" -> Displaying final results tables...")
    _display_results_table(Res_1, Res_0, oldWG1, oldWG0, temp_comp_results_df, parent)

    return Res_1, Res_0
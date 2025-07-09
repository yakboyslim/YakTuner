"""
Multiplicative Fuel Factor (MFF) Tuning Module

This module analyzes engine logs to calculate and recommend adjustments to the
five primary MFF tables (IDX0-IDX4). It processes fuel trim data
to determine the required multiplicative fuel factor, fits a 3D surface to this
data, and applies a confidence-based algorithm to generate new table values.

The results are presented in a 2x3 grid of tables and optional interactive
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
from utils import ColoredTable, plot_3d_surface

# --- Helper Classes ---

# --- Helper Functions ---

def _get_mff_parameters():
    """Returns hardcoded parameters for MFF tuning."""
    params = {
        'confidence': 0.7,
        'show_3d_plot': True
    }
    return params

def _prepare_mff_data(log, logvars):
    """Adds derived columns, filters log data, and warns about missing variables."""
    if "OILTEMP" in logvars:
        log = log[log['OILTEMP'] > 180].copy()

    # Use .loc to ensure we are modifying the DataFrame directly, not a copy.
    if 'LAM_DIF' not in logvars:
        log.loc[:, 'LAM_DIF'] = 1/log['LAMBDA_SP'] - 1/log['LAMBDA']
        messagebox.showwarning('Recommendation', 'Recommend logging LAM DIF. Using calculated value, but may introduce inaccuracy.')

    if "FAC_MFF_ADD" in logvars:
        log.loc[:, 'final_ltft'] = log["FAC_MFF_ADD"]
    else:
        log.loc[:, 'final_ltft'] = log['STFT']

    if 'FAC_LAM_OUT' in logvars:
        log.loc[:, 'final_stft'] = log['FAC_LAM_OUT']
    else:
        log.loc[:, 'final_stft'] = log['STFT']

    # --- Core Calculation Change: Create a MULTIPLICATIVE factor ---
    additive_correction = (log['final_stft'] + log['final_ltft'])/100 - log['LAM_DIF']
    log.loc[:, 'MFF_FACTOR'] = 1.0 + additive_correction

    if 'MFF_COR' in logvars:
        log.loc[:, 'MFF_FACTOR'] = additive_correction + log['MFF_COR']
    else:
        messagebox.showwarning('Recommendation', 'Recommend logging MFF_COR for increased accuracy.')

    log = log[log['state_lam'] == 1].copy()
    log = log.drop(columns=['final_ltft', 'final_stft'], errors='ignore')
    return log

def _create_bins(log, mffxaxis, mffyaxis):
    """Discretizes log data into bins based on MFF map axes."""
    xedges = [0] + [(mffxaxis[i] + mffxaxis[i + 1]) / 2 for i in range(len(mffxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mffyaxis[i] + mffyaxis[i + 1]) / 2 for i in range(len(mffyaxis) - 1)] + [np.inf]

    # Also, use .loc to assign new columns safely and avoid warnings.
    log.loc[:, 'X'] = pd.cut(log['RPM'], bins=xedges, labels=False, duplicates='drop')
    log.loc[:, 'Y'] = pd.cut(log['MAF'], bins=yedges, labels=False, duplicates='drop') # Use Airmass (MAF) for Y-axis
    return log

def _fit_surface_mff(log_data, mffxaxis, mffyaxis):
    """Fits a 3D surface to the MFF correction data using griddata."""
    if log_data.empty or len(log_data) < 3:
        return np.ones((len(mffyaxis), len(mffxaxis))) # Default to 1.0 for multiplicative factor

    points = log_data[['RPM', 'MAF']].values
    values = log_data['MFF_FACTOR'].values
    grid_x, grid_y = np.meshgrid(mffxaxis, mffyaxis)

    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    return np.nan_to_num(fitted_surface, nan=1.0) # Fill any remaining NaNs with 1.0

def _calculate_mff_correction(log_data, blend_surface, old_table, mffxaxis, mffyaxis, confidence):
    """Applies confidence interval logic to determine the final correction table."""
    new_table = old_table.copy()
    changed_mask = np.zeros_like(old_table, dtype=bool)
    max_count = 100.0
    interp_factor = 0.25

    for i in range(len(mffxaxis)):
        for j in range(len(mffyaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            count = len(cell_data)

            if count > 3:
                mean, std_dev = stats.norm.fit(cell_data['MFF_FACTOR'])
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

    # Quantize the final table to a common ECU resolution for multiplicative factors (1/1024)
    recommended_table = np.round(new_table * 1024) / 1024
    # Recalculate the final change amount after quantization
    final_change = recommended_table - old_table
    return recommended_table, final_change, changed_mask

def _display_mff_results(results, mfftables, parent):
    """Creates a Toplevel window with a 2x3 grid to display the 5 MFF tables."""
    window = Toplevel(parent)
    window.title("MFF Table Recommendations")

    # Use a 2x3 grid for 5 tables
    frames = [Frame(window) for _ in range(5)]
    grid_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

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
        pt.color_cells(results[f"IDX{i}"].to_numpy(), mfftables[i])

# --- Main Function ---

def MFF_tune(log, mffxaxis, mffyaxis, mfftables, combmodes_MFF, logvars, parent):
    """Main orchestrator for the MFF tuning process."""
    print(" -> Initializing MFF analysis...")
    params = _get_mff_parameters()

    print(" -> Preparing MFF data from logs...")
    log = _prepare_mff_data(log, logvars)

    print(" -> Creating data bins from MFF axes...")
    log = _create_bins(log, mffxaxis, mffyaxis)

    results = {}
    # Loop through all 5 MFF tables
    for idx in range(5):
        print(f" -> Processing MFF Table IDX{idx}...")
        current_table = mfftables[idx]
        idx_modes = np.where(combmodes_MFF == idx)[0]
        log_filtered = log[log['CMB'].isin(idx_modes)].copy()
        log_filtered.dropna(subset=['RPM', 'MAF', 'MFF_FACTOR'], inplace=True)

        print(f"   -> Fitting 3D surface for IDX{idx}...")
        blend_surface = _fit_surface_mff(log_filtered, mffxaxis, mffyaxis)

        print(f"   -> Calculating correction map for IDX{idx}...")
        recommended_table, final_change, changed_mask = _calculate_mff_correction(
            log_filtered, blend_surface, current_table, mffxaxis, mffyaxis, params['confidence']
        )

        # Store results
        xlabels = [str(x) for x in mffxaxis]
        ylabels = [str(y) for y in mffyaxis]
        results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

        if params['show_3d_plot']:
            print(f"   -> Plotting 3D surface for IDX{idx}...")
            plot_3d_surface(
                title=f"IDX{idx} MFF Correction (Changes Marked)",
                xaxis=mffxaxis,
                yaxis=mffyaxis,
                old_map=current_table,
                new_map=recommended_table,
                log_data=log_filtered,
                changed_mask=changed_mask,
                x_label='Engine Speed (RPM)',
                y_label='Airmass (MAF)',
                z_label='Multiplicative Fuel Factor',
                data_col_name='MFF_FACTOR'
            )

    print(" -> Displaying final results tables...")
    _display_mff_results(results, mfftables, parent)
    return results
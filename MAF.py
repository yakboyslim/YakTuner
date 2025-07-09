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
from utils import ColoredTable, plot_3d_surface

# --- Helper Classes ---

# --- Helper Functions ---

def _get_maf_parameters():
    """Returns hardcoded parameters for MAF tuning."""
    params = {
        'confidence': 0.7,
        'show_3d_plot': True
    }
    return params

def _prepare_maf_data(log, logvars):
    """Adds derived columns, filters log data, and warns about missing variables."""
    # Use .loc for direct assignment
    log.loc[:, 'MAP'] = log['MAP'] * 10

    if "OILTEMP" in logvars:
        log = log[log['OILTEMP'] > 180].copy()

    if 'LAM_DIF' not in logvars:
        log.loc[:, 'LAM_DIF'] = 1/log['LAMBDA_SP'] - 1/log['LAMBDA']
        messagebox.showwarning('Recommendation', 'Recommend logging LAM DIF. Using calculated value, but may introduce inaccuracy.')

    # This logic is sound, but we'll assign the final result with .loc
    if "FAC_MFF_ADD" in logvars:
        final_ltft = log["FAC_MFF_ADD"]
    else:
        final_ltft = log['LTFT']

    if 'FAC_LAM_OUT' in logvars:
        final_stft = log['FAC_LAM_OUT']
    else:
        final_stft = log['STFT']

    log.loc[:, 'ADD_MAF'] = final_stft + final_ltft - log['LAM_DIF']

    if 'MAF_COR' in logvars:
        log.loc[:, 'ADD_MAF'] = log['ADD_MAF'] + log['MAF_COR']
    else:
        messagebox.showwarning('Recommendation', 'Recommend logging MAF_COR for increased accuracy.')

    log = log[log['state_lam'] == 1].copy()
    return log


def _create_bins(log, mafxaxis, mafyaxis):
    """Discretizes log data into bins based on MAF map axes."""
    xedges = [0] + [(mafxaxis[i] + mafxaxis[i + 1]) / 2 for i in range(len(mafxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mafyaxis[i] + mafyaxis[i + 1]) / 2 for i in range(len(mafyaxis) - 1)] + [np.inf]

    log.loc[:, 'X'] = pd.cut(log['RPM'], bins=xedges, labels=False, duplicates='drop')
    log.loc[:, 'Y'] = pd.cut(log['MAP'], bins=yedges, labels=False, duplicates='drop')
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

def _display_maf_results(results, maftables, parent):
    """Creates a Toplevel window with a 2x2 grid to display the MAF tables."""
    window = Toplevel(parent)
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
        pt.color_cells(results[f"IDX{i}"].to_numpy(), maftables[i])

# --- Main Function ---

def MAF_tune(log, mafxaxis, mafyaxis, maftables, combmodes_MAF, logvars, parent):
    """Main orchestrator for the MAF tuning process."""
    print(" -> Initializing MAF analysis...")
    params = _get_maf_parameters()

    print(" -> Preparing MAF data from logs...")
    log = _prepare_maf_data(log, logvars)

    print(" -> Creating data bins from MAF axes...")
    log = _create_bins(log, mafxaxis, mafyaxis)

    results = {}
    for idx in range(4):
        print(f" -> Processing MAF Table IDX{idx}...")
        current_table = maftables[idx]
        idx_modes = np.where(combmodes_MAF == idx)[0]
        log_filtered = log[log['CMB'].isin(idx_modes)].copy()
        log_filtered.dropna(subset=['RPM', 'MAP', 'ADD_MAF'], inplace=True)

        print(f"   -> Fitting 3D surface for IDX{idx}...")
        blend_surface = _fit_surface_maf(log_filtered, mafxaxis, mafyaxis)

        print(f"   -> Calculating correction map for IDX{idx}...")
        recommended_table, final_change, changed_mask = _calculate_maf_correction(
            log_filtered, blend_surface, current_table, mafxaxis, mafyaxis, params['confidence']
        )

        # Store results
        xlabels = [str(x) for x in mafxaxis]
        ylabels = [str(y) for y in mafyaxis]
        results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

        if params['show_3d_plot']:
            print(f"   -> Plotting 3D surface for IDX{idx}...")
            plot_3d_surface(
                title=f"IDX{idx} MAF Correction (Changes Marked)",
                xaxis=mafxaxis,
                yaxis=mafyaxis,
                old_map=current_table,
                new_map=recommended_table,
                log_data=log_filtered,
                changed_mask=changed_mask,
                x_label='Engine Speed (RPM)',
                y_label='Manifold Absolute Pressure (MAP)',
                z_label='Additive MAF Correction',
                data_col_name='ADD_MAF'
            )

    print(" -> Displaying final results tables...")
    _display_maf_results(results, maftables, parent)
    return results
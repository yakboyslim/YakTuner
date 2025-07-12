# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/MAF.py

"""
Mass Airflow (MAF) Correction Tuning Module

This module contains pure, non-UI functions to analyze engine logs and recommend
adjustments to the four primary MAF correction tables (IDX0-IDX3).
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate
import streamlit as st

# --- Helper Functions ---

def _process_and_filter_maf_data(log, logvars):
    """
    A pure function to prepare and filter log data for MAF tuning.
    It returns a processed DataFrame and a list of warnings.
    """
    warnings = []
    df = log.copy()

    # Use .loc for direct assignment to avoid SettingWithCopyWarning
    df.loc[:, 'MAP'] = df['MAP'] * 10

    if "OILTEMP" in logvars:
        df = df[df['OILTEMP'] > 180].copy()

    if 'LAM_DIF' not in logvars:
        df.loc[:, 'LAM_DIF'] = 1/df['LAMBDA_SP'] - 1/df['LAMBDA']
        warnings.append('Recommend logging LAM DIF. Using calculated value, but may introduce inaccuracy.')

    if "FAC_MFF_ADD" in logvars:
        final_ltft = df["FAC_MFF_ADD"]
    else:
        final_ltft = df['LTFT']

    if 'FAC_LAM_OUT' in logvars:
        final_stft = df['FAC_LAM_OUT']
    else:
        final_stft = df['STFT']

    df.loc[:, 'ADD_MAF'] = final_stft + final_ltft - df['LAM_DIF']

    if 'MAF_COR' in logvars:
        df.loc[:, 'ADD_MAF'] = df['ADD_MAF'] + df['MAF_COR']
    else:
        warnings.append('Recommend logging MAF_COR for increased accuracy.')

    df = df[df['state_lam'] == 1].copy()
    return df, warnings

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

    # Quantize the final table to the ECU's resolution (5.12 = 256 / 50)
    recommended_table = np.round(new_table * 5.12) / 5.12
    return recommended_table

# --- Main Orchestrator Function ---
@st.cache_data(show_spinner="Running MAF analysis...")
def run_maf_analysis(log, mafxaxis, mafyaxis, maftables, combmodes_MAF, logvars):
    """
    Main orchestrator for the MAF tuning process. A pure computational function.

    Args:
        log (pd.DataFrame): The mapped log data.
        mafxaxis, mafyaxis (np.ndarray): The axes for the MAF tables.
        maftables (list[np.ndarray]): A list of the four original MAF tables.
        combmodes_MAF (np.ndarray): The combination modes map.
        logvars (list): A list of available variable names in the log.

    Returns:
        dict: A dictionary containing all results.
    """
    print(" -> Initializing MAF analysis...")
    params = {'confidence': 0.7} # Hardcoded parameter

    print(" -> Preparing MAF data from logs...")
    processed_log, warnings = _process_and_filter_maf_data(log, logvars)

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'results_maf': None}

    print(" -> Creating data bins from MAF axes...")
    log_binned = _create_bins(processed_log, mafxaxis, mafyaxis)

    results = {}
    for idx in range(4):
        print(f" -> Processing MAF Table IDX{idx}...")
        current_table = maftables[idx]
        idx_modes = np.where(combmodes_MAF == idx)[0]
        log_filtered = log_binned[log_binned['CMB'].isin(idx_modes)].copy()
        log_filtered.dropna(subset=['RPM', 'MAP', 'ADD_MAF'], inplace=True)

        print(f"   -> Fitting 3D surface for IDX{idx}...")
        blend_surface = _fit_surface_maf(log_filtered, mafxaxis, mafyaxis)

        print(f"   -> Calculating correction map for IDX{idx}...")
        recommended_table = _calculate_maf_correction(
            log_filtered, blend_surface, current_table, mafxaxis, mafyaxis, params['confidence']
        )

        # Store results as a DataFrame
        xlabels = [str(x) for x in mafxaxis]
        ylabels = [str(y) for y in mafyaxis]
        results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    # 3D plotting and table display are now handled by the UI (Streamlit).

    print(" -> MAF analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'results_maf': results
    }
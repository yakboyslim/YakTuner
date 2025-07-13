
"""
Multiplicative Fuel Factor (MFF) Tuning Module

This module contains pure, non-UI functions to analyze engine logs and recommend
adjustments to the five primary MFF tables (IDX0-IDX4).
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate

# --- Helper Functions ---

def _get_mff_parameters():
    """Returns hardcoded parameters for MFF tuning."""
    params = {
        'confidence': 0.7,
        'show_3d_plot': True
    }
    return params

def _process_and_filter_mff_data(log, logvars):
    """
    A pure function to prepare and filter log data for MFF tuning.
    It returns a processed DataFrame and a list of warnings.
    """
    warnings = []
    df = log.copy()

    if "OILTEMP" in logvars:
        df = df[df['OILTEMP'] > 180].copy()

    if 'LAM_DIF' not in logvars:
        df.loc[:, 'LAM_DIF'] = 1/df['LAMBDA_SP'] - 1/df['LAMBDA']
        warnings.append('Recommend logging LAM DIF. Using calculated value, but may introduce inaccuracy.')

    if "FAC_MFF_ADD" in logvars:
        final_ltft = df["FAC_MFF_ADD"]
    else:
        final_ltft = df['STFT'] # Note: Original code used STFT here, preserving that.

    if 'FAC_LAM_OUT' in logvars:
        final_stft = df['FAC_LAM_OUT']
    else:
        final_stft = df['STFT']

    # --- Core Calculation: Create a MULTIPLICATIVE factor ---
    additive_correction = (final_stft + final_ltft)/100 - df['LAM_DIF']
    df.loc[:, 'MFF_FACTOR'] = 1.0 + additive_correction

    if 'MFF_COR' in logvars:
        # Assuming MFF_COR is also an additive correction to the factor
        df.loc[:, 'MFF_FACTOR'] = additive_correction + log['MFF_COR']
    else:
        warnings.append('Recommend logging MFF_COR for increased accuracy.')

    df = df[df['state_lam'] == 1].copy()
    return df, warnings

def _create_bins(log, mffxaxis, mffyaxis):
    """Discretizes log data into bins based on MFF map axes."""
    xedges = [0] + [(mffxaxis[i] + mffxaxis[i + 1]) / 2 for i in range(len(mffxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mffyaxis[i] + mffyaxis[i + 1]) / 2 for i in range(len(mffyaxis) - 1)] + [np.inf]

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

    # Quantize the final table to a common ECU resolution for multiplicative factors (1/1024)
    recommended_table = np.round(new_table * 1024) / 1024
    return recommended_table

# --- Main Orchestrator Function ---
def run_mff_analysis(log, mffxaxis, mffyaxis, mfftables, combmodes_MFF, logvars):
    """
    Main orchestrator for the MFF tuning process. A pure computational function.

    Args:
        log (pd.DataFrame): The mapped log data.
        mffxaxis, mffyaxis (np.ndarray): The axes for the MFF tables.
        mfftables (list[np.ndarray]): A list of the five original MFF tables.
        combmodes_MFF (np.ndarray): The combination modes map.
        logvars (list): A list of available variable names in the log.

    Returns:
        dict: A dictionary containing all results.
    """
    print(" -> Initializing MFF analysis...")
    params = {'confidence': 0.7} # Hardcoded parameter

    print(" -> Preparing MFF data from logs...")
    processed_log, warnings = _process_and_filter_mff_data(log, logvars)

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'results_mff': None}

    print(" -> Creating data bins from MFF axes...")
    log_binned = _create_bins(processed_log, mffxaxis, mffyaxis)

    results = {}
    # Loop through all 5 MFF tables
    for idx in range(5):
        print(f" -> Processing MFF Table IDX{idx}...")
        current_table = mfftables[idx]
        idx_modes = np.where(combmodes_MFF == idx)[0]
        log_filtered = log_binned[log_binned['CMB'].isin(idx_modes)].copy()
        log_filtered.dropna(subset=['RPM', 'MAF', 'MFF_FACTOR'], inplace=True)

        print(f"   -> Fitting 3D surface for IDX{idx}...")
        blend_surface = _fit_surface_mff(log_filtered, mffxaxis, mffyaxis)

        print(f"   -> Calculating correction map for IDX{idx}...")
        recommended_table = _calculate_mff_correction(
            log_filtered, blend_surface, current_table, mffxaxis, mffyaxis, params['confidence']
        )

        # Store results as a DataFrame
        xlabels = [str(x) for x in mffxaxis]
        ylabels = [str(y) for y in mffyaxis]
        results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    # 3D plotting and table display are now handled by the UI (Streamlit).

    print(" -> MFF analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'results_mff': results
    }
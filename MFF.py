# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/MFF.py

"""
Multiplicative Fuel Factor (MFF) Tuning Module

This module contains pure, non-UI functions to analyze engine logs and recommend
adjustments to the five primary MFF tables (IDX0-IDX4).
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate


# --- Helper Functions ---

def _process_and_filter_mff_data(log, logvars, tuning_mode='MFF'):
    """
    A pure function to prepare and filter log data for MFF tuning using the new
    unified correction formula. It returns a processed DataFrame and a list of warnings.
    """
    warnings = []
    df = log.copy()

    # --- Step-wise filtering with specific warnings ---
    # Step 1: Oil Temperature Filter
    if "OILTEMP" in logvars:
        df = df[df['OILTEMP'] > 180].copy()
        if df.empty:
            warnings.append("No data remained after filtering for Oil Temperature > 180°F.")
            return pd.DataFrame(), warnings

    # Step 2: Pre-emptive check for division by zero
    if 'LAMBDA_SP' in df.columns:
        zero_sp_mask = df['LAMBDA_SP'] == 0
        if zero_sp_mask.any():
            warnings.append("Log contains rows where 'LAMBDA_SP' is zero. These rows are being ignored.")
            df = df[~zero_sp_mask].copy()
            if df.empty:
                warnings.append("No data remained after filtering for zero-value 'LAMBDA_SP'.")
                return pd.DataFrame(), warnings
    else:
        warnings.append("Log variable 'LAMBDA_SP' not found. Cannot calculate fuel error.")
        return pd.DataFrame(), warnings

    # --- New Unified Correction Formula ---
    required_vars = ['LAMBDA', 'LAMBDA_SP']
    if not all(v in df.columns for v in required_vars):
        raise ValueError(f"MFF analysis requires essential log variables: {required_vars}")

    maf_cor = df.get('MAF_COR', 0.0)
    if 'MAF_COR' not in logvars: warnings.append("Log 'MAF_COR' for best accuracy.")

    mff_cor = df.get('MFF_COR', 1.0)
    if 'MFF_COR' not in logvars: warnings.append("Log 'MFF_COR' for best accuracy.")

    # --- FIX: Simplified retrieval of STFT and LTFT variables ---
    fac_lam_out = df.get('FAC_LAM_OUT', 0.0)
    stft = df.get('STFT', 0.0)
    fac_mff_add = df.get('FAC_MFF_ADD', 0.0)
    ltft = df.get('LTFT', 0.0)
    # --- END FIX ---

    # --- FIX: Simplified degradation logic for LTFT ---
    ltft_correction_term = 1.0
    if 'FAC_MFF_ADD' in logvars:
        ltft_correction_term = (1 + fac_mff_add / 100)
        warnings.append("Using 'FAC_MFF_ADD' for LTFT correction.")
    elif 'LTFT' in logvars:
        ltft_correction_term = (1 + ltft / 100)
        warnings.append("Using 'LTFT' as fallback for LTFT correction.")
    else:
        warnings.append("No suitable LTFT correction variable found. Assuming neutral LTFT correction (1.0).")
    # --- END FIX ---

    # --- FIX: Simplified degradation logic for STFT ---
    stft_correction_term = 1.0
    if 'FAC_LAM_OUT' in logvars:
        stft_correction_term = (1 + fac_lam_out / 100)
        warnings.append("Using 'FAC_LAM_OUT' for STFT correction.")
    elif 'STFT' in logvars:
        stft_correction_term = (1 + stft / 100)
        warnings.append("Using 'STFT' as fallback for STFT correction.")
    else:
        warnings.append("No suitable STFT correction variable found. Assuming neutral STFT correction (1.0).")
    # --- END FIX ---

    total_ecu_factor = (1 + maf_cor / 100) * stft_correction_term * mff_cor * ltft_correction_term
    measured_error = df['LAMBDA'] / df['LAMBDA_SP']
    target_factor = total_ecu_factor * measured_error

    if tuning_mode == 'MFF':
        mff_cor_new = target_factor / (1 + maf_cor / 100)
        df.loc[:, 'MFF_FACTOR'] = mff_cor_new
    elif tuning_mode == 'BOTH':
        if 'MAF_COR_NEW' not in df.columns:
            raise ValueError("'tuning_mode' is 'BOTH', but the log DataFrame is missing the 'MAF_COR_NEW' column.")
        maf_cor_new = df['MAF_COR_NEW']
        mff_cor_new = target_factor / (1 + maf_cor_new / 100)
        df.loc[:, 'MFF_FACTOR'] = mff_cor_new

    return df, warnings


def _create_bins(log, mffxaxis, mffyaxis):
    """Discretizes log data into bins based on MFF map axes."""
    xedges = [0] + [(mffxaxis[i] + mffxaxis[i + 1]) / 2 for i in range(len(mffxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mffyaxis[i] + mffyaxis[i + 1]) / 2 for i in range(len(mffyaxis) - 1)] + [np.inf]

    log.loc[:, 'X'] = pd.cut(log['RPM'], bins=xedges, labels=False, duplicates='drop')
    log.loc[:, 'Y'] = pd.cut(log['MAF'], bins=yedges, labels=False, duplicates='drop')
    return log


def _fit_surface_mff(log_data, mffxaxis, mffyaxis):
    """
    Fits a surface to the MFF correction data, gracefully handling low-dimensional and
    non-finite data to prevent Qhull and other interpolation errors.
    """
    if log_data.empty:
        return np.ones((len(mffyaxis), len(mffxaxis)))

    # Proactively clean data to remove rows with non-finite values (NaN or inf)
    finite_mask = np.isfinite(log_data['RPM']) & np.isfinite(log_data['MAF']) & np.isfinite(log_data['MFF_FACTOR'])
    clean_log_data = log_data[finite_mask]

    if clean_log_data.empty or len(clean_log_data) < 3:
        return np.ones((len(mffyaxis), len(mffxaxis)))

    points = clean_log_data[['RPM', 'MAF']].values
    values = clean_log_data['MFF_FACTOR'].values
    grid_x, grid_y = np.meshgrid(mffxaxis, mffyaxis)

    x_variation = np.ptp(points[:, 0]) > 1e-6
    y_variation = np.ptp(points[:, 1]) > 1e-6

    if x_variation and y_variation:
        try:
            fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
        except Exception:
            fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
    elif x_variation:
        print("Warning: MFF log data is flat in the Y-dimension (MAF). Using 1D interpolation.")
        unique_x, mean_values_idx = np.unique(points[:, 0], return_inverse=True)
        avg_values = np.bincount(mean_values_idx, weights=values) / np.bincount(mean_values_idx)
        interp_values = np.interp(mffxaxis, unique_x, avg_values, left=avg_values[0], right=avg_values[-1])
        fitted_surface = np.tile(interp_values, (len(mffyaxis), 1))
    elif y_variation:
        print("Warning: MFF log data is flat in the X-dimension (RPM). Using 1D interpolation.")
        unique_y, mean_values_idx = np.unique(points[:, 1], return_inverse=True)
        avg_values = np.bincount(mean_values_idx, weights=values) / np.bincount(mean_values_idx)
        interp_values = np.interp(mffyaxis, unique_y, avg_values, left=avg_values[0], right=avg_values[-1])
        fitted_surface = np.tile(interp_values, (len(mffxaxis), 1)).T
    else:
        print("Warning: MFF log data has no variation in X or Y dimensions. Using mean value.")
        mean_value = np.mean(values)
        fitted_surface = np.full((len(mffyaxis), len(mffxaxis)), mean_value)

    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        if nearest_fill is not None and not np.all(np.isnan(nearest_fill)):
            fitted_surface[nan_mask] = nearest_fill

    filled_surface = np.nan_to_num(fitted_surface, nan=1.0)

    believable_min = 0.92
    believable_max = 1.08
    clamped_surface = np.clip(filled_surface, believable_min, believable_max)

    return clamped_surface


def _calculate_mff_correction(log_data, blend_surface, old_table, mffxaxis, mffyaxis, confidence, additive_mode=False):
    """
    Applies confidence interval logic to determine the final correction table.
    """
    new_table = old_table.copy()
    max_count = 50
    interp_factor = 0.25

    for i in range(len(mffxaxis)):
        for j in range(len(mffyaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]

            # Proactively clean data for statistical analysis
            cell_data = cell_data[np.isfinite(cell_data['MFF_FACTOR'])]

            count = len(cell_data)

            if count > 3:
                mean, std_dev = stats.norm.fit(cell_data['MFF_FACTOR'])
                surface_val = blend_surface[j, i]
                target_val = (surface_val * interp_factor) + (mean * (1 - interp_factor))
                low_ci, high_ci = stats.norm.interval(confidence, loc=target_val,
                                                      scale=std_dev if std_dev > 0 else 1e-9)
                current_val_from_table = old_table[j, i]
                comparison_val = 1.0 if additive_mode else current_val_from_table

                if not (low_ci <= comparison_val <= high_ci):
                    weight = min(count, max_count) / max_count
                    change_amount = (target_val - comparison_val) * weight
                    new_table[j, i] = current_val_from_table + change_amount

    recommended_table = np.round(new_table * 1024) / 1024
    return recommended_table


# --- Main Orchestrator Function ---
def run_mff_analysis(log, mffxaxis, mffyaxis, mfftables, combmodes_MFF, logvars, tuning_mode='MFF'):
    """
    Main orchestrator for the MFF tuning process. A pure computational function.
    """
    try:
        print(" -> Initializing MFF analysis...")
        params = {'confidence': 0.7}

        print(" -> Preparing MFF data from logs...")
        processed_log, warnings = _process_and_filter_mff_data(log, logvars, tuning_mode=tuning_mode)

        additive_mode = 'MFF_COR' not in logvars
        if additive_mode:
            warnings.append("MFF_COR not found in logs. Switching to additive correction mode.")

        if processed_log.empty:
            # The warnings list will already contain the specific reason from the filter function
            return {'status': 'Failure', 'warnings': warnings, 'results_mff': None}

        print(" -> Creating data bins from MFF axes...")
        log_binned = _create_bins(processed_log, mffxaxis, mffyaxis)

        if 'CMB' not in log_binned.columns:
            warnings.append("Log variable 'CMB' (Combination Mode) not found. Cannot determine which MFF table to correct.")
            return {'status': 'Failure', 'warnings': warnings, 'results_mff': None}

        results = {}
        for idx in range(5):
            print(f" -> Processing MFF Table IDX{idx}...")
            current_table = mfftables[idx]
            idx_modes = np.where(combmodes_MFF == idx)[0]
            log_filtered = log_binned[log_binned['CMB'].isin(idx_modes)].copy()

            log_filtered = log_filtered[np.isfinite(log_filtered['RPM']) & np.isfinite(log_filtered['MAF']) & np.isfinite(
                log_filtered['MFF_FACTOR'])]

            if log_filtered.empty:
                warnings.append(f"No valid data found for MFF Table IDX{idx}. This table will not be changed.")
                xlabels = [str(x) for x in mffxaxis]
                ylabels = [str(y) for y in mffyaxis]
                results[f'IDX{idx}'] = pd.DataFrame(current_table, columns=xlabels, index=ylabels)
                continue

            print(f"   -> Fitting 3D surface for IDX{idx}...")
            blend_surface = _fit_surface_mff(log_filtered, mffxaxis, mffyaxis)

            print(f"   -> Calculating correction map for IDX{idx}...")
            recommended_table = _calculate_mff_correction(
                log_filtered, blend_surface, current_table, mffxaxis, mffyaxis, params['confidence'],
                additive_mode=additive_mode
            )

            xlabels = [str(x) for x in mffxaxis]
            ylabels = [str(y) for y in mffyaxis]
            results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

        print(" -> MFF analysis complete.")
        return {
            'status': 'Success',
            'warnings': warnings,
            'results_mff': results
        }
    except Exception as e:
        return {
            'status': 'Failure',
            'warnings': [f"A critical error occurred inside the MFF module: {e}"],
            'results_mff': None
        }
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re

def _format_tta_att_suffix(map_suffix):
    """
    Formats a map suffix like 'H[STND][1][1]' or 'H_LFT_1_1_2' into a
    human-readable name like 'H[VVL0][INT 0][EXH 0]'.
    (Copied from TTA_ATT.py to ensure consistency)
    """
    pattern = re.compile(r"(\w+)\[(\w+(?:_\d+)?)\]\[(\d+)\]\[(\d+)\]")
    match = pattern.match(map_suffix)

    if not match:
        return map_suffix

    try:
        prefix, mode, int_index_str, exh_index_str = match.groups()
        int_index = int(int_index_str) - 1
        exh_index = int(exh_index_str) - 1

        if mode.upper() == 'STND':
            vvl_mode = "VVL0"
        elif mode.upper().startswith('LFT_'):
            vvl_num = mode.split('_')[1]
            vvl_mode = f"VVL{vvl_num}"
        else:
            return map_suffix

        return f"{prefix}[{vvl_mode}][INT {int_index}][EXH {exh_index}]"

    except (ValueError, IndexError):
        return map_suffix

def find_tta_att_pairs(all_maps):
    """
    Identifies TTA and ATT map pairs from the loaded map dictionary.
    Returns a dictionary of valid pairs.
    """
    tta_base = "IP_MAF_STK_SP_VVL_CAM_"
    att_base = "IP_TQI_REF_N_M_AIR_VVL_CAM_"

    pairs = {}
    processed_suffixes = set()

    for map_name in all_maps.keys():
        if map_name.startswith(tta_base) and not map_name.endswith(('_X', '_Y')):
            map_suffix = map_name[len(tta_base):]

            if map_suffix in processed_suffixes:
                continue

            tta_name = map_name
            att_name = att_base + map_suffix

            # Required axes
            tta_x = tta_name + "_X" # RPM
            tta_y = tta_name + "_Y" # Torque
            att_x = att_name + "_X" # RPM
            att_y = att_name + "_Y" # Airflow

            required_maps = [tta_name, att_name, tta_x, tta_y, att_x, att_y]

            if all(m in all_maps for m in required_maps):
                processed_suffixes.add(map_suffix)
                readable_name = _format_tta_att_suffix(map_suffix)
                pairs[readable_name] = {
                    'tta_data': all_maps[tta_name],
                    'tta_x': all_maps[tta_x], # RPM
                    'tta_y': all_maps[tta_y], # Torque
                    'att_data': all_maps[att_name],
                    'att_x': all_maps[att_x], # RPM
                    'att_y': all_maps[att_y]  # Airflow
                }

    return pairs

def extrapolate_tta(tta_data, tta_y_axis, desired_max_tq):
    """
    Extrapolates the TTA table based on the desired max torque.

    Args:
        tta_data: 2D numpy array (Rows=Torque, Cols=RPM).
        tta_y_axis: 1D numpy array (Torque).
        desired_max_tq: Float, the new maximum torque for the last row.

    Returns:
        new_tta_data: Extrapolated 2D numpy array.
        new_y_axis: Updated 1D numpy array (Torque).
    """
    # Create copies to avoid modifying originals
    new_tta_data = tta_data.copy().astype(float)
    new_y_axis = tta_y_axis.copy().astype(float)

    # Update the last value of the Y-axis
    new_y_axis[-1] = desired_max_tq

    # Identify the ceiling value (Global Max Z)
    ceiling_value = np.max(tta_data)

    rows, cols = tta_data.shape

    for col in range(cols):
        column_data = tta_data[:, col]

        # Identify non-ceiling points
        # We consider a point "ceiling" if it equals the global max.
        # Use a small epsilon for float comparison safety
        is_ceiling = column_data >= (ceiling_value - 1e-5)

        # Points to fit: Not ceiling
        valid_mask = ~is_ceiling

        # Check if we have enough points to fit a line (at least 2)
        if np.sum(valid_mask) < 2:
            # Not enough data to extrapolate safely.
            continue

        y_fit = tta_y_axis[valid_mask]
        z_fit = column_data[valid_mask]

        # Linear Fit: Z = m * Y + c
        m, c = np.polyfit(y_fit, z_fit, 1)

        # 1. Update Ceiling Cells (including last row if it was ceiling)
        # We use the NEW Y axis values.
        # Note: For rows 0 to N-2, new_y_axis == tta_y_axis.
        # For row N-1 (last row), new_y_axis is desired_max_tq.

        # Find indices that need updating:
        # - Any cell that was "ceiling"
        # - The last row (always, because Y changed)

        indices_to_update = np.where(is_ceiling)[0]
        # Ensure last row is in the update list
        if (rows - 1) not in indices_to_update:
            indices_to_update = np.append(indices_to_update, rows - 1)

        for row_idx in indices_to_update:
            y_val = new_y_axis[row_idx]
            new_z = m * y_val + c

            # Additional check: Don't produce negative airflow
            new_z = max(0.0, new_z)

            new_tta_data[row_idx, col] = new_z

    return new_tta_data, new_y_axis

def generate_inverse_att(new_tta_data, tta_x_axis, new_tta_y_axis, att_x_axis, att_y_axis):
    """
    Generates the ATT table by inverting the New TTA table using column-wise 1D interpolation.

    Args:
        new_tta_data: The new TTA Z-values (Airflow). (Rows=Torque, Cols=RPM)
        tta_x_axis: TTA X-axis (RPM).
        new_tta_y_axis: New TTA Y-axis (Torque).
        att_x_axis: ATT X-axis (RPM).
        att_y_axis: ATT Y-axis (Airflow).

    Returns:
        new_att_data: Interpolated/Extrapolated ATT table (Torque). (Rows=Airflow, Cols=RPM)
    """
    # 1. Create a "Virtual TTA" interpolated to the ATT RPM grid.
    #    TTA is (Torque x TTA_RPM). We want (Torque x ATT_RPM).
    #    Rows are preserved (Torque), Columns are resampled.

    num_torque_rows = new_tta_data.shape[0]
    num_att_rpm_cols = len(att_x_axis)

    virtual_tta = np.zeros((num_torque_rows, num_att_rpm_cols))

    for r in range(num_torque_rows):
        # TTA Airflow values for this torque row across original TTA RPMs
        z_row = new_tta_data[r, :]

        # Create interpolator: Airflow = f(RPM) for this constant torque
        # Use 'linear' interpolation, and extrapolate if ATT RPMs are outside TTA RPM range
        f_rpm = interp1d(tta_x_axis, z_row, kind='linear', fill_value='extrapolate')

        # Calculate Airflow at ATT RPMs
        virtual_tta[r, :] = f_rpm(att_x_axis)

    # 2. Invert each column (RPM) to map Airflow -> Torque.
    #    For each column c (corresponding to att_x_axis[c]):
    #      We have a curve: Torque (new_tta_y_axis) vs Airflow (virtual_tta[:, c])
    #      We want to find Torque for target Airflows (att_y_axis).

    new_att_data = np.zeros((len(att_y_axis), len(att_x_axis)))

    for c in range(num_att_rpm_cols):
        col_airflow = virtual_tta[:, c]
        col_torque = new_tta_y_axis

        # Check monotonicity of Airflow vs Torque
        # TTA maps generally have Airflow increasing with Torque.
        # If not strictly monotonic, interp1d might fail or behave oddly.
        # We can sort just in case, or handle it?
        # Usually physics dictates monotonic.
        # If multiple torque values produce same airflow, inversion is ambiguous.
        # We assume monotonic.

        try:
            # We want Torque = f(Airflow)
            f_inv = interp1d(col_airflow, col_torque, kind='linear', fill_value='extrapolate')
            new_att_data[:, c] = f_inv(att_y_axis)
        except Exception as e:
            # Fallback for errors (e.g., flat line airflow)
            print(f"Error inverting column {c} (RPM {att_x_axis[c]}): {e}")
            new_att_data[:, c] = 0.0

    # Clamp negative torques to 0 if needed? Or allow them?
    # Usually torque isn't negative in these maps (reference torque).
    new_att_data = np.maximum(new_att_data, 0.0)

    return new_att_data

def run_tta_extrapolation(all_maps, desired_max_tq):
    """
    Main entry point. Finds pairs, extrapolates, and returns results.
    """
    pairs = find_tta_att_pairs(all_maps)
    results = {}

    if not pairs:
        return {'status': 'Failed', 'message': "No TTA/ATT pairs found."}

    for name, data in pairs.items():
        try:
            # 1. Extrapolate TTA
            new_tta, new_y_axis = extrapolate_tta(
                data['tta_data'],
                data['tta_y'],
                desired_max_tq
            )

            # 2. Generate ATT
            # First, update the ATT Y-axis (Airflow) to cover the new max airflow
            max_airflow = np.max(new_tta)
            new_att_y_axis = data['att_y'].copy().astype(float)
            new_att_y_axis[-1] = max_airflow

            new_att = generate_inverse_att(
                new_tta,
                data['tta_x'],
                new_y_axis,
                data['att_x'],
                new_att_y_axis
            )

            results[name] = {
                'original_tta': pd.DataFrame(data['tta_data'], index=data['tta_y'], columns=data['tta_x']),
                'new_tta': pd.DataFrame(new_tta, index=new_y_axis, columns=data['tta_x']),
                'original_att': pd.DataFrame(data['att_data'], index=data['att_y'], columns=data['att_x']),
                'new_att': pd.DataFrame(new_att, index=new_att_y_axis, columns=data['att_x']),
                'new_torque_axis': new_y_axis,
                'new_att_airflow_axis': new_att_y_axis
            }

        except Exception as e:
            # Log error but continue with other pairs
            print(f"Error processing {name}: {e}")
            results[name] = {'error': str(e)}

    return {'status': 'Success', 'results': results}

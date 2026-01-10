import numpy as np
import pandas as pd
from scipy.interpolate import griddata
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
            # Strategy: Just keep the old values, but we MUST update the last row
            # if we changed the axis.
            # If we can't fit, maybe assume 0 slope? Or just use the last valid value?
            # For now, let's leave it alone but warn if possible.
            # However, if the last row IS the only point or something, we have issues.
            # Let's try to fit even with all points if valid_mask is too small,
            # assuming the "ceiling" detection might be too aggressive?
            # No, if all are ceiling, it's a flat line.
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

            # Ensure we don't accidentally lower a value below the ceiling if it was capped?
            # User said "Z values... will need to be increased".
            # The fit should handle this.

            # Additional check: Don't produce negative airflow
            new_z = max(0.0, new_z)

            new_tta_data[row_idx, col] = new_z

    return new_tta_data, new_y_axis

def generate_inverse_att(new_tta_data, tta_x_axis, new_tta_y_axis, att_x_axis, att_y_axis):
    """
    Generates the ATT table by inverting the New TTA table.

    Args:
        new_tta_data: The new TTA Z-values (Airflow).
        tta_x_axis: TTA X-axis (RPM).
        new_tta_y_axis: New TTA Y-axis (Torque).
        att_x_axis: ATT X-axis (RPM).
        att_y_axis: ATT Y-axis (Airflow).

    Returns:
        new_att_data: Interpolated ATT table (Torque).
    """
    # TTA Mapping: (RPM, Torque) -> Airflow
    # We want ATT: (RPM, Airflow) -> Torque

    # 1. Create source points from TTA
    # meshgrid returns (Y-dim, X-dim) arrays
    rpm_mesh, torque_mesh = np.meshgrid(tta_x_axis, new_tta_y_axis)

    # Source X (RPM), Source Y (Airflow - which is the Z of TTA)
    # We are mapping (RPM, Airflow) -> Torque

    # Flatten arrays
    src_rpm = rpm_mesh.flatten()
    src_torque = torque_mesh.flatten()
    src_airflow = new_tta_data.flatten()

    # Input Points: (RPM, Airflow)
    points = np.column_stack((src_rpm, src_airflow))
    values = src_torque

    # 2. Create target query points for ATT
    att_rpm_mesh, att_airflow_mesh = np.meshgrid(att_x_axis, att_y_axis)
    query_points = np.column_stack((att_rpm_mesh.flatten(), att_airflow_mesh.flatten()))

    # 3. Interpolate
    # Use 'linear' interpolation. 'fill_value' handles out of bounds.
    # Out of bounds might happen if new TTA doesn't cover the low/high airflow of ATT.
    # For extrapolation, 'nearest' is a safer fallback for fill, or we can leave NaNs.
    # griddata doesn't support extrapolation with 'linear'.
    # LinearNDInterpolator vs griddata: griddata is a convenience wrapper.

    # Logic from TTA_ATT.py used griddata with method='linear'.
    # It did: interpolated_tta_inv_table = np.nan_to_num(..., nan=0.0)
    # We should probably match that behavior.

    new_att_flat = griddata(points, values, query_points, method='linear')
    new_att_data = new_att_flat.reshape(att_y_axis.shape[0], att_x_axis.shape[0])

    # Fill NaNs. TTA_ATT used 0.0.
    # However, for Torque, 0.0 might be valid or might be wrong.
    # If we are outside the range, maybe we should clamp?
    # For now, 0.0 is safe-ish, but let's check if we can do better.
    # Actually, TTA_ATT.py logic was: interpolated_tta_inv_table = np.nan_to_num(interpolated_tta_inv_table, nan=0.0)

    new_att_data = np.nan_to_num(new_att_data, nan=0.0)

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

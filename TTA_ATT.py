import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import streamlit as st


def _format_tta_att_suffix(map_suffix):
    """
    Formats a map suffix like 'H_LFT_1_1_2' into a human-readable
    name like 'H[VVL1][INT 0][EXH 1]'.
    """
    # --- THIS IS THE DIAGNOSTIC LINE ---
    st.warning(f"DEBUG: Formatting '{map_suffix}'...")
    # ------------------------------------
    try:
        parts = map_suffix.split('_')
        if len(parts) < 3:
            return map_suffix

        exh_index = int(parts.pop()) - 1
        int_index = int(parts.pop()) - 1
        prefix = parts[0]
        mode_identifier = parts[1].upper()

        if mode_identifier == 'STD' and len(parts) == 2:
            vvl_mode = "VVL0"
        elif mode_identifier == 'LFT' and len(parts) == 3:
            vvl_mode = f"VVL{parts[2]}"
        else:
            return map_suffix

        formatted_name = f"{prefix}[{vvl_mode}][INT {int_index}][EXH {exh_index}]"
        # --- ADD A SECOND DIAGNOSTIC LINE ---
        st.warning(f"DEBUG: Result -> '{formatted_name}'")
        # --------------------------------------
        return formatted_name

    except (ValueError, IndexError):
        return map_suffix


def run_tta_att_analysis(all_maps):
    """
    Compares the inverse of Torque-to-Airflow (TTA) tables against their
    corresponding Airflow-to-Torque (ATT) tables to check for consistency.
    This version discovers existing tables using simple string matching.
    """
    warnings = []
    results = {}
    processed_suffixes = set()
    debug_logs = []

    # --- Define base names ---
    tta_base = "IP_MAF_STK_SP_VVL_CAM_"
    att_base = "IP_TQI_REF_N_M_AIR_VVL_CAM_"

    debug_logs.append("-> Initializing TTA/ATT Consistency Check...")

    # Iterate through all map names that were actually loaded from the file
    for map_name in all_maps.keys():
        # Check if the current map is a potential TTA base table
        if map_name.startswith(tta_base) and not map_name.endswith(('_X', '_Y')):
            # Extract the unique suffix from the found table name
            map_suffix = map_name[len(tta_base):]

            if map_suffix in processed_suffixes:
                continue

            # --- Build the names of its 5 required partners. ---
            tta_name = map_name
            att_name = att_base + map_suffix
            tta_x_axis_name = tta_name + "_X"
            tta_y_axis_name = tta_name + "_Y"
            att_x_axis_name = att_name + "_X"
            att_y_axis_name = att_name + "_Y"

            required_map_names = [
                tta_name, att_name, tta_x_axis_name, tta_y_axis_name,
                att_x_axis_name, att_y_axis_name
            ]

            # Check which of the required partners are missing.
            missing_maps = [m for m in required_map_names if m not in all_maps]

            if missing_maps:
                # This is a high-quality warning for partially defined sets.
                warnings.append(f"Incomplete set for '{map_suffix}'. Missing: {', '.join(missing_maps)}")
                continue

            # --- If we get here, all 6 maps for this set were found. ---
            processed_suffixes.add(map_suffix)
            tab_name = _format_tta_att_suffix(map_suffix)
            debug_logs.append(f"  -> Found complete set for {tab_name}. Processing...")

            try:
                tta_table = all_maps[tta_name]
                tta_rpm_axis = all_maps[tta_x_axis_name]
                tta_torque_axis = all_maps[tta_y_axis_name]
                att_table = all_maps[att_name]
                att_rpm_axis = all_maps[att_x_axis_name]
                att_airmass_axis = all_maps[att_y_axis_name]

                # --- Perform 2D Interpolation ---
                rpm_mesh, _ = np.meshgrid(tta_rpm_axis, tta_torque_axis)
                source_points = np.array([rpm_mesh.flatten(), tta_table.flatten()]).T
                _, torque_mesh = np.meshgrid(tta_rpm_axis, tta_torque_axis)
                source_values = torque_mesh.flatten()
                att_rpm_mesh, att_airmass_mesh = np.meshgrid(att_rpm_axis, att_airmass_axis)
                query_points = np.array([att_rpm_mesh.flatten(), att_airmass_mesh.flatten()]).T
                interpolated_torques_flat = griddata(source_points, source_values, query_points, method='linear')
                interpolated_tta_inv_table = interpolated_torques_flat.reshape(att_table.shape)
                interpolated_tta_inv_table = np.nan_to_num(interpolated_tta_inv_table, nan=0.0)

                results[tab_name] = {
                    'original_att': pd.DataFrame(att_table, index=att_airmass_axis, columns=att_rpm_axis),
                    'recommended_tta_inv': pd.DataFrame(
                        interpolated_tta_inv_table, index=att_airmass_axis, columns=att_rpm_axis
                    )
                }

            except Exception as e:
                warnings.append(f"Error processing '{tab_name}' after finding all maps: {e}")

    debug_logs.append(
        f" -> TTA/ATT Consistency Check complete. Found and processed {len(results)} complete table pairs.")

    if not results:
        final_warnings = warnings
        if not warnings:
            final_warnings.append(
                "No complete TTA/ATT table pairs were found. Check if maps are named correctly and have all 6 required components (TTA, ATT, and 4 axes)."
            )
        return {'status': 'Failed', 'warnings': final_warnings, 'debug_logs': debug_logs, 'results': {}}

    return {
        'status': 'Success',
        'warnings': warnings,
        'debug_logs': debug_logs,
        'results': results
    }
import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import sys
import re
from st_copy_button import st_copy_button

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports ---
from tuning_loader import TuningData
from TTA_Extrapolator import run_tta_extrapolation

# --- Constants (Copied from main app) ---
MAP_DEFINITIONS_CSV_PATH = "map_definitions.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'
XDF_SUBFOLDER = "XDFs"
PREDEFINED_FIRMWARES = ['S50', 'A05', 'V30', 'O30', 'LB6']
ALL_FIRMWARES = PREDEFINED_FIRMWARES + ['Other']

# --- Page Configuration ---
st.set_page_config(
    page_title="TTA Extrapolator",
    layout="wide"
)

st.title("📈 TTA Extrapolator")
st.markdown("Extrapolate Torque-to-Airflow (TTA) tables to a new maximum torque and regenerate the Airflow-to-Torque (ATT) tables.")

# --- Helper Functions (Duplicated from main app to avoid refactoring risk) ---
def display_table_with_copy_button(title: str, styled_df, raw_df: pd.DataFrame):
    """
    Displays a title, a styled DataFrame with its index, and a button to copy
    the raw data (without index/header) to the clipboard.
    """
    st.write(title)
    clipboard_text = raw_df.to_csv(sep='\t', index=False, header=False)
    st.dataframe(styled_df)

    # Unique key for button
    clean_title = re.sub(r'[^a-zA-Z0-9]', '', title)
    button_key = f"copy_btn_{clean_title}_{hash(title)}"

    st_copy_button(clipboard_text, f"📋 Copy Data", key=button_key)

def style_changed_cells(new_df: pd.DataFrame, old_df: pd.DataFrame, threshold=0.0):
    """
    Compares two DataFrames and returns a Styler object with changed cells highlighted.

    Args:
        new_df: The new data frame.
        old_df: The original data frame.
        threshold: Minimum relative change required to highlight a cell (default 0.0).
                   Example: 0.05 means only highlight if changed by >5%.
    """
    try:
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)

        # Check for shape mismatch. If shapes differ, we can't do cell-by-cell comparison easily.
        # But here we assume the underlying grid shape (rows x cols) matches, even if axis values changed.
        if new_df_c.shape != old_df_c.shape:
             # Fallback or just return unstyled if dimensions differ completely
            return new_df.style.format("{:.2f}")

        # Compare VALUES directly, ignoring index labels (which might have changed)
        new_vals = new_df_c.values
        old_vals = old_df_c.values

        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)
        increase_style = 'background-color: #2B442B'
        decrease_style = 'background-color: #442B2B'

        # Calculate relative difference (handling divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs((new_vals - old_vals) / old_vals)
            # If old value was 0, any non-zero new value is infinite change
            rel_diff[old_vals == 0] = np.inf

        # Determine which cells changed significantly
        # If threshold is 0, we just check for any inequality
        if threshold == 0:
             is_diff = ~np.isclose(new_vals, old_vals, rtol=1e-5)
        else:
             is_diff = rel_diff > threshold

        # Apply styles
        # Note: We must use iloc for positional assignment in style_df matching the numpy mask
        mask_inc = is_diff & (new_vals > old_vals)
        mask_dec = is_diff & (new_vals < old_vals)

        # Apply to style dataframe
        # We need to iterate or do a broadcast assignment.
        # Pandas style apply is cell-wise or column/row-wise.
        # Easier to just construct the style string matrix.

        # Construct the style array directly
        style_arr = np.full(new_vals.shape, '', dtype=object)
        style_arr[mask_inc] = increase_style
        style_arr[mask_dec] = decrease_style

        # Create a DataFrame from the style array with matching index/columns
        style_df = pd.DataFrame(style_arr, index=new_df.index, columns=new_df.columns)

        return new_df.style.apply(lambda x: style_df, axis=None).format("{:.2f}")
    except (ValueError, TypeError) as e:
        return new_df.style.format("{:.2f}")

@st.cache_resource(show_spinner=False)
def load_maps(bin_content, xdf_content, xdf_name, firmware_setting):
    """Loads maps from bin/xdf."""
    # Logic copied/adapted from load_all_maps_streamlit
    try:
        loader = TuningData(bin_content)
    except Exception as e:
        st.error(f"Failed to load binary: {e}")
        return None

    if xdf_content:
        tmp_xdf_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
                tmp_xdf.write(xdf_content)
                tmp_xdf_path = tmp_xdf.name

            loader.load_from_xdf(tmp_xdf_path, XDF_MAP_LIST_CSV)

        except Exception as e:
            st.error(f"Failed to parse XDF.")
            return None
        finally:
            if os.path.exists(tmp_xdf_path):
                os.remove(tmp_xdf_path)

    if firmware_setting != 'Other':
        try:
            firmware_col = f"address_{firmware_setting}"
            loader.load_from_manual_config(MAP_DEFINITIONS_CSV_PATH, firmware_col)
        except Exception as e:
            st.warning(f"Could not load manual definitions: {e}")

    return loader.maps


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    firmware = st.radio(
        "Firmware Version",
        options=ALL_FIRMWARES,
        horizontal=True,
        key="firmware"
    )

# --- Main UI ---
st.subheader("1. Upload & Configure")
col1, col2 = st.columns(2)

with col1:
    uploaded_bin_file = st.file_uploader("Upload .bin file", type=['bin', 'all'])

    uploaded_xdf_file = None
    if firmware == 'Other':
        uploaded_xdf_file = st.file_uploader("Upload .xdf file", type=['xdf'])

with col2:
    desired_max_tq = st.number_input(
        "Desired Max Torque",
        min_value=0.0,
        max_value=2000.0,
        value=500.0,
        step=10.0,
        help="The new maximum torque value for the last row of the TTA tables."
    )

if st.button("🚀 Run Extrapolation", type="primary", use_container_width=True):
    if not uploaded_bin_file:
        st.error("Please upload a BIN file.")
    elif firmware == 'Other' and not uploaded_xdf_file:
        st.error("Please upload an XDF file for 'Other' firmware.")
    else:
        with st.spinner("Loading maps..."):
            bin_content = uploaded_bin_file.getvalue()
            xdf_content = uploaded_xdf_file.getvalue() if uploaded_xdf_file else None
            xdf_name = uploaded_xdf_file.name if uploaded_xdf_file else None

            xdf_content_bytes = None
            if firmware in PREDEFINED_FIRMWARES:
                local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{firmware}.xdf")
                if os.path.exists(local_xdf_path):
                    with open(local_xdf_path, "rb") as f:
                        xdf_content_bytes = f.read()
            elif xdf_content:
                xdf_content_bytes = xdf_content

            all_maps = load_maps(bin_content, xdf_content_bytes, xdf_name, firmware)

        if all_maps:
            with st.spinner("Extrapolating..."):
                results = run_tta_extrapolation(all_maps, desired_max_tq)

            if results['status'] == 'Success':
                st.balloons()
                st.success("Extrapolation Complete!")

                # Iterate through results
                res_data = results['results']
                sorted_names = sorted(res_data.keys())

                if not sorted_names:
                    st.warning("No TTA/ATT pairs found in this file.")
                else:
                    # Display the New Torque Axis first (since it's shared)
                    # We pick the first result to show the axis
                    first_res = res_data[sorted_names[0]]
                    new_axis = first_res['new_torque_axis']
                    new_att_axis = first_res['new_att_airflow_axis']

                    col_ax1, col_ax2 = st.columns(2)
                    with col_ax1:
                        st.info(f"**New TTA Torque Axis (Last: {new_axis[-1]})**")
                        st.code('\t'.join(map(str, new_axis)), language=None)
                        st_copy_button('\t'.join(map(str, new_axis)), "📋 Copy TTA Axis", key="copy_axis_tta")

                    with col_ax2:
                        st.info(f"**New ATT Airflow Axis (Last: {new_att_axis[-1]:.2f})**")
                        st.code('\t'.join(map(str, new_att_axis)), language=None)
                        st_copy_button('\t'.join(map(str, new_att_axis)), "📋 Copy ATT Axis", key="copy_axis_att")

                    st.divider()

                    # Create tabs for each map pair
                    tabs = st.tabs(sorted_names)
                    for i, tab in enumerate(tabs):
                        with tab:
                            map_name = sorted_names[i]
                            pair_res = res_data[map_name]

                            if 'error' in pair_res:
                                st.error(f"Error processing {map_name}: {pair_res['error']}")
                                continue

                            col_tta, col_att = st.columns(2)

                            with col_tta:
                                st.subheader("TTA (Torque -> Airflow)")
                                styled_tta = style_changed_cells(pair_res['new_tta'], pair_res['original_tta'], threshold=0.0)
                                display_table_with_copy_button(f"New TTA: {map_name}", styled_tta, pair_res['new_tta'])

                            with col_att:
                                st.subheader("ATT (Airflow -> Torque)")
                                styled_att = style_changed_cells(pair_res['new_att'], pair_res['original_att'], threshold=0.05)
                                display_table_with_copy_button(f"New ATT: {map_name}", styled_att, pair_res['new_att'])
            else:
                st.error(results.get('message', "Analysis Failed"))

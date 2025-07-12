import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import re

# --- Import the custom tuning modules ---
from WG import run_wg_analysis
from LPFP import run_lpfp_analysis
from MAF import run_maf_analysis
from MFF import run_mff_analysis
from KNK import run_knk_analysis

from tuning_loader import TuningData

default_vars = "variables.csv"
MAP_DEFINITIONS_CSV_PATH = "map_definitions.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'

# --- Page Configuration
st.set_page_config(
    page_title="YAKtuner Online",
    layout="wide"
)

st.title("☁️ YAKtuner Online")

# --- 1. Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Tuner Settings")

    # --- Module Selection ---
    run_wg = st.checkbox("Tune Wastegate (WG)", value=True)  # Default to True for convenience
    run_maf = st.checkbox("Tune Mass Airflow (MAF)", value=False)
    run_mff = st.checkbox("Tune Mass Fuel Flow (MFF)", value=False)
    run_ign = st.checkbox("Tune Ignition (KNK)", value=False)
    run_lpfp = st.checkbox("Tune Low Pressure Pump Duty (LPFP)", value=False)

    st.divider()

    # --- Firmware Selection ---
    firmware = st.radio("Firmware Version", ('S50', 'A05', 'V30'), horizontal=True)

    st.divider()

    # --- WG Specific Settings ---
    if run_wg:
        st.subheader("WG Settings")
        use_swg_logic = st.checkbox("Use SWG Logic")

    # --- LPFP Specific Settings ---
    if run_lpfp:
        st.subheader("LPFP Settings")
        lpfp_drive = st.radio("2WD or 4WD", ('2WD', '4WD'))

    # --- KNK Settings Widgets (Refactored) ---
    if run_ign:
        ign_map_options = {
            "Stock (Base Correction)": 0, "SP Map 1": 1, "SP Map 2": 2, "SP Map 3": 3,
            "SP Map 4": 4, "SP Map 5": 5, "SP Flex Modifier": 6
        }
        selected_map_name = st.selectbox(
            "Ignition Map Selection", options=list(ign_map_options.keys())
        )
        ign_map = ign_map_options[selected_map_name]
        max_adv = st.slider("Max Advance", 0.0, 2.0, 0.75, 0.25)

# --- 2. Main Area for File Uploads ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Tune & Log Files")
    uploaded_bin_file = st.file_uploader("Upload .bin file", type=['bin', 'all'])
    uploaded_log_files = st.file_uploader("Upload .csv log files", type=['csv'], accept_multiple_files=True)

with col2:
    st.subheader("Configuration Files")
    uploaded_xdf_file = st.file_uploader("Upload .xdf file", type=['xdf'])
    uploaded_vars = st.file_uploader("Upload variables.csv", type=['csv'])


def normalize_header(header_name):
    """Normalizes a log file header for case-insensitive and unit-agnostic comparison."""
    normalized = re.sub(r'\s*\([^)]*\)\s*$', '', str(header_name))
    return normalized.lower().strip()


def map_log_variables_streamlit(log_df, varconv_df):
    """Performs automatic and interactive variable mapping in a Streamlit-native way."""
    if 'mapping_initialized' not in st.session_state:
        st.session_state.mapping_initialized = True
        st.session_state.mapping_complete = False
        st.session_state.vars_to_map = []
        st.session_state.varconv_array = varconv_df.to_numpy()
        st.session_state.log_df_mapped = log_df.copy()

        st.info("Performing automatic variable mapping...")
        varconv = st.session_state.varconv_array
        logvars = log_df.columns.tolist()
        missing_vars_indices = []

        for i in range(1, varconv.shape[1]):
            target_var_from_csv = varconv[0, i]
            normalized_target = normalize_header(target_var_from_csv)
            match_found = False
            for log_header in logvars:
                if normalize_header(log_header) == normalized_target:
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={log_header: varconv[1, i]}
                    )
                    st.session_state.varconv_array[0, i] = log_header
                    match_found = True
                    break
            if not match_found:
                missing_vars_indices.append(i)

        # Always store the result for the download feature
        st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)

        if not missing_vars_indices:
            st.session_state.mapping_complete = True
            st.success("All variables mapped automatically.")
        else:
            st.session_state.vars_to_map = missing_vars_indices
            st.rerun()

    if st.session_state.vars_to_map:
        varconv = st.session_state.varconv_array
        current_var_index = st.session_state.vars_to_map[0]

        if varconv.shape[0] > 2 and pd.notna(varconv[2, current_var_index]):
            prompt_name = varconv[2, current_var_index]
        else:
            prompt_name = varconv[1, current_var_index]

        st.warning(f"Could not find a match for: **{prompt_name}**")
        st.write(
            f"Please select the corresponding column from your log file. The last known name was `{varconv[0, current_var_index]}`.")

        with st.form(key=f"mapping_form_{current_var_index}"):
            log_headers = log_df.columns.tolist()
            options = ['[Not Logged]'] + log_headers
            selected_header = st.selectbox("Select Log File Column:", options=options)
            submitted = st.form_submit_button("Confirm Mapping")

            if submitted:
                if selected_header != '[Not Logged]':
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={selected_header: varconv[1, current_var_index]}
                    )
                    st.session_state.varconv_array[0, current_var_index] = selected_header

                st.session_state.vars_to_map.pop(0)

                # --- THE FIX ---
                # Check if we just mapped the last variable
                if not st.session_state.vars_to_map:
                    st.session_state.mapping_complete = True
                    # Save the final state for the download feature
                    st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)

                st.rerun()
        return None

    if st.session_state.mapping_complete:
        return st.session_state.log_df_mapped

    return None

@st.cache_resource(show_spinner="Loading tune files (XDF, BIN)...")
def load_all_maps_streamlit(bin_file, xdf_file, firmware_setting):
    """Loads all ECU maps from the uploaded binary and XDF files."""
    st.info("Loading tune data from binary file...")
    bin_content = bin_file.getvalue()
    try:
        loader = TuningData(bin_content)
    except Exception as e:
        st.error(f"Failed to initialize the binary file loader. Error: {e}")
        return None

    if xdf_file is not None:
        st.info(f"Parsing maps from XDF file: {xdf_file.name}...")
        tmp_xdf_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
                tmp_xdf.write(xdf_file.getvalue())
                tmp_xdf_path = tmp_xdf.name
            loader.load_from_xdf(tmp_xdf_path, XDF_MAP_LIST_CSV)
        except Exception as e:
            st.error(f"Failed to parse XDF file.")
            st.exception(e)
            return None
        finally:
            if os.path.exists(tmp_xdf_path):
                os.remove(tmp_xdf_path)
    else:
        st.warning("No XDF file provided. Relying solely on manual definitions.")

    st.info(f"Loading additional maps from '{MAP_DEFINITIONS_CSV_PATH}'...")
    try:
        firmware_col = f"address_{firmware_setting}"
        loader.load_from_manual_config(MAP_DEFINITIONS_CSV_PATH, firmware_col)
    except FileNotFoundError:
        st.error(f"'{MAP_DEFINITIONS_CSV_PATH}' not found.")
        return None
    except Exception as e:
        st.error(f"Failed to load manual map definitions. Error: {e}")
        return None

    all_maps = loader.maps
    if not all_maps:
        st.error("No maps were loaded. Check your XDF and manual configuration files.")
        return None

    st.success(f"Successfully loaded {len(all_maps)} maps into memory.")
    with st.expander("Click to view list of all loaded maps"):
        sorted_map_names = sorted(list(all_maps.keys()))
        num_columns = 3
        columns = st.columns(num_columns)
        maps_per_column = (len(sorted_map_names) + num_columns - 1) // num_columns
        for i in range(num_columns):
            with columns[i]:
                for map_name in sorted_map_names[i * maps_per_column:(i + 1) * maps_per_column]:
                    st.code(map_name, language=None)

    # Return the full dictionary of maps for the main script to use
    return all_maps

def style_changed_cells(new_df: pd.DataFrame, old_df: pd.DataFrame):
    """
    Compares two DataFrames and returns a Styler object with changed cells highlighted.
    - Green for an increase in value.
    - Red for a decrease in value.

    Args:
        new_df: The new DataFrame with recommended values.
        old_df: The original DataFrame to compare against.

    Returns:
        A pandas Styler object ready for display in Streamlit.
    """
    try:
        # Ensure data types are float for mathematical comparison
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)

        # Align frames to handle any shape differences, crucial for robust comparison
        old_df_aligned, new_df_aligned = old_df_c.align(new_df_c, join='outer', axis=None)

        # Initialize a style DataFrame with the same shape, filled with empty strings
        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)

        # Define the highlight styles for increase and decrease
        increase_style = 'background-color: #2B442B'  # Dark Green
        decrease_style = 'background-color: #442B2B'  # Dark Red

        # Apply styles to the style DataFrame based on direct comparison
        style_df[new_df_aligned > old_df_aligned] = increase_style
        style_df[new_df_aligned < old_df_aligned] = decrease_style

        # Apply the generated DataFrame of styles to the original DataFrame
        return new_df.style.apply(lambda x: style_df, axis=None)

    except (ValueError, TypeError):
        # If data can't be converted to float, return the unstyled DataFrame
        return new_df.style
# --- 3. Run Button and Logic ---
st.divider()

if st.button("🚀 Run YAKtuner Analysis", type="primary", use_container_width=True):
    st.session_state.run_analysis = True
    for key in ['mapping_initialized', 'mapping_complete', 'vars_to_map', 'varconv_array', 'log_df_mapped']:
        if key in st.session_state:
            del st.session_state[key]

if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    required_files = {"BIN file": uploaded_bin_file, "Log file(s)": uploaded_log_files}
    missing_files = [name for name, file in required_files.items() if not file]
    if missing_files:
        st.error(f"Please upload all required files. Missing: {', '.join(missing_files)}")
        st.session_state.run_analysis = False
        st.stop()

    # --- Phase 1: Interactive Variable Mapping ---
    # This phase is outside the main try/except so it can pause without triggering a premature state reset.
    try:
        log_df = pd.concat((pd.read_csv(f, encoding='latin1').iloc[:, :-1] for f in uploaded_log_files),
                           ignore_index=True)
        if uploaded_vars is not None:
            logvars_df = pd.read_csv(uploaded_vars, header=None)
        elif os.path.exists(default_vars):
            logvars_df = pd.read_csv(default_vars, header=None)
        else:
            st.error(f"Default '{default_vars}' not found. Please upload a variables.csv file.")
            st.stop()

        mapped_log_df = map_log_variables_streamlit(log_df, logvars_df)

        # If mapping is in progress, it returns None. We must stop to wait for user input.
        if mapped_log_df is None:
            st.stop()

    except Exception as e:
        st.error("An error occurred during file loading or variable mapping.")
        st.exception(e)
        st.session_state.run_analysis = False # Reset state on failure
        st.stop()


    # --- Phase 2: Main Analysis Pipeline ---
    # This code only runs if the mapping phase is fully complete.
    with st.spinner('Performing analysis... This may take a moment.'):
        try:
            # --- START: ADD THIS BLOCK ---
            # Display a success message and the download button for the updated variables
            if 'updated_varconv_df' in st.session_state:
                st.success("Variable mapping complete.")
                st.download_button(
                    label="📥 Download Updated variables.csv",
                    data=st.session_state.updated_varconv_df.to_csv(header=False, index=False).encode('utf-8'),
                    file_name='variables_updated.csv',
                    mime='text/csv',
                    help="Download the variable mapping file with your new selections for future use."
                )
                st.info("Proceeding to load tune maps...")
            else:
                 st.write("✅ Variable mapping complete. Proceeding to load tune maps...")
            all_maps = load_all_maps_streamlit(
                bin_file=uploaded_bin_file,
                xdf_file=uploaded_xdf_file,
                firmware_setting=firmware
            )

            if all_maps:
                # --- WG Tuning Module Integration ---
                if run_wg:
                    st.subheader("Wastegate (WG) Tuning Results")
                    # (The rest of the WG integration code remains the same)
                    try:
                        if use_swg_logic:
                            x_axis_key, y_axis_key = 'swgpid0_X', 'swgpid0_Y'
                            temp_comp_key, temp_comp_axis_key = 'tempcomp', 'tempcompaxis'
                        else:
                            x_axis_key, y_axis_key = 'wgpid0_X', 'wgpid0_Y'
                            temp_comp_key, temp_comp_axis_key = None, None
                        wg_x_axis = all_maps.get(x_axis_key)
                        wg_y_axis = all_maps.get(y_axis_key)
                        old_wg0_np = all_maps.get('wgpid0')
                        old_wg1_np = all_maps.get('wgpid1')
                        temp_comp_data = all_maps.get(temp_comp_key) if temp_comp_key else None
                        temp_comp_axis_data = all_maps.get(temp_comp_axis_key) if temp_comp_axis_key else None
                        essential_maps = {'wgpid0': old_wg0_np, 'wgpid1': old_wg1_np, x_axis_key: wg_x_axis,
                                          y_axis_key: wg_y_axis}
                        if use_swg_logic:
                            essential_maps[temp_comp_key] = temp_comp_data
                            essential_maps[temp_comp_axis_key] = temp_comp_axis_data
                        missing_essential = [name for name, data in essential_maps.items() if data is None]
                        if missing_essential:
                            st.error(
                                f"A required map for WG tuning is missing: {', '.join(missing_essential)}. Check your definitions.")
                            st.stop()
                        wg_results = run_wg_analysis(
                            log_df=mapped_log_df,
                            wgxaxis=wg_x_axis,
                            wgyaxis=wg_y_axis,
                            oldWG0=old_wg0_np,
                            oldWG1=old_wg1_np,
                            logvars=mapped_log_df.columns.tolist(),
                            WGlogic=use_swg_logic,
                            tempcomp=temp_comp_data,
                            tempcompaxis=temp_comp_axis_data
                        )
                        if wg_results['warnings']:
                            for warning in wg_results['warnings']:
                                st.warning(f"WG Analysis Warning: {warning}")
                        if wg_results['status'] == 'Success':
                            st.success("WG analysis completed successfully.")
                            res_vvl0 = wg_results['results_vvl0']
                            res_vvl1 = wg_results['results_vvl1']
                            scatter_plot = wg_results['scatter_plot_fig']
                            temp_comp = wg_results['temp_comp_results']
                            exh_labels = [str(x) for x in wg_x_axis]
                            int_labels = [str(y) for y in wg_y_axis]
                            original_vvl0_df = pd.DataFrame(old_wg0_np, index=int_labels, columns=exh_labels)
                            original_vvl1_df = pd.DataFrame(old_wg1_np, index=int_labels, columns=exh_labels)
                            styled_vvl0 = style_changed_cells(res_vvl0, original_vvl0_df)
                            styled_vvl1 = style_changed_cells(res_vvl1, original_vvl1_df)
                            tab1, tab2, tab3 = st.tabs(["📈 Recommended Tables", "📊 Scatter Plot", "🌡️ Temp Comp"])
                            with tab1:
                                st.write("#### Recommended WGPID0 (VVL0)")
                                st.dataframe(styled_vvl0)
                                st.write("#### Recommended WGPID1 (VVL1)")
                                st.dataframe(styled_vvl1)
                            with tab2:
                                if scatter_plot:
                                    st.pyplot(scatter_plot)
                                else:
                                    st.info("Scatter plot was not generated.")
                            with tab3:
                                if temp_comp is not None:
                                    st.write("#### Recommended Temperature Compensation")
                                    st.dataframe(temp_comp)
                                else:
                                    st.info("No temperature compensation adjustments were recommended.")
                        else:
                            st.error("WG analysis failed. Check warnings and console logs for details.")
                    except KeyError as e:
                        st.error(
                            f"A required map for WG tuning is missing: {e}. Check your XDF or manual definitions.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during WG tuning.")
                        st.exception(e)

                # --- MAF Tuning Module Integration ---
                if run_maf:
                    st.subheader("Mass Airflow (MAF) Tuning Results")
                    # (The rest of the MAF integration code remains the same)
                    try:
                        st.info("Loading MAF tables: maftable0-3, combmodes_MAF, and axes...")
                        maf_x_axis = all_maps.get('maftable0_X')
                        maf_y_axis = all_maps.get('maftable0_Y')
                        combmodes = all_maps.get('combmodes_MAF')
                        original_maf_tables = [all_maps.get(f'maftable{i}') for i in range(4)]
                        essential_maps = {
                            'maftable0_X': maf_x_axis, 'maftable0_Y': maf_y_axis, 'combmodes_MAF': combmodes,
                            'maftable0': original_maf_tables[0], 'maftable1': original_maf_tables[1],
                            'maftable2': original_maf_tables[2], 'maftable3': original_maf_tables[3],
                        }
                        missing_essential = [name for name, data in essential_maps.items() if data is None]
                        if missing_essential:
                            st.error(f"A required map for MAF tuning is missing: {', '.join(missing_essential)}. Check your definitions.")
                            st.stop()
                        maf_results = run_maf_analysis(
                            log=mapped_log_df, mafxaxis=maf_x_axis, mafyaxis=maf_y_axis,
                            maftables=original_maf_tables, combmodes_MAF=combmodes,
                            logvars=mapped_log_df.columns.tolist()
                        )
                        if maf_results['warnings']:
                            for warning in maf_results['warnings']:
                                st.warning(f"MAF Analysis Warning: {warning}")
                        if maf_results['status'] == 'Success':
                            st.success("MAF analysis completed successfully.")
                            recommended_maf_dfs = maf_results['results_maf']
                            tabs = st.tabs([f"📈 MAF Table IDX{i}" for i in range(4)])
                            for i, tab in enumerate(tabs):
                                with tab:
                                    original_df = pd.DataFrame(original_maf_tables[i], index=[str(y) for y in maf_y_axis], columns=[str(x) for x in maf_x_axis])
                                    recommended_df = recommended_maf_dfs[f'IDX{i}']
                                    styled_table = style_changed_cells(recommended_df, original_df)
                                    st.write(f"#### Recommended `maftable{i}`")
                                    st.dataframe(styled_table)
                        else:
                            st.error("MAF analysis failed. Check warnings for details.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during MAF tuning.")
                        st.exception(e)

                # --- MFF Tuning Module Integration ---
                if run_mff:
                    st.subheader("Multiplicative Fuel Factor (MFF) Tuning Results")
                    # (The rest of the MFF integration code remains the same)
                    try:
                        st.info("Loading MFF tables: MFFtable0-4, combmodes_MFF, and axes...")
                        mff_x_axis = all_maps.get('MFFtable0_X')
                        mff_y_axis = all_maps.get('MFFtable0_Y')
                        combmodes = all_maps.get('combmodes_MFF')
                        original_mff_tables = [all_maps.get(f'MFFtable{i}') for i in range(5)]
                        essential_maps = {
                            'MFFtable0_X': mff_x_axis, 'MFFtable0_Y': mff_y_axis, 'combmodes_MFF': combmodes,
                            'MFFtable0': original_mff_tables[0], 'MFFtable1': original_mff_tables[1],
                            'MFFtable2': original_mff_tables[2], 'MFFtable3': original_mff_tables[3],
                            'MFFtable4': original_mff_tables[4],
                        }
                        missing_essential = [name for name, data in essential_maps.items() if data is None]
                        if missing_essential:
                            st.error(f"A required map for MFF tuning is missing: {', '.join(missing_essential)}. Check your definitions.")
                            st.stop()
                        mff_results = run_mff_analysis(
                            log=mapped_log_df, mffxaxis=mff_x_axis, mffyaxis=mff_y_axis,
                            mfftables=original_mff_tables, combmodes_MFF=combmodes,
                            logvars=mapped_log_df.columns.tolist()
                        )
                        if mff_results['warnings']:
                            for warning in mff_results['warnings']:
                                st.warning(f"MFF Analysis Warning: {warning}")
                        if mff_results['status'] == 'Success':
                            st.success("MFF analysis completed successfully.")
                            recommended_mff_dfs = mff_results['results_mff']
                            tabs = st.tabs([f"📈 MFF Table IDX{i}" for i in range(5)])
                            for i, tab in enumerate(tabs):
                                with tab:
                                    original_df = pd.DataFrame(original_mff_tables[i], index=[str(y) for y in mff_y_axis], columns=[str(x) for x in mff_x_axis])
                                    recommended_df = recommended_mff_dfs[f'IDX{i}']
                                    styled_table = style_changed_cells(recommended_df, original_df)
                                    st.write(f"#### Recommended `MFFtable{i}`")
                                    st.dataframe(styled_table)
                        else:
                            st.error("MFF analysis failed. Check warnings for details.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during MFF tuning.")
                        st.exception(e)

                # --- KNK (Ignition) Tuning Module Integration ---
                if run_ign:
                    st.subheader("Ignition Timing (KNK) Tuning Results")
                    # (The rest of the KNK integration code remains the same)
                    try:
                        st.info("Loading ignition tables: igmap0-5, igxaxis, igyaxis...")
                        ign_x_axis = all_maps.get('igxaxis')
                        ign_y_axis = all_maps.get('igyaxis')
                        original_ign_tables = [all_maps.get(f'igmap{i}') for i in range(6)]
                        essential_maps = {
                            'igxaxis': ign_x_axis, 'igyaxis': ign_y_axis,
                            'igmap0': original_ign_tables[0]
                        }
                        missing_essential = [name for name, data in essential_maps.items() if data is None]
                        if missing_essential:
                            st.error(f"A required map for KNK tuning is missing: {', '.join(missing_essential)}. Check your definitions.")
                            st.stop()
                        knk_results = run_knk_analysis(
                            log=mapped_log_df, igxaxis=ign_x_axis, igyaxis=ign_y_axis,
                            IGNmaps=original_ign_tables, max_adv=max_adv, map_num=ign_map
                        )
                        if knk_results['warnings']:
                            for warning in knk_results['warnings']:
                                st.warning(f"KNK Analysis Warning: {warning}")
                        if knk_results['status'] == 'Success':
                            st.success("KNK analysis completed successfully.")
                            recommended_knk_df = knk_results['results_knk']
                            scatter_plot = knk_results['scatter_plot_fig']
                            base_map_np = knk_results['base_map']
                            tab1, tab2 = st.tabs(["📈 Recommended Table", "📊 Knock Scatter Plot"])
                            with tab1:
                                st.write(f"#### Recommended Ignition Table (Correcting `{selected_map_name}`)")
                                original_df = pd.DataFrame(base_map_np, index=[str(y) for y in ign_y_axis], columns=[str(x) for x in ign_x_axis])
                                styled_table = style_changed_cells(recommended_knk_df, original_df)
                                st.dataframe(styled_table)
                            with tab2:
                                if scatter_plot:
                                    st.pyplot(scatter_plot)
                                else:
                                    st.info("Scatter plot was not generated (no knock events found).")
                        else:
                            st.error("KNK analysis failed. Check warnings for details.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during KNK tuning.")
                        st.exception(e)

                # --- LPFP Tuning Module Integration ---
                if run_lpfp:
                    st.subheader("Low-Pressure Fuel Pump (LPFP) Tuning Results")
                    # (The rest of the LPFP integration code remains the same)
                    try:
                        table_key = 'lpfppwm' if lpfp_drive == '2WD' else 'lpfppwm4wd'
                        st.info(f"Using LPFP table: `{table_key}`")
                        lpfp_x_axis = all_maps.get('lpfppwm_X')
                        lpfp_y_axis = all_maps.get('lpfppwm_Y')
                        old_lpfp_table_np = all_maps.get(table_key)
                        essential_maps = {
                            'lpfppwm_X': lpfp_x_axis, 'lpfppwm_Y': lpfp_y_axis, table_key: old_lpfp_table_np
                        }
                        missing_essential = [name for name, data in essential_maps.items() if data is None]
                        if missing_essential:
                            st.error(f"A required map for LPFP tuning is missing: {', '.join(missing_essential)}. Check your definitions.")
                            st.stop()
                        lpfp_results = run_lpfp_analysis(
                            log=mapped_log_df, xaxis=lpfp_x_axis, yaxis=lpfp_y_axis,
                            old_table=old_lpfp_table_np, logvars=mapped_log_df.columns.tolist()
                        )
                        if lpfp_results['warnings']:
                            for warning in lpfp_results['warnings']:
                                st.warning(f"LPFP Analysis Warning: {warning}")
                        if lpfp_results['status'] == 'Success':
                            st.success("LPFP analysis completed successfully.")
                            recommended_lpfp_df = lpfp_results['results_lpfp']
                            original_lpfp_df = pd.DataFrame(old_lpfp_table_np, index=[str(y) for y in lpfp_y_axis], columns=[str(x) for x in lpfp_x_axis])
                            styled_lpfp_table = style_changed_cells(recommended_lpfp_df, original_lpfp_df)
                            st.write(f"#### Recommended {table_key.upper()} Table")
                            st.dataframe(styled_lpfp_table)
                        else:
                            st.error("LPFP analysis failed. Check warnings for details.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during LPFP tuning.")
                        st.exception(e)

                st.success("Full Analysis Complete!")
                st.balloons()

        except Exception as e:
            st.error("An unexpected error occurred during the main analysis pipeline.")
            st.exception(e)
        finally:
            # This now correctly resets the state only after a full run (success or failure).
            st.session_state.run_analysis = False
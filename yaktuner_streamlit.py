import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import re
import traceback
import sys
import difflib
from io import BytesIO

# --- Add project root to sys.path ---
# This is a robust way to ensure that local modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Import the custom tuning modules ---
from WG import run_wg_analysis
from LPFP import run_lpfp_analysis
from MAF import run_maf_analysis
from MFF import run_mff_analysis
from KNK import run_knk_analysis
from tuning_loader import TuningData
from error_reporter import send_to_google_sheets

# --- Constants ---
default_vars = "variables.csv"
MAP_DEFINITIONS_CSV_PATH = "map_definitions.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'
XDF_SUBFOLDER = "XDFs"
PREDEFINED_FIRMWARES = ['S50', 'A05', 'V30', 'O30', 'LB6']
ALL_FIRMWARES = PREDEFINED_FIRMWARES + ['Other']

# --- Page Configuration ---
st.set_page_config(
    page_title="YAKtuner Online",
    layout="wide"
)

st.title("☁️ YAKtuner Online")

# --- 1. Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Tuner Settings")

    # --- Module Selection ---
    run_wg = st.checkbox("Tune Wastegate (WG)", value=True, key="run_wg")
    run_maf = st.checkbox("Tune Mass Airflow (MAF)", value=False, key="run_maf")
    run_mff = st.checkbox("Tune Mass Fuel Flow (MFF)", value=False, key="run_mff")
    run_ign = st.checkbox("Tune Ignition (KNK)", value=False, key="run_ign")
    run_lpfp = st.checkbox("Tune Low Pressure Pump Duty (LPFP)", value=False, key="run_lpfp")

    st.divider()

    # --- Firmware Selection ---
    firmware = st.radio(
        "Firmware Version",
        options=ALL_FIRMWARES,
        horizontal=True,
        help="...",
        key="firmware" # Add key
    )

    st.divider()

    # --- Module-Specific Settings ---
    if run_wg:
        st.subheader("WG Settings")
        use_swg_logic = st.checkbox("Use SWG Logic", key="use_swg_logic")

    if run_lpfp:
        st.subheader("LPFP Settings")
        lpfp_drive = st.radio("2WD or 4WD", ('2WD', '4WD'), key="lpfp_drive")


    if run_ign:
        st.subheader("Ignition Settings")
        ign_map_options = {
            "Stock (Base Correction)": 0, "SP Map 1": 1, "SP Map 2": 2, "SP Map 3": 3,
            "SP Map 4": 4, "SP Map 5": 5, "SP Flex Modifier": 6
        }
        selected_map_name = st.selectbox(
            "Ignition Map Selection", options=list(ign_map_options.keys()), key="selected_ign_map"
        )
        ign_map = ign_map_options[selected_map_name]
        max_adv = st.slider("Max Advance", 0.0, 2.0, 0.75, 0.25, key="max_adv")

    st.divider()

    # --- Donation Link ---
    paypal_link = "https://www.paypal.com/donate/?hosted_button_id=MN43RKBR8AT6L"
    st.markdown(f"""
    <style>
        .paypal-button {{
            display: inline-block; padding: 8px 16px; font-size: 14px; font-weight: bold;
            color: #ffffff !important; background-color: #0070ba; border: none; border-radius: 5px;
            text-align: center; text-decoration: none; cursor: pointer; transition: background-color 0.3s;
        }}
        .paypal-button:hover {{
            background-color: #005ea6; color: #ffffff !important; text-decoration: none;
        }}
    </style>
    <div style="text-align: center; margin-top: 20px;">
        <a href="{paypal_link}" target="_blank" class="paypal-button">☕ Support YAKtuner</a>
    </div>
    """, unsafe_allow_html=True)

# --- 2. Main Area for File Uploads ---
st.subheader("1. Upload Tune & Log Files")
uploaded_bin_file = st.file_uploader("Upload .bin file", type=['bin', 'all'])
uploaded_log_files = st.file_uploader("Upload .csv log files", type=['csv'], accept_multiple_files=True)

uploaded_xdf_file = None
if firmware == 'Other':
    st.subheader("2. Upload Configuration File")
    st.info("Since you selected 'Other' firmware, you must provide an XDF file.")
    uploaded_xdf_file = st.file_uploader("Upload .xdf definition file", type=['xdf'])


# --- Helper Functions ---

def normalize_header(header_name):
    """Normalizes a log file header for case-insensitive and unit-agnostic comparison."""
    normalized = re.sub(r'\s*\([^)]*\)\s*$', '', str(header_name))
    return normalized.lower().strip()


def _find_alias_match(aliases, log_headers):
    """Finds an exact match for a list of aliases within the log headers."""
    normalized_aliases = {normalize_header(a) for a in aliases}
    for header in log_headers:
        if normalize_header(header) in normalized_aliases:
            return header
    return None


def _find_best_match(target_name, log_headers, cutoff=0.7):
    """Finds the best fuzzy match for a target name from a list of log headers."""
    normalized_target = normalize_header(target_name)
    normalized_headers = {h: normalize_header(h) for h in log_headers}
    search_space = list(normalized_headers.values())
    best_matches = difflib.get_close_matches(normalized_target, search_space, n=1, cutoff=cutoff)
    if best_matches:
        for original_header, normalized_header in normalized_headers.items():
            if normalized_header == best_matches[0]:
                return original_header
    return None


def map_log_variables_streamlit(log_df, varconv_df):
    """
    Performs a 3-tiered, robust, automatic, and interactive variable mapping.
    This function is designed to be called from within an st.status block.
    """
    if 'mapping_initialized' not in st.session_state:
        st.session_state.mapping_initialized = True
        st.session_state.mapping_complete = False
        st.session_state.vars_to_map = []
        st.session_state.varconv_array = varconv_df.to_numpy()
        st.session_state.log_df_mapped = log_df.copy()

        varconv = st.session_state.varconv_array
        log_headers = log_df.columns.tolist()
        missing_vars_indices = []

        for i in range(1, varconv.shape[1]):
            aliases_str = str(varconv[0, i])
            canonical_name = varconv[1, i]
            friendly_name = varconv[2, i] if varconv.shape[0] > 2 and pd.notna(varconv[2, i]) else canonical_name

            match_found = False
            best_match = None

            # Tier 1: Alias Matching
            aliases = aliases_str.split(',')
            alias_match = _find_alias_match(aliases, log_headers)
            if alias_match:
                best_match = alias_match
                match_found = True

            # Tier 2: Fuzzy Matching
            if not match_found:
                fuzzy_match = _find_best_match(friendly_name, log_headers)
                if fuzzy_match:
                    best_match = fuzzy_match
                    match_found = True

            if match_found:
                st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                    columns={best_match: canonical_name}
                )
                st.session_state.varconv_array[0, i] = best_match
            else:
                missing_vars_indices.append(i)

        st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)

        if not missing_vars_indices:
            st.session_state.mapping_complete = True
        else:
            st.session_state.vars_to_map = missing_vars_indices
            st.rerun()

    if st.session_state.vars_to_map:
        varconv = st.session_state.varconv_array
        current_var_index = st.session_state.vars_to_map[0]
        prompt_name = varconv[2, current_var_index] if varconv.shape[0] > 2 and pd.notna(
            varconv[2, current_var_index]) else varconv[1, current_var_index]

        st.warning(f"Could not find a match for: **{prompt_name}**")
        st.write(
            f"Please select the corresponding column from your log file. The last known name was `{varconv[0, current_var_index]}`.")

        with st.form(key=f"mapping_form_{current_var_index}"):
            options = ['[Not Logged]'] + log_df.columns.tolist()
            selected_header = st.selectbox("Select Log File Column:", options=options)
            submitted = st.form_submit_button("Confirm Mapping")

            if submitted:
                if selected_header != '[Not Logged]':
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={selected_header: varconv[1, current_var_index]}
                    )
                    st.session_state.varconv_array[0, current_var_index] = selected_header

                st.session_state.vars_to_map.pop(0)
                if not st.session_state.vars_to_map:
                    st.session_state.mapping_complete = True
                    st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)
                st.rerun()
        return None

    if st.session_state.mapping_complete:
        return st.session_state.log_df_mapped
    return None


@st.cache_resource(show_spinner=False)
def load_all_maps_streamlit(bin_content, xdf_content, xdf_name, firmware_setting):
    """Loads all ECU maps from file contents. Accepts bytes to be cache-friendly."""
    st.write("Loading tune data from binary file...")
    try:
        loader = TuningData(bin_content)
    except Exception as e:
        st.error(f"Failed to initialize the binary file loader. Error: {e}")
        return None

    if xdf_content is not None:
        st.write(f"Parsing maps from XDF: {xdf_name}...")
        tmp_xdf_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
                tmp_xdf.write(xdf_content)
                tmp_xdf_path = tmp_xdf.name
            loader.load_from_xdf(tmp_xdf_path, XDF_MAP_LIST_CSV)
        except Exception as e:
            st.error(f"Failed to parse XDF file.")
            st.exception(e)
            return None
        finally:
            if os.path.exists(tmp_xdf_path):
                os.remove(tmp_xdf_path)

    if firmware_setting != 'Other':
        st.write(f"Loading additional maps for '{firmware_setting}' from '{MAP_DEFINITIONS_CSV_PATH}'...")
        try:
            firmware_col = f"address_{firmware_setting}"
            loader.load_from_manual_config(MAP_DEFINITIONS_CSV_PATH, firmware_col)
        except FileNotFoundError:
            st.error(f"'{MAP_DEFINITIONS_CSV_PATH}' not found.")
            return None
        except KeyError:
            st.warning(f"Firmware column for '{firmware_setting}' not found in '{MAP_DEFINITIONS_CSV_PATH}'. Skipping.")
        except Exception as e:
            st.error(f"Failed to load manual map definitions. Error: {e}")
            return None

    all_maps = loader.maps
    if not all_maps:
        st.error("No maps were loaded. Check your XDF and manual configuration files.")
        return None

    st.write(f"Successfully loaded {len(all_maps)} maps into memory.")
    with st.expander("Click to view list of all loaded maps"):
        sorted_map_names = sorted(list(all_maps.keys()))
        num_columns = 3
        columns = st.columns(num_columns)
        maps_per_column = (len(sorted_map_names) + num_columns - 1) // num_columns
        for i in range(num_columns):
            with columns[i]:
                for map_name in sorted_map_names[i * maps_per_column:(i + 1) * maps_per_column]:
                    st.code(map_name, language=None)
    return all_maps


def style_changed_cells(new_df: pd.DataFrame, old_df: pd.DataFrame):
    """Compares two DataFrames and returns a Styler object with changed cells highlighted."""
    try:
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)
        old_df_aligned, new_df_aligned = old_df_c.align(new_df_c, join='outer', axis=None)
        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)
        increase_style = 'background-color: #2B442B'
        decrease_style = 'background-color: #442B2B'
        style_df[new_df_aligned > old_df_aligned] = increase_style
        style_df[new_df_aligned < old_df_aligned] = decrease_style
        return new_df.style.apply(lambda x: style_df, axis=None)
    except (ValueError, TypeError):
        return new_df.style

@st.cache_data(show_spinner="Running WG analysis...")
def cached_run_wg_analysis(*args, **kwargs):
    return run_wg_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running MAF analysis...")
def cached_run_maf_analysis(*args, **kwargs):
    return run_maf_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running MFF analysis...")
def cached_run_mff_analysis(*args, **kwargs):
    return run_mff_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running KNK analysis...")
def cached_run_knk_analysis(*args, **kwargs):
    return run_knk_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running LPFP analysis...")
def cached_run_lpfp_analysis(*args, **kwargs):
    return run_lpfp_analysis(*args, **kwargs)

# --- 3. Run Button and Logic ---
st.divider()

if st.button("🚀 Run YAKtuner Analysis", type="primary", use_container_width=True):
    st.session_state.run_analysis = True
    # Clear previous mapping state on a new run
    for key in ['mapping_initialized', 'mapping_complete', 'vars_to_map', 'varconv_array', 'log_df_mapped']:
        if key in st.session_state:
            del st.session_state[key]

if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    required_files = {"BIN file": uploaded_bin_file, "Log file(s)": uploaded_log_files}
    missing_files = [name for name, file in required_files.items() if not file]

    if missing_files:
        st.error(f"Please upload all required files. Missing: {', '.join(missing_files)}")
        st.session_state.run_analysis = False  # Reset state and allow the script to end gracefully
    else:
        # --- All files are present, proceed with analysis ---
        try:
            # Initialize all result dictionaries
            wg_results, maf_results, mff_results, knk_results, lpfp_results = None, None, None, None, None
            all_maps_data = {}  # To store map data for result display

            # --- Phase 1: Interactive Variable Mapping ---
            mapped_log_df = None
            with st.status("Mapping log variables...", expanded=True) as mapping_status:
                log_df = pd.concat((pd.read_csv(f, encoding='latin1').iloc[:, :-1] for f in uploaded_log_files),
                                   ignore_index=True)

                if not os.path.exists(default_vars):
                    raise FileNotFoundError(
                        f"Critical file missing: The default '{default_vars}' could not be found in the app directory.")

                logvars_df = pd.read_csv(default_vars, header=None)
                mapped_log_df = map_log_variables_streamlit(log_df, logvars_df)

                if mapped_log_df is not None:
                    mapping_status.update(label="Variable mapping complete.", state="complete", expanded=False)
                else:
                    # The mapping function is handling the UI and reruns, just wait.
                    mapping_status.update(label="Waiting for user input to map variables...", state="running", expanded=True)

            # --- This is the key change: The rest of the script only runs if mapping is complete ---
            if mapped_log_df is not None:
                # --- Phase 2: Main Analysis Pipeline ---
                with st.status("Starting YAKtuner analysis...", expanded=True) as status:
                    if 'updated_varconv_df' in st.session_state:
                        with st.expander("View Variable Mapping Results"):
                            varconv_array = st.session_state.updated_varconv_df.to_numpy()
                            mapping_summary_df = pd.DataFrame({
                                "Required Variable": varconv_array[2, 1:] if varconv_array.shape[0] > 2 else varconv_array[1, 1:],
                                "Matched Log Column": varconv_array[0, 1:],
                                "Internal App Name": varconv_array[1, 1:]
                            })
                            st.dataframe(mapping_summary_df, use_container_width=True)

                    status.update(label="Loading tune files...")
                    bin_content = uploaded_bin_file.getvalue()
                    xdf_content = None
                    xdf_name = None

                    if firmware in PREDEFINED_FIRMWARES:
                        local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{firmware}.xdf")
                        if os.path.exists(local_xdf_path):
                            with open(local_xdf_path, "rb") as f:
                                xdf_content = f.read()
                            xdf_name = os.path.basename(local_xdf_path)
                        else:
                            raise FileNotFoundError(f"The pre-packaged XDF for {firmware} was not found at '{local_xdf_path}'.")
                    elif firmware == 'Other':
                        if uploaded_xdf_file is None:
                            raise FileNotFoundError("Please upload an XDF file for the 'Other' firmware option.")
                        xdf_content = uploaded_xdf_file.getvalue()
                        xdf_name = uploaded_xdf_file.name

                    all_maps = load_all_maps_streamlit(
                        bin_content=bin_content, xdf_content=xdf_content, xdf_name=xdf_name, firmware_setting=firmware
                    )

                    if all_maps:
                        if run_wg:
                            with st.status("Running Wastegate (WG) analysis...", expanded=True) as module_status:
                                try:
                                    if use_swg_logic:
                                        x_axis_key, y_axis_key, temp_comp_key, temp_comp_axis_key = 'swgpid0_X', 'swgpid0_Y', 'tempcomp', 'tempcompaxis'
                                    else:
                                        x_axis_key, y_axis_key, temp_comp_key, temp_comp_axis_key = 'wgpid0_X', 'wgpid0_Y', None, None

                                    essential_keys = [x_axis_key, y_axis_key, 'wgpid0', 'wgpid1']
                                    if use_swg_logic: essential_keys.extend([temp_comp_key, temp_comp_axis_key])

                                    module_maps = {key: all_maps.get(key) for key in essential_keys if key}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(f"A required map is missing: {', '.join(missing)}")
                                    all_maps_data['wg'] = module_maps

                                    wg_results = cached_run_wg_analysis(
                                        log_df=mapped_log_df, wgxaxis=module_maps[x_axis_key], wgyaxis=module_maps[y_axis_key],
                                        oldWG0=module_maps['wgpid0'], oldWG1=module_maps['wgpid1'],
                                        logvars=mapped_log_df.columns.tolist(),
                                        WGlogic=use_swg_logic, tempcomp=module_maps.get(temp_comp_key),
                                        tempcompaxis=module_maps.get(temp_comp_axis_key)
                                    )
                                    if wg_results['status'] == 'Success':
                                        module_status.update(label="Wastegate (WG) analysis complete.", state="complete", expanded=False)
                                    else:
                                        st.error("WG analysis failed. Check warnings and console logs for details.")
                                        module_status.update(label="Wastegate (WG) analysis failed.", state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during WG tuning: {e}")
                                    module_status.update(label="Wastegate (WG) analysis failed.", state="error", expanded=True)

                        if run_maf:
                            with st.status("Running Mass Airflow (MAF) analysis...", expanded=True) as module_status:
                                try:
                                    keys = ['maftable0_X', 'maftable0_Y', 'combmodes_MAF'] + [f'maftable{i}' for i in range(4)]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(f"A required map is missing: {', '.join(missing)}")
                                    all_maps_data['maf'] = module_maps

                                    maf_results = cached_run_maf_analysis(
                                        log=mapped_log_df, mafxaxis=module_maps['maftable0_X'],
                                        mafyaxis=module_maps['maftable0_Y'],
                                        maftables=[module_maps[f'maftable{i}'] for i in range(4)],
                                        combmodes_MAF=module_maps['combmodes_MAF'],
                                        logvars=mapped_log_df.columns.tolist()
                                    )
                                    if maf_results['status'] == 'Success':
                                        module_status.update(label="Mass Airflow (MAF) analysis complete.", state="complete", expanded=False)
                                    else:
                                        st.error("MAF analysis failed. Check warnings for details.")
                                        module_status.update(label="Mass Airflow (MAF) analysis failed.", state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during MAF tuning: {e}")
                                    module_status.update(label="Mass Airflow (MAF) analysis failed.", state="error", expanded=True)

                        if run_mff:
                            with st.status("Running Fuel Factor (MFF) analysis...", expanded=True) as module_status:
                                try:
                                    keys = ['MFFtable0_X', 'MFFtable0_Y', 'combmodes_MFF'] + [f'MFFtable{i}' for i in range(5)]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(f"A required map is missing: {', '.join(missing)}")
                                    all_maps_data['mff'] = module_maps

                                    mff_results = cached_run_mff_analysis(
                                        log=mapped_log_df, mffxaxis=module_maps['MFFtable0_X'],
                                        mffyaxis=module_maps['MFFtable0_Y'],
                                        mfftables=[module_maps[f'MFFtable{i}'] for i in range(5)],
                                        combmodes_MFF=module_maps['combmodes_MFF'],
                                        logvars=mapped_log_df.columns.tolist()
                                    )
                                    if mff_results['status'] == 'Success':
                                        module_status.update(label="Fuel Factor (MFF) analysis complete.", state="complete", expanded=False)
                                    else:
                                        st.error("MFF analysis failed. Check warnings for details.")
                                        module_status.update(label="Fuel Factor (MFF) analysis failed.", state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during MFF tuning: {e}")
                                    module_status.update(label="Fuel Factor (MFF) analysis failed.", state="error", expanded=True)

                        if run_ign:
                            with st.status("Running Ignition (KNK) analysis...", expanded=True) as module_status:
                                try:
                                    keys = ['igxaxis', 'igyaxis'] + [f'igmap{i}' for i in range(6)]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None and key in ['igxaxis', 'igyaxis', 'igmap0']]
                                    if missing: raise KeyError(f"A required map for KNK tuning is missing: {', '.join(missing)}")
                                    all_maps_data['knk'] = module_maps

                                    knk_results = cached_run_knk_analysis(
                                        log=mapped_log_df, igxaxis=module_maps['igxaxis'],
                                        igyaxis=module_maps['igyaxis'],
                                        IGNmaps=[module_maps.get(f'igmap{i}') for i in range(6)], max_adv=max_adv,
                                        map_num=ign_map
                                    )
                                    if knk_results['status'] == 'Success':
                                        module_status.update(label="Ignition (KNK) analysis complete.", state="complete", expanded=False)
                                    else:
                                        st.error("KNK analysis failed. Check warnings for details.")
                                        module_status.update(label="Ignition (KNK) analysis failed.", state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during KNK tuning: {e}")
                                    module_status.update(label="Ignition (KNK) analysis failed.", state="error", expanded=True)

                        if run_lpfp:
                            with st.status("Running Fuel Pump (LPFP) analysis...", expanded=True) as module_status:
                                try:
                                    table_key = 'lpfppwm' if lpfp_drive == '2WD' else 'lpfppwm4wd'
                                    keys = ['lpfppwm_X', 'lpfppwm_Y', table_key]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing:
                                        raise KeyError(
                                            f"A required map for LPFP tuning is missing: {', '.join(missing)}")
                                    all_maps_data['lpfp'] = module_maps
                                    all_maps_data['lpfp']['table_key'] = table_key

                                    lpfp_results = cached_run_lpfp_analysis(
                                        log=mapped_log_df, xaxis=module_maps['lpfppwm_X'],
                                        yaxis=module_maps['lpfppwm_Y'],
                                        old_table=module_maps[table_key], logvars=mapped_log_df.columns.tolist()
                                    )
                                    if lpfp_results['status'] == 'Success':
                                        module_status.update(label="Fuel Pump (LPFP) analysis complete.",
                                                             state="complete", expanded=False)
                                    else:
                                        st.error("LPFP analysis failed. Check warnings for details.")
                                        module_status.update(label="Fuel Pump (LPFP) analysis failed.",
                                                             state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during LPFP tuning: {e}")
                                    module_status.update(label="Fuel Pump (LPFP) analysis failed.", state="error",
                                                         expanded=True)

                    status.update(label="Analysis complete!", state="complete", expanded=False)

                # On successful completion of the 'with' block, show balloons
                st.balloons()

                # --- Phase 3: Display All Results ---
                st.header("📈 Analysis Results")

                if wg_results and wg_results.get('status') == 'Success':
                    with st.expander("Wastegate (WG) Tuning Results", expanded=True):
                        if wg_results['warnings']:
                            for warning in wg_results['warnings']: st.warning(f"WG Analysis Warning: {warning}")

                        res_vvl0, res_vvl1 = wg_results['results_vvl0'], wg_results['results_vvl1']
                        scatter_plot, temp_comp = wg_results['scatter_plot_fig'], wg_results['temp_comp_results']
                        module_maps = all_maps_data['wg']
                        x_axis_key = 'swgpid0_X' if use_swg_logic else 'wgpid0_X'
                        y_axis_key = 'swgpid0_Y' if use_swg_logic else 'wgpid0_Y'

                        exh_labels = [str(x) for x in module_maps[x_axis_key]]
                        int_labels = [str(y) for y in module_maps[y_axis_key]]

                        original_vvl0_df = pd.DataFrame(module_maps['wgpid0'], index=int_labels, columns=exh_labels)
                        original_vvl1_df = pd.DataFrame(module_maps['wgpid1'], index=int_labels, columns=exh_labels)
                        styled_vvl0 = style_changed_cells(res_vvl0, original_vvl0_df)
                        styled_vvl1 = style_changed_cells(res_vvl1, original_vvl1_df)

                        tab1, tab2, tab3 = st.tabs(["📈 Recommended Tables", "📊 Scatter Plot", "🌡️ Temp Comp"])
                        with tab1:
                            st.write("#### Recommended WGPID0 (VVL0)");
                            st.dataframe(styled_vvl0)
                            st.write("#### Recommended WGPID1 (VVL1)");
                            st.dataframe(styled_vvl1)
                        with tab2:
                            if scatter_plot:
                                st.pyplot(scatter_plot)
                            else:
                                st.info("Scatter plot was not generated.")
                        with tab3:
                            if temp_comp is not None:
                                st.write("#### Recommended Temperature Compensation");
                                st.dataframe(temp_comp)
                            else:
                                st.info("No temperature compensation adjustments were recommended.")

                if maf_results and maf_results.get('status') == 'Success':
                    with st.expander("Mass Airflow (MAF) Tuning Results", expanded=True):
                        if maf_results['warnings']:
                            for warning in maf_results['warnings']: st.warning(f"MAF Analysis Warning: {warning}")
                        recommended_maf_dfs = maf_results['results_maf']
                        module_maps = all_maps_data['maf']
                        tabs = st.tabs([f"📈 MAF Table IDX{i}" for i in range(4)])
                        for i, tab in enumerate(tabs):
                            with tab:
                                original_df = pd.DataFrame(module_maps[f'maftable{i}'],
                                                           index=[str(y) for y in module_maps['maftable0_Y']],
                                                           columns=[str(x) for x in module_maps['maftable0_X']])
                                recommended_df = recommended_maf_dfs[f'IDX{i}']
                                styled_table = style_changed_cells(recommended_df, original_df)
                                st.write(f"#### Recommended `maftable{i}`");
                                st.dataframe(styled_table)

                if mff_results and mff_results.get('status') == 'Success':
                    with st.expander("Multiplicative Fuel Factor (MFF) Tuning Results", expanded=True):
                        if mff_results['warnings']:
                            for warning in mff_results['warnings']: st.warning(f"MFF Analysis Warning: {warning}")
                        recommended_mff_dfs = mff_results['results_mff']
                        module_maps = all_maps_data['mff']
                        tabs = st.tabs([f"📈 MFF Table IDX{i}" for i in range(5)])
                        for i, tab in enumerate(tabs):
                            with tab:
                                original_df = pd.DataFrame(module_maps[f'MFFtable{i}'],
                                                           index=[str(y) for y in module_maps['MFFtable0_Y']],
                                                           columns=[str(x) for x in module_maps['MFFtable0_X']])
                                recommended_df = recommended_mff_dfs[f'IDX{i}']
                                styled_table = style_changed_cells(recommended_df, original_df)
                                st.write(f"#### Recommended `MFFtable{i}`");
                                st.dataframe(styled_table)

                if knk_results and knk_results.get('status') == 'Success':
                    with st.expander("Ignition Timing (KNK) Tuning Results", expanded=True):
                        if knk_results['warnings']:
                            for warning in knk_results['warnings']: st.warning(f"KNK Analysis Warning: {warning}")
                        recommended_knk_df, scatter_plot, base_map_np = knk_results['results_knk'], knk_results[
                            'scatter_plot_fig'], knk_results['base_map']
                        module_maps = all_maps_data['knk']
                        tab1, tab2 = st.tabs(["📈 Recommended Table", "📊 Knock Scatter Plot"])
                        with tab1:
                            st.write(f"#### Recommended Ignition Table (Correcting `{selected_map_name}`)")
                            original_df = pd.DataFrame(base_map_np,
                                                       index=[str(y) for y in module_maps['igyaxis']],
                                                       columns=[str(x) for x in module_maps['igxaxis']])
                            styled_table = style_changed_cells(recommended_knk_df, original_df)
                            st.dataframe(styled_table)
                        with tab2:
                            if scatter_plot:
                                st.pyplot(scatter_plot)
                            else:
                                st.info("Scatter plot was not generated (no knock events found).")

                if lpfp_results and lpfp_results.get('status') == 'Success':
                    with st.expander("Low-Pressure Fuel Pump (LPFP) Tuning Results", expanded=True):
                        if lpfp_results['warnings']:
                            for warning in lpfp_results['warnings']: st.warning(f"LPFP Analysis Warning: {warning}")
                        recommended_lpfp_df = lpfp_results['results_lpfp']
                        module_maps = all_maps_data['lpfp']
                        table_key = module_maps['table_key']
                        original_lpfp_df = pd.DataFrame(module_maps[table_key],
                                                        index=[str(y) for y in module_maps['lpfppwm_Y']],
                                                        columns=[str(x) for x in module_maps['lpfppwm_X']])
                        styled_lpfp_table = style_changed_cells(recommended_lpfp_df, original_lpfp_df)
                        st.write(f"#### Recommended {table_key.upper()} Table")
                        st.dataframe(styled_lpfp_table)

                # --- ADDED ---
                # After a successful run, reset the flag to prevent re-running on the next interaction.
                st.session_state.run_analysis = False

        except Exception as e:
            st.error(f"An unexpected error occurred during the analysis: {e}")
            st.write("You can help improve YAKtuner by sending this error report to the developer.")
            traceback_str = traceback.format_exc()

            with st.form(key="error_report_form"):
                st.write("**An unexpected error occurred.** You can help by sending this report.")
                user_description = st.text_area(
                    "Optional: Please describe what you were doing when the error occurred."
                )
                user_contact = st.text_input(
                    "Optional: Email or username for follow-up questions."
                )
                st.text_area(
                    "Technical Error Details (for submission)",
                    value=traceback_str,
                    height=200,
                    disabled=True
                )
                submit_button = st.form_submit_button("Submit Error Report")

                if submit_button:
                    with st.spinner("Sending report..."):
                        success, message = send_to_google_sheets(traceback_str, user_description, user_contact)
                        if success:
                            st.success("Thank you! Your error report has been sent.")
                        else:
                            st.error(f"Sorry, the report could not be sent. Reason: {message}")
                            st.error("Please copy the details below and report it manually.")

            with st.expander("Click to view technical error details"):
                st.code(traceback_str, language=None)

            # --- ADDED ---
            # After an unsuccessful run, also reset the flag.
            st.session_state.run_analysis = False
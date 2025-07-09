"""
YAKtuner - Main Application Script

This script serves as the entry point for the YAKtuner application. It performs the following steps:
1.  Displays a GUI to gather user settings for the tuning process (e.g., which modules to run,
    firmware version, output options).
2.  Prompts the user to select one or more log files (.csv) and a binary tune file (.bin).
3.  Loads a central map configuration file (`map_definitions.csv`) that defines the memory
    addresses and parameters for all readable ECU maps.
4.  Dynamically renames columns in the log data based on a user-configurable `variables.csv`,
    prompting for manual mapping if a variable is not found.
5.  Calls the refactored `BinRead` module to read all required map data from the binary file in
    a single, efficient operation.
6.  Passes the log data and the extracted map data to the appropriate tuning modules (WG, MAF, KNK)
    based on the user's initial selections.
7.  Optionally saves the results from the tuning modules to new CSV files.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Listbox
import customtkinter as ctk
import numpy as np
import pandas as pd
import re
import pprint

# Import the custom tuning modules
import KNK
import MAF
import WG
import LPFP
import MFF
from tuning_loader import TuningData

# --- Constants ---
# Define constants for configuration filenames to avoid "magic strings".
VARIABLES_CSV_PATH = "variables.csv"
MAP_DEFINITIONS_CSV_PATH = "map_definitions.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'


def setup_and_run_gui(parent):
    """
    Creates and displays the main GUI for gathering user settings.
    """
    # This line now correctly uses the 'parent' argument that is passed in.
    settings_window = ctk.CTkToplevel(parent)
    settings_window.title("YakTuner Settings")

    # --- Make the window modal to pause the main script ---
    settings_window.transient(parent)
    settings_window.grab_set()

    settings = {}


    # --- Main Frames for Organization ---
    tune_frame = ctk.CTkFrame(settings_window)
    tune_frame.pack(pady=10, padx=10, fill="x")

    knk_settings_frame = ctk.CTkFrame(settings_window, fg_color="transparent")
    knk_settings_frame.pack(pady=(0, 10), padx=10, fill="x")

    firmware_frame = ctk.CTkFrame(settings_window)
    firmware_frame.pack(pady=10, padx=10, fill="x")

    options_frame = ctk.CTkFrame(settings_window)
    options_frame.pack(pady=10, padx=10, fill="x")
    # --- Tuning Module Section ---
    tune_label = ctk.CTkLabel(tune_frame, text="Tuning Modules", font=ctk.CTkFont(weight="bold"))
    tune_label.grid(row=0, column=0, columnspan=5, sticky="w", padx=5, pady=(5, 10))

    # Define variables for checkboxes
    wg_tuner_var = ctk.BooleanVar()
    maf_tuner_var = ctk.BooleanVar()
    mff_tuner_var = ctk.BooleanVar()
    ign_tuner_var = ctk.BooleanVar()
    lpfp_tuner_var = ctk.BooleanVar()
    swg_logic_var = ctk.BooleanVar()

    cbx_wg = ctk.CTkCheckBox(tune_frame, text="Tune WG?", variable=wg_tuner_var)
    cbx_wg.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    cbx_maf = ctk.CTkCheckBox(tune_frame, text="Tune MAF?", variable=maf_tuner_var)
    cbx_maf.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    cbx_mff = ctk.CTkCheckBox(tune_frame, text="Tune MFF?", variable=mff_tuner_var)
    cbx_mff.grid(row=1, column=2, padx=10, pady=5, sticky="w")
    cbx_ign = ctk.CTkCheckBox(tune_frame, text="Tune Ignition?", variable=ign_tuner_var)
    cbx_ign.grid(row=1, column=3, padx=10, pady=5, sticky="w")
    cbx_lpfp = ctk.CTkCheckBox(tune_frame, text="Tune LPFP PWM?", variable=lpfp_tuner_var)
    cbx_lpfp.grid(row=1, column=4, padx=10, pady=5, sticky="w")

    # SWG checkbox, dependent on the WG checkbox
    cbx_swg = ctk.CTkCheckBox(tune_frame, text="Use SWG Logic?", variable=swg_logic_var, state="disabled")
    cbx_swg.grid(row=2, column=0, padx=10, pady=5, sticky="w")

    # LPFP drive type widgets
    lpfp_drive_type_var = ctk.StringVar(value="2WD")
    rb_2wd = ctk.CTkRadioButton(tune_frame, text="2WD", variable=lpfp_drive_type_var, value="2WD", state="disabled")
    rb_2wd.grid(row=2, column=4, padx=10, pady=5, sticky="w")
    rb_4wd = ctk.CTkRadioButton(tune_frame, text="4WD", variable=lpfp_drive_type_var, value="4WD", state="disabled")
    rb_4wd.grid(row=3, column=4, padx=10, pady=5, sticky="w")

    # --- KNK Settings Widgets (Refactored) ---
    knk_label = ctk.CTkLabel(knk_settings_frame, text="Ignition Tuner Settings", font=ctk.CTkFont(weight="bold"))
    knk_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5))

    # Max Advance setting
    max_adv_label = ctk.CTkLabel(knk_settings_frame, text="Max Advance:")
    max_adv_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
    max_adv_var = ctk.StringVar(value="0.75")
    max_adv_entry = ctk.CTkEntry(knk_settings_frame, textvariable=max_adv_var, width=100)
    max_adv_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)

    # SP Map Number setting
    map_num_label = ctk.CTkLabel(knk_settings_frame, text="SP Map (0=None, 6=Flex):")
    map_num_label.grid(row=1, column=2, sticky='w', padx=15, pady=2)
    map_num_var = ctk.StringVar(value="6")
    map_num_entry = ctk.CTkEntry(knk_settings_frame, textvariable=map_num_var, width=100)
    map_num_entry.grid(row=1, column=3, sticky='w', padx=5, pady=2)

    # --- Toggle Functions for UI Interactivity ---
    def toggle_swg_checkbox():
        cbx_swg.configure(state="normal" if wg_tuner_var.get() else "disabled")
        if not wg_tuner_var.get():
            swg_logic_var.set(False)

    def toggle_lpfp_options():
        state = "normal" if lpfp_tuner_var.get() else "disabled"
        rb_2wd.configure(state=state)
        rb_4wd.configure(state=state)

    def toggle_knk_settings():
        state = "normal" if ign_tuner_var.get() else "disabled"
        for child in knk_settings_frame.winfo_children():
            child.configure(state=state)

    # Link the toggle functions to their respective checkboxes
    cbx_wg.configure(command=toggle_swg_checkbox)
    cbx_lpfp.configure(command=toggle_lpfp_options)
    cbx_ign.configure(command=toggle_knk_settings)

    # Set initial state of disabled widgets
    toggle_knk_settings()
    toggle_lpfp_options()

    # --- Firmware Section ---
    firmware_label = ctk.CTkLabel(firmware_frame, text="Firmware Version", font=ctk.CTkFont(weight="bold"))
    firmware_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 0))

    firmware_var = ctk.StringVar(value="S50")
    rb_s50 = ctk.CTkRadioButton(firmware_frame, text="S50", variable=firmware_var, value="S50")
    rb_s50.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    rb_a05 = ctk.CTkRadioButton(firmware_frame, text="A05", variable=firmware_var, value="A05")
    rb_a05.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    rb_v30 = ctk.CTkRadioButton(firmware_frame, text="V30", variable=firmware_var, value="V30")
    rb_v30.grid(row=1, column=2, padx=10, pady=5, sticky="w")

    # --- Other Options Section ---
    options_label = ctk.CTkLabel(options_frame, text="Other Options", font=ctk.CTkFont(weight="bold"))
    options_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))

    cbx_save = ctk.CTkCheckBox(options_frame, text="Save Results to CSV?")
    cbx_save.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    cbx_reset_vars = ctk.CTkCheckBox(options_frame, text="Reset Variable Mappings?")
    cbx_reset_vars.grid(row=1, column=1, padx=10, pady=5, sticky="w")

    def on_continue():
        nonlocal settings
        settings = {
            'WGtune': wg_tuner_var.get(),
            'MAFtune': maf_tuner_var.get(),
            'MFFtune': mff_tuner_var.get(),
            'IGtune': ign_tuner_var.get(),
            'LPFPtune': lpfp_tuner_var.get(),
            'save': cbx_save.get(),
            'WGlogic': swg_logic_var.get(),
            'firmware': firmware_var.get(),
            'LPFPdrivetype': lpfp_drive_type_var.get(),
            'var_reset': cbx_reset_vars.get(),
            'knk_max_adv': max_adv_var.get(),
            'knk_map_num': map_num_var.get(),
            'did_continue': True
        }
        settings_window.destroy()

    continue_btn = ctk.CTkButton(settings_window, text="CONTINUE", command=on_continue)
    continue_btn.pack(pady=20)

    parent.wait_window(settings_window)

    return settings


def load_data_files():
    """
    Prompts the user to select log files and a binary tune file.

    Returns:
        tuple: A tuple containing (log_dataframe, bin_file_path, list_of_log_paths).
               Returns (None, None, None) if the user cancels any dialog.
    """
    # --- Load Log Files ---
    log_paths = filedialog.askopenfilenames(
        title="Select Log CSV File(s)",
        filetypes=[("CSV files", "*.csv")]
    )
    if not log_paths:
        return None, None, None

    # Concatenate all selected log files into a single DataFrame.
    try:
        log_list = [pd.read_csv(f, encoding='latin1').iloc[:, :-1] for f in log_paths]
        log_df = pd.concat(log_list, ignore_index=True)
    except Exception as e:
        messagebox.showerror("Log File Error", f"Could not read or process log files.\nError: {e}")
        return None, None, None

    # --- Select Bin File ---
    bin_path = filedialog.askopenfilename(
        title="Select Bin File",
        filetypes=[("Binary files", "*.bin")]
    )
    if not bin_path:
        return None, None, None

    # --- Select Bin File ---
    xdf_path = filedialog.askopenfilename(
        title="Select XDF File",
        filetypes=[("Definition files", "*.xdf")]
    )
    if not xdf_path:
        return None, None, None

    return log_df, bin_path, log_paths, xdf_path


def normalize_header(header_name):
    """
    Normalizes a log file header for case-insensitive and unit-agnostic comparison.
    - Converts to lowercase.
    - Removes parenthetical units at the end of the string, e.g., "(kpa)".
    - Strips leading/trailing whitespace.
    """
    # Remove content in parentheses (like units) from the end of the string.
    normalized = re.sub(r'\s*\([^)]*\)\s*$', '', str(header_name))
    # Convert to lowercase and strip whitespace for a clean comparison.
    return normalized.lower().strip()


def map_log_variables(log_df, varconv_df, var_reset):
    """
    Renames columns in the log DataFrame based on the `variables.csv` mapping.
    This version is case-insensitive and ignores parenthetical units in headers.
    If a variable is not found, it interactively prompts the user to map it.

    Args:
        log_df (pd.DataFrame): The DataFrame loaded from log files.
        varconv_df (pd.DataFrame): The DataFrame loaded from `variables.csv`.
        var_reset (bool): This parameter is kept for signature compatibility but is no longer
                        used, as the new logic is superior to the old "reset" behavior.

    Returns:
        pd.DataFrame or None: The log DataFrame with columns renamed, or None on critical error.
    """
    # Add a validation check to ensure the essential first two rows exist.
    if varconv_df.shape[0] < 2:
        messagebox.showerror(
            "Configuration Error",
            f"The '{VARIABLES_CSV_PATH}' file is invalid.\n\n"
            "It must contain at least two rows:\n"
            "Row 1: The variable name from your log file (e.g., 'RPM (1/min)').\n"
            "Row 2: The internal standardized name (e.g., 'RPM').\n\n"
            "Please correct the file and restart the application."
        )
        return None  # Return None to signal a critical failure

    varconv = varconv_df.to_numpy()
    logvars = log_df.columns.tolist()

    # This list will hold the indices of variables that could not be found automatically.
    missing_vars_indices = []

    # --- REVISED LOGIC ---
    # The old 'if var_reset:' logic forced a remap of all variables.
    # This new, simpler logic always attempts to find a match first, and only
    # prompts the user for variables that are truly missing from the logs.

    # Loop through each variable required by the application (from variables.csv).
    for i in range(1, varconv.shape[1]):
        # The target name from the first row of variables.csv (e.g., "RPM (1/min)")
        target_var_from_csv = varconv[0, i]
        # A normalized version for robust matching (e.g., "rpm")
        normalized_target = normalize_header(target_var_from_csv)

        match_found = False

        # Loop through the actual headers in the loaded log file to find a match.
        for log_header in logvars:
            normalized_log_header = normalize_header(log_header)

            # Compare the normalized versions for a flexible, case-insensitive match.
            if normalized_log_header == normalized_target:
                # Found a match! Rename the column to the standardized internal name.
                log_df = log_df.rename(columns={log_header: varconv[1, i]})

                # Update the mapping array with the exact header we found.
                # This "self-healing" makes future runs more accurate.
                varconv[0, i] = log_header

                match_found = True
                break  # Match found, move to the next required variable.

        if not match_found:
            # If we looped through all log headers and found no match,
            # add this variable's index to the list to be manually mapped.
            missing_vars_indices.append(i)

    # If the list of missing variables is not empty, prompt the user for each one.
    if missing_vars_indices:
        for i in missing_vars_indices:
            # Use a dictionary to store the result from the dialog callback.
            selection_result = {'selected_var': None}

            # Determine the title for the dialog window.
            # Use the descriptive name from the 3rd row if it exists and is not empty.
            if varconv.shape[0] > 2 and pd.notna(varconv[2, i]):
                window_title_var_name = varconv[2, i]
            else:
                # Otherwise, fall back to the internal variable name from the 2nd row.
                window_title_var_name = varconv[1, i]

            # Create a new Toplevel window for the prompt.
            var_window = Toplevel()
            var_window.title(f"Select variable for: {window_title_var_name}")
            var_window.geometry("400x400")

            listbox = Listbox(var_window, selectmode='single')
            listbox.pack(fill='both', expand=True)

            # Populate listbox with default, "Not Logged", and available log variables.
            options = [varconv[0, i]] + ['Not Logged'] + logvars
            for opt in options:
                listbox.insert(tk.END, opt)

            def on_select(event=None):
                selection = listbox.curselection()
                if selection and selection[0] > 1:  # 0 is default, 1 is "Not Logged"
                    selection_result['selected_var'] = logvars[selection[0] - 2]
                var_window.destroy()

            tk.Button(var_window, text="Select", command=on_select).pack()
            var_window.wait_window()  # Wait for the dialog to close.

            # If a new variable was selected, update the mapping and rename the column.
            selected_var = selection_result['selected_var']
            if selected_var:
                varconv[0, i] = selected_var
                log_df = log_df.rename(columns={selected_var: varconv[1, i]})

        # Save the updated mappings back to variables.csv for future use.
        pd.DataFrame(varconv).to_csv(VARIABLES_CSV_PATH, header=False, index=False)

    return log_df

def main():
    """
    The main execution function of the application.
    """

    print("--- YAKtuner Application Started ---")
    print("Awaiting user settings from GUI...")

    root = ctk.CTk()
    root.withdraw()

    # 1. Get user settings from the GUI.
    settings = setup_and_run_gui(root)
    if not settings.get('did_continue'):
        print("Operation cancelled by user.")
        root.destroy()  # Clean up the hidden root
        return

    print("[STATUS] User settings collected successfully.")
    print("\n--- Step 1: Loading Configuration Files ---")

    # 2. Load map definitions and variable configurations.
    try:
        # Specify 'latin1' encoding to handle non-UTF-8 characters.
        map_definitions = pd.read_csv(MAP_DEFINITIONS_CSV_PATH, encoding='latin1')

        logvars_df = pd.read_csv(VARIABLES_CSV_PATH, encoding='latin1', header=None)
        print(f"[OK] Loaded '{MAP_DEFINITIONS_CSV_PATH}' and '{VARIABLES_CSV_PATH}'.")

    except FileNotFoundError as e:
        messagebox.showerror("Configuration Error", f"A required CSV file is missing: {e}")
        return
    except Exception as e:
        # Catch other potential reading errors.
        messagebox.showerror("Configuration File Error", f"Could not read a config file.\nError: {e}")
        return

    print("\n--- Step 2: Awaiting User File Selection ---")

    # 3. Get paths for the binary and log files.
    log_df, bin_path, log_paths, xdf_path = load_data_files()
    if log_df is None:
        # User cancelled one of the file dialogs.
        print("File selection cancelled. Exiting.")
        return

    print(f"[OK] Loaded {len(log_paths)} log file(s) and selected tune files.")
    print("\n--- Step 3: Mapping Log Variables ---")

    # 4. Map log variables to standard names, prompting user if needed.
    log_df = map_log_variables(log_df, logvars_df, settings['var_reset'])
    if log_df is None:
        # This check handles critical errors from the mapping function.
        print("Halting execution due to an error in variable mapping.")
        return
    print("[OK] Log variables mapped successfully.")
    logvars = log_df.columns.tolist()

    # Determine the correct firmware column to use for map addresses.
    print("\n--- Step 4: Loading Tune Data from Binary ---")
    firmware_col = f"address_{settings['firmware']}"

    if not os.path.exists(bin_path):
        print(f"FATAL: Binary file '{bin_path}' not found. Exiting.")
        return # Or show an error in the GUI

    # ===========================================
    # Initialize the loader with the binary file
    # ===========================================
    try:
        loader = TuningData(bin_path)
    except FileNotFoundError as e:
        print(e)
        return # Or show an error in the GUI

    # ===========================================
    # 2. Load maps from the XDF file first
    # ===========================================
    loader.load_from_xdf(xdf_path, XDF_MAP_LIST_CSV)

    # ===========================================
    # 3. Load any additional maps from the manual config
    # ===========================================
    # This will also overwrite any XDF maps if they share a name.
    # You could add a checkbox in your GUI to enable/disable this.
    loader.load_from_manual_config(MAP_DEFINITIONS_CSV_PATH, firmware_col)

    all_maps = loader.maps

    # Unpack the returned dictionary into variables for the tuning modules.
    try:
        maftables = [all_maps['maftable0'], all_maps['maftable1'], all_maps['maftable2'], all_maps['maftable3']]
        mfftables = [all_maps['MFFtable0'], all_maps['MFFtable1'], all_maps['MFFtable2'], all_maps['MFFtable3'],
                     all_maps['MFFtable4']]
        IGNmaps = [all_maps['igmap0'], all_maps['igmap1'], all_maps['igmap2'], all_maps['igmap3'],
                   all_maps['igmap4'], all_maps['igmap5']]
    except KeyError as e:
        messagebox.showerror("Map Definition Error",
                             f"A required map is missing: {e}. Please check '{MAP_DEFINITIONS_CSV_PATH}'.")
        return


    print("\n--- Step 5: Running Selected Tuning Modules ---")

    # --- 5. Run Selected Tuning Modules ---
    if settings['WGtune']:
        print("\n[MODULE] Running Wastegate (WG) Tuner...")
        try:
            temp_comp_map = None
            temp_comp_axis = None
            if settings['WGlogic']:
                # Only attempt to access these maps if SWG logic is enabled.
                temp_comp_map = all_maps['tempcomp']
                temp_comp_axis = all_maps['tempcompaxis']

            Res_WG1, Res_WG0 = WG.WG_tune(
                log_df,
                # Pass the correct X and Y axes based on WGlogic setting
                all_maps['swgpid0_X'] if settings['WGlogic'] else all_maps['wgpid0_X'],
                all_maps['swgpid0_Y'] if settings['WGlogic'] else all_maps['wgpid0_Y'],
                all_maps['wgpid0'],
                all_maps['wgpid1'],
                logvars,
                True,  # Placeholder for a 'plot' setting
                settings['WGlogic'],
                # Pass the potentially None maps to the tuner function
                temp_comp_map,
                temp_comp_axis,
                root
            )
            if settings['save'] and Res_WG1 is not None and Res_WG0 is not None:
                output_dir = os.path.dirname(log_paths[0])
                Res_WG1.to_csv(os.path.join(output_dir, "WG1_Results.csv"))
                Res_WG0.to_csv(os.path.join(output_dir, "WG0_Results.csv"))
        except KeyError as e:
            # This will now only catch essential missing maps like wgpid0, etc.
            messagebox.showerror("Map Definition Error",
                                 f"A required WG map is missing: {e}. Please check your map definitions.")
        except Exception as e:
            print(f"[ERROR] WG Tuner failed: {e}")

    if settings['MAFtune']:
        print("\n[MODULE] Running Mass Airflow (MAF) Tuner...")
        try:
            Res_MAF = MAF.MAF_tune(
                log_df,
                all_maps['maftable0_X'],
                all_maps['maftable0_Y'],
                maftables,
                all_maps['combmodes_MAF'],
                logvars,
                root
            )
            if settings['save'] and Res_MAF is not None:
                output_dir = os.path.dirname(log_paths[0])
                for i in range(4):
                    Res_MAF[f'IDX{i}'].to_csv(os.path.join(output_dir, f"MAF_IDX{i}_Results.csv"))
        except KeyError as e:
            messagebox.showerror("Map Definition Error",
                                 f"A required map is missing: {e}. Please check '{MAP_DEFINITIONS_CSV_PATH}'.")
        except Exception as e:
            print(f"[ERROR] MAF Tuner failed: {e}")

    if settings['MFFtune']:
        print("\n[MODULE] Running Mass Fuel Factor (MFF) Tuner...")
        try:
            Res_MFF = MFF.MFF_tune(
                log_df,
                all_maps['MFFtable0_X'],
                all_maps['MFFtable0_Y'],
                mfftables,
                all_maps['combmodes_MFF'],
                logvars,
                root
            )
            if settings['save'] and Res_MFF is not None:
                output_dir = os.path.dirname(log_paths[0])
                for i in range(5):  # Loop through 5 tables
                    Res_MFF[f'IDX{i}'].to_csv(os.path.join(output_dir, f"MFF_IDX{i}_Results.csv"))
        except KeyError as e:
            messagebox.showerror("Map Definition Error",
                                 f"A required map is missing: {e}. Please check '{MAP_DEFINITIONS_CSV_PATH}'.")
        except Exception as e:
            print(f"[ERROR] MFF Tuner failed: {e}")

    if settings['IGtune']:
        print("\n[MODULE] Running Ignition (KNK) Tuner...")
        try:
            knk_max_adv = float(settings['knk_max_adv'])
            knk_map_num = int(settings['knk_map_num'])

            if not (0 <= knk_map_num <= 6):
                raise ValueError("SP Map Number must be between 0 and 6.")

            Res_KNK = KNK.KNK(
                log_df,
                all_maps['igxaxis'],
                all_maps['igyaxis'],
                IGNmaps,
                max_adv=knk_max_adv,
                map_num=knk_map_num,
                parent=root
                )
            if settings['save'] and Res_KNK is not None:
                output_dir = os.path.dirname(log_paths[0])
                Res_KNK.to_csv(os.path.join(output_dir, "IGN_Results.csv"))

        except ValueError as e:
            messagebox.showerror("Invalid KNK Input", f"Please check your Ignition Tuner settings.\n\nError: {e}")
        except KeyError as e:
            messagebox.showerror("Map Definition Error",
                                 f"A required map is missing: {e}. Please check '{MAP_DEFINITIONS_CSV_PATH}'.")
        except Exception as e:
            print(f"[ERROR] Ignition Tuner failed: {e}")

    if settings['LPFPtune']:
        print("\n[MODULE] Running Low-Pressure Fuel Pump (LPFP) Tuner...")
        try:
            # Select the correct table based on user's 2WD/4WD choice
            table_to_correct = all_maps['lpfppwm'] if settings['LPFPdrivetype'] == '2WD' else all_maps['lpfppwm4wd']

            Res_LPFP = LPFP.LPFP_tune(
                log_df,
                all_maps['lpfppwm_X'],
                all_maps['lpfppwm_Y'],
                table_to_correct,
                logvars,
                root
            )
            if settings['save'] and Res_LPFP is not None:
                output_dir = os.path.dirname(log_paths[0])
                drive_type = settings['LPFPdrivetype']
                Res_LPFP.to_csv(os.path.join(output_dir, f"LPFP_{drive_type}_Results.csv"))
        except KeyError as e:
            messagebox.showerror("Map Definition Error",
                                 f"A required LPFP map is missing: {e}. Please check '{MAP_DEFINITIONS_CSV_PATH}'.")
        except Exception as e:
            print(f"[ERROR] LPFP Tuner failed: {e}")

    # Instead, start the Tkinter event loop. The application will now wait
    # for you to close the result windows.
    print("Tuning process complete. Displaying results. Close all windows to exit.")
    root.mainloop()

if __name__ == "__main__":
    # This ensures the script runs only when executed directly.
    main()
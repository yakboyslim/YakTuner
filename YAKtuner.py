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

# Import the custom tuning modules
import BinRead
import KNK
import MAF
import WG

# --- Constants ---
# Define constants for configuration filenames to avoid "magic strings".
VARIABLES_CSV_PATH = "variables.csv"
MAP_DEFINITIONS_CSV_PATH = "map_definitions.csv"


def setup_and_run_gui():
    """
    Creates and displays the main GUI for gathering user settings.

    This function builds a user interface using customtkinter to allow the user to:
    - Select which tuning modules to run (WG, MAF, Ignition).
    - Choose exactly one firmware version (S50, A05, V30, Custom).
    - Set module-specific options (like SWG logic for WG tuning).
    - Configure output and utility options.

    Returns:
        dict: A dictionary containing all the user's selections. Returns an empty dict
              if the user closes the window without clicking "CONTINUE".
    """
    root = ctk.CTk()
    root.title("YakTuner Settings")

    # FIX: Removed the fixed root.geometry() call.
    # The window will now automatically resize to fit its content.

    # This dictionary will hold the final settings.
    settings = {}

    # --- Main Frames for Organization ---
    tune_frame = ctk.CTkFrame(root)
    tune_frame.pack(pady=10, padx=10, fill="x")

    firmware_frame = ctk.CTkFrame(root)
    firmware_frame.pack(pady=10, padx=10, fill="x")

    options_frame = ctk.CTkFrame(root)
    options_frame.pack(pady=10, padx=10, fill="x")

    # --- Tuning Module Section ---
    tune_label = ctk.CTkLabel(tune_frame, text="Tuning Modules", font=ctk.CTkFont(weight="bold"))
    tune_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 0))

    cbx_wg = ctk.CTkCheckBox(tune_frame, text="Tune WG?")
    cbx_wg.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    cbx_maf = ctk.CTkCheckBox(tune_frame, text="Tune MAF?")
    cbx_maf.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    cbx_ign = ctk.CTkCheckBox(tune_frame, text="Tune Ignition?")
    cbx_ign.grid(row=1, column=2, padx=10, pady=5, sticky="w")

    # SWG checkbox, dependent on the WG checkbox
    cbx_swg = ctk.CTkCheckBox(tune_frame, text="Use SWG Logic?", state="disabled")
    cbx_swg.grid(row=2, column=0, padx=10, pady=5, sticky="w")

    def toggle_swg_checkbox():
        """Enable/disable the SWG checkbox based on the WG checkbox state."""
        if cbx_wg.get() == 1:
            cbx_swg.configure(state="normal")
        else:
            cbx_swg.configure(state="disabled")
            cbx_swg.deselect()

    # Link the toggle function to the WG checkbox
    cbx_wg.configure(command=toggle_swg_checkbox)

    # --- Firmware Section ---
    firmware_label = ctk.CTkLabel(firmware_frame, text="Firmware Selection (Must choose one)",
                                  font=ctk.CTkFont(weight="bold"))
    firmware_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(5, 0))

    firmware_var = ctk.StringVar(value="S50")  # Set a default value
    firmware_options = ["S50", "A05", "V30", "Custom"]
    for i, option in enumerate(firmware_options):
        rb = ctk.CTkRadioButton(
            firmware_frame,
            text=option,
            variable=firmware_var,
            value=option
        )
        rb.grid(row=1, column=i, padx=10, pady=5, sticky="w")

    # --- Other Options Section ---
    options_label = ctk.CTkLabel(options_frame, text="Other Options", font=ctk.CTkFont(weight="bold"))
    options_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))

    cbx_save = ctk.CTkCheckBox(options_frame, text="Save results to CSV?")
    cbx_save.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    cbx_reset_vars = ctk.CTkCheckBox(options_frame, text="Reset Variable Names?")
    cbx_reset_vars.grid(row=1, column=1, padx=10, pady=5, sticky="w")

    def on_continue():
        """
        Callback function to capture settings when the user clicks "CONTINUE".
        """
        nonlocal settings
        settings = {
            'WGtune': bool(cbx_wg.get()),
            'MAFtune': bool(cbx_maf.get()),
            'IGtune': bool(cbx_ign.get()),
            'save': bool(cbx_save.get()),
            'WGlogic': bool(cbx_swg.get()),
            'firmware': firmware_var.get(),  # Get the single selected firmware
            'var_reset': bool(cbx_reset_vars.get()),
            'did_continue': True
        }
        root.quit()

    continue_btn = ctk.CTkButton(root, text="CONTINUE", command=on_continue)
    continue_btn.pack(pady=20)

    root.mainloop()
    root.destroy()
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
        log_list = [pd.read_csv(f).iloc[:, :-1] for f in log_paths]
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

    return log_df, bin_path, log_paths


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
        var_reset (bool): If True, forces the user to remap all variables.

    Returns:
        pd.DataFrame: The log DataFrame with columns renamed.
    """
    varconv = varconv_df.to_numpy()
    logvars = log_df.columns.tolist()
    missing_vars_indices = []

    if var_reset:
        # If reset is checked, mark all variables for re-mapping.
        missing_vars_indices = list(range(1, varconv.shape[1]))
    else:
        # Loop through each variable we need to find (from variables.csv).
        for i in range(1, varconv.shape[1]):
            target_var_from_csv = varconv[0, i]
            normalized_target = normalize_header(target_var_from_csv)

            match_found = False
            # Loop through the actual headers in the log file to find a match.
            for log_header in logvars:
                normalized_log_header = normalize_header(log_header)

                # Compare the normalized versions for a flexible match.
                if normalized_log_header == normalized_target:
                    # Found a match! Rename the column in the DataFrame.
                    log_df = log_df.rename(columns={log_header: varconv[1, i]})

                    # Update the mapping file with the exact header we found.
                    # This self-healing makes future runs faster.
                    varconv[0, i] = log_header

                    match_found = True
                    break  # Move to the next variable in variables.csv

            if not match_found:
                # If we looped through all log headers and found no match,
                # mark it for manual user intervention.
                missing_vars_indices.append(i)

    # If any variables are still missing, prompt the user for each one.
    if missing_vars_indices:
        for i in missing_vars_indices:
            # Use a dictionary to store the result from the dialog callback.
            selection_result = {'selected_var': None}

            # Create a new Toplevel window for the prompt.
            var_window = Toplevel()
            var_window.title(f"Select variable for: {varconv[2, i]}")
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
    Main function to orchestrate the entire tuning process.
    """
    current_dir = os.getcwd()

    # --- 1. Get User Input from GUI ---
    settings = setup_and_run_gui()
    if not settings.get('did_continue'):
        print("User closed the GUI. Exiting.")
        return

    # --- 2. Load Data ---
    try:
        varconv_df = pd.read_csv(os.path.join(current_dir, VARIABLES_CSV_PATH), header=None, dtype=str)
        map_definitions = pd.read_csv(os.path.join(current_dir, MAP_DEFINITIONS_CSV_PATH))
    except FileNotFoundError as e:
        messagebox.showerror("Configuration Error", f"A required file is missing: {e.filename}")
        return

    log_df, bin_path, log_paths = load_data_files()
    if log_df is None:
        print("User cancelled file selection. Exiting.")
        return

    # --- 3. Process and Map Log Variables ---
    log_df = map_log_variables(log_df, varconv_df, settings['var_reset'])
    logvars = log_df.columns.tolist()

    # --- 4. Read Tune Data from Binary File ---
    #
    # REFACTORED LOGIC: Handle the single firmware selection from the radio buttons.
    # This is cleaner and more robust than the previous if/elif chain.
    #
    selected_firmware = settings['firmware']
    if selected_firmware == 'Custom':
        firmware_col = 'address_custom'
    else:
        firmware_col = f"address_{selected_firmware}"

    # Validate that the required address column exists in the map definitions CSV.
    if firmware_col not in map_definitions.columns:
        messagebox.showerror(
            "Configuration Error",
            f"The selected firmware '{selected_firmware}' requires a column named '{firmware_col}' "
            "in map_definitions.csv, but it was not found."
        )
        return

    # Define dynamic overrides. If "SWG?" is checked, we override the resolution for WG axes.
    overrides = {}
    if settings['WGlogic']:
        overrides['wgyaxis'] = {'res': 1 / 0.082917524986648}
        overrides['wgxaxis'] = {'res': 1.0}

    # Read ALL maps from the binary file in a single, efficient function call.
    all_maps = BinRead.read_maps_from_config(bin_path, map_definitions, firmware_col, overrides)

    # Unpack the returned dictionary into variables for the tuning modules.
    try:
        maftables = [all_maps['maftable0'], all_maps['maftable1'], all_maps['maftable2'], all_maps['maftable3']]
        IGNmaps = [all_maps['igmap0'], all_maps['igmap1'], all_maps['igmap2'], all_maps['igmap3'], all_maps['igmap4']]
    except KeyError as e:
        messagebox.showerror("Map Definition Error", f"A required map is missing: {e}. Please check 'map_definitions.csv'.")
        return

    # --- 5. Run Selected Tuning Modules ---
    if settings['WGtune']:
        Res_1, Res_0 = WG.WG_tune(
            log_df, all_maps['wgxaxis'], all_maps['wgyaxis'], all_maps['currentWG0'],
            all_maps['currentWG1'], logvars, True, settings['WGlogic'],
            all_maps['tempcomp'], all_maps['tempcompaxis']
        )
        if settings['save']:
            output_dir = os.path.dirname(log_paths[0])
            Res_1.to_csv(os.path.join(output_dir, "VVL1 Results.csv"))
            Res_0.to_csv(os.path.join(output_dir, "VVL0 Results.csv"))

    if settings['MAFtune']:
        MAFresults = MAF.MAF_tune(
            log_df, all_maps['mafxaxis'], all_maps['mafyaxis'],
            maftables, all_maps['combmodes'], logvars
        )
        if settings['save']:
            output_dir = os.path.dirname(log_paths[0])
            for idx in range(4):
                MAFresults[f"IDX{idx}"].to_csv(os.path.join(output_dir, f"MAF_STD[{idx}] Results.csv"))

    if settings['IGtune']:
        # CORRECTED call with the right number of arguments
        Res_KNK = KNK.KNK(
            log_df,
            all_maps['igxaxis'],
            all_maps['igyaxis'],
            IGNmaps
        )
        if settings['save']:
            output_dir = os.path.dirname(log_paths[0])
            Res_KNK.to_csv(os.path.join(output_dir, "KNK Results.csv"))

    messagebox.showinfo("Complete", "Tuning process has finished.")


if __name__ == "__main__":
    # This ensures the script runs only when executed directly.
    main()
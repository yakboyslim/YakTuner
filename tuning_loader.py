# tuning_loader.py

import pandas as pd
import numpy as np
from asteval import Interpreter  # A safe math expression evaluator. Install with: pip install asteval
import pprint
import os

# Assuming these scripts are in the same project directory
from xdf_parser import parse_xdf_maps
from BinRead import read_maps_from_config


# --- Helper functions for processing XDF data ---

def _map_xdf_type_to_numpy(data_size_bits, is_signed, endian='<'):
    """Maps XDF data type info to a NumPy dtype string."""
    if data_size_bits == 8:
        return f"{endian}{'i' if is_signed else 'u'}1"
    elif data_size_bits == 16:
        return f"{endian}{'i' if is_signed else 'u'}2"
    elif data_size_bits == 32:
        return f"{endian}{'i' if is_signed else 'u'}4"
    elif data_size_bits == 64:
        return f"{endian}{'i' if is_signed else 'u'}8"
    else:
        raise ValueError(f"Unsupported data size: {data_size_bits} bits")


def _apply_equation(raw_values, equation_str):
    """Safely applies the conversion equation to the raw data values."""
    if not equation_str or equation_str.strip().upper() == 'X':
        return raw_values  # No conversion needed

    aeval = Interpreter()
    aeval.eval(f'def convert(X): return {equation_str}')
    return np.vectorize(aeval.symtable['convert'])(raw_values)


def _read_data_from_xdf_definitions(xdf_definitions, binary_file_path):
    """Reads data from a binary file using definitions parsed from an XDF."""
    processed_maps = {}
    try:
        with open(binary_file_path, 'rb') as f:
            for name, definition in xdf_definitions.items():
                try:
                    address = int(definition['address'], 16)
                    rows, cols = definition['rows'], definition['cols']
                    dtype_str = _map_xdf_type_to_numpy(definition['data_size_bits'], definition['signed'])

                    f.seek(address)
                    count = rows * cols
                    raw_data = np.fromfile(f, dtype=dtype_str, count=count)

                    if raw_data.size != count:
                        print(
                            f"Warning: For '{name}', expected {count} values but only read {raw_data.size}. File may be truncated.")
                        continue

                    # Reshape using column-major order ('F') as is standard for XDF
                    reshaped_data = raw_data.reshape((rows, cols), order='F')
                    physical_data = _apply_equation(reshaped_data, definition['equation'])


                    if definition.get('is_axis', False):
                        physical_data = physical_data.flatten()

                    processed_maps[name] = physical_data
                    print(f"  [XDF] -> Successfully read and processed '{name}'.")

                except Exception as e:
                    print(f"Error processing XDF map '{name}': {e}")
    except FileNotFoundError:
        print(f"Error: Binary file not found at '{binary_file_path}'")
        return {}
    return processed_maps


# --- Main Loader Class ---

class TuningData:
    """
    Manages loading tuning data from multiple sources (XDF, manual CSV)
    into a unified dictionary of NumPy arrays.
    """

    def __init__(self, binary_file_path):
        """
        Initializes the loader with the path to the binary file.
        """
        if not os.path.exists(binary_file_path):
            raise FileNotFoundError(f"Binary file not found: {binary_file_path}")
        self.binary_file_path = binary_file_path
        self.maps = {}  # This will hold the final data: {'map_name': np.array, ...}
        print(f"TuningData loader initialized for: {os.path.basename(binary_file_path)}")

    def load_from_xdf(self, xdf_file_path, xdf_map_list_csv):
        """
        Parses an XDF to get map definitions, then reads the data from the binary.
        """
        print(f"\n--- Loading maps from XDF: {os.path.basename(xdf_file_path)} ---")
        if not os.path.exists(xdf_file_path):
            print(f"Warning: XDF file not found at '{xdf_file_path}'. Skipping XDF load.")
            return

        if not os.path.exists(xdf_map_list_csv):
            print(f"Warning: XDF map list CSV not found at '{xdf_map_list_csv}'. Skipping XDF load.")
            return

        xdf_definitions = parse_xdf_maps(xdf_file_path, xdf_map_list_csv)
        if not xdf_definitions:
            print("No map definitions were parsed from the XDF. Nothing to load.")
            return

        xdf_maps_data = _read_data_from_xdf_definitions(xdf_definitions, self.binary_file_path)
        self.maps.update(xdf_maps_data)
        print(f"--- XDF loading complete. Loaded {len(xdf_maps_data)} maps. ---")

    def load_from_manual_config(self, manual_config_csv_path, firmware_address_col, overrides=None):
        """
        Reads maps defined in a manual CSV configuration file.
        """
        print(f"\n--- Loading maps from manual config: {os.path.basename(manual_config_csv_path)} ---")
        if not os.path.exists(manual_config_csv_path):
            print(f"Warning: Manual config file not found at '{manual_config_csv_path}'. Skipping manual load.")
            return

        try:
            config_df = pd.read_csv(manual_config_csv_path)
        except Exception as e:
            print(f"Error reading manual config CSV '{manual_config_csv_path}': {e}")
            return

        manual_maps_data = read_maps_from_config(
            self.binary_file_path, config_df, firmware_address_col, overrides
        )

        for name, data in manual_maps_data.items():
            if name in self.maps:
                print(f"Warning: Map '{name}' from manual config is overwriting a previously loaded map.")
            else:
                print(f"  [Manual] -> Successfully read and processed '{name}'.")
            self.maps[name] = data

        print(f"--- Manual loading complete. Loaded {len(manual_maps_data)} maps. ---")


# --- TEST SCRIPT ---
if __name__ == "__main__":
    """
    This block allows the script to be run directly from the command line
    for testing and debugging purposes.
    """
    # --- Define file paths for testing ---
    # NOTE: You will need to replace these with the actual paths to your files.
    binary_file = "2867 V6.9.15.bin"  # The binary tune file
    xdf_file = "SC8S50.ALL.xdf"
    xdf_map_list = "maps_to_parse.csv"
    manual_config_file = "map_definitions.csv"
    firmware_col = "address_S50"  # Example firmware column from manual config

    # --- Basic Usage Example ---
    print("--- Starting TuningData Loader Test ---")

    # Check if the essential binary file exists before proceeding
    if not os.path.exists(binary_file):
        print(f"\nFATAL: Test binary file '{binary_file}' not found.")
        print("Please ensure the file exists in the same directory as the script or provide a full path.")
    else:
        try:
            # 1. Initialize the loader with the binary file path
            loader = TuningData(binary_file)

            # 2. Load maps defined in the XDF file
            # This uses the xdf_parser to get definitions, then reads the data
            loader.load_from_xdf(xdf_file, xdf_map_list)

            # 3. Load maps from the manual CSV configuration
            # This will also overwrite any maps with the same name loaded from the XDF
            loader.load_from_manual_config(manual_config_file, firmware_col)

            # 4. Print the results
            if loader.maps:
                print("\n--- Verifying Loaded Map Shapes ---")

                # Example of accessing a specific map's data (e.g., its shape)
                if 'maftable0' in loader.maps:
                    print(f"  'maftable0' (2D Table) has shape: {loader.maps['maftable0'].shape}")
                if 'maftable0_X' in loader.maps:
                    print(f"  'maftable0_X' (Axis) has shape: {loader.maps['maftable0_X'].shape} <-- Should be 1D")
                if 'maftable0_Y' in loader.maps:
                    print(f"  'maftable0_Y' (Axis) has shape: {loader.maps['maftable0_Y'].shape} <-- Should be 1D")

                print("\n--- Test Finished Successfully ---")
            else:
                print("\n--- Test Finished: No maps were loaded. Check warnings above. ---")

        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
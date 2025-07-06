import numpy as np
import pandas as pd

def read_maps_from_config(bin_file_path, config_df, firmware_address_col, overrides=None):
    """
    Reads multiple data maps from a binary file based on a configuration DataFrame.

    Args:
        bin_file_path (str): Path to the binary .bin file.
        config_df (pd.DataFrame): DataFrame containing map definitions.
        firmware_address_col (str): The name of the column in config_df that contains
                                    the addresses for the selected firmware.
        overrides (dict, optional): A dictionary to override specific parameters for
                                    certain maps. Example: {'wgxaxis': {'res': 1.0}}

    Returns:
        dict: A dictionary where keys are variable names and values are the read numpy arrays.
    """
    if overrides is None:
        overrides = {}

    results = {}

    with open(bin_file_path, 'rb') as bin_file:
        for _, row in config_df.iterrows():
            var_name = row['variable_name']

            # Get base parameters from the DataFrame row
            try:
                params = {
                    'address': int(str(row[firmware_address_col]), 16),
                    'rows': int(row['rows']),
                    'cols': int(row['cols']),
                    'offset': float(row['offset']),
                    'res': float(row['res']),
                    'prec': row['prec']
                }
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse parameters for '{var_name}'. Skipping. Error: {e}")
                continue

            # Apply any specific overrides from the main script
            if var_name in overrides:
                params.update(overrides[var_name])

            # Read and process the data from the binary file
            bin_file.seek(params['address'])
            count = params['rows'] * params['cols']
            data = np.fromfile(bin_file, dtype=params['prec'], count=count)

            if data.size != count:
                print(f"Warning: Read wrong number of bytes for '{var_name}'. Expected {count}, got {data.size}. Skipping.")
                continue

            data = data.reshape((params['rows'], params['cols']), order='F')
            data = data.astype(float)  # Use float for calculations to preserve precision
            data = (data - params['offset']) / params['res']

            # If a map is just a single row (like an axis), flatten it to a 1D array
            # for consistency with the original code's expectations.
            if params['rows'] == 1:
                data = np.ravel(data)

            results[var_name] = data

    return results
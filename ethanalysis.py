import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# --- CONFIGURATION: UPDATE THESE VALUES ---
# ==============================================================================
# Fill in the exact column names from your CSV file here.

CSV_FILE_NAME = 'ethanol_target_factor_analysis.csv'
ETHANOL_COLUMN = 'Eth Content (%)'  # The column with your ethanol percentage
CORRECTION_FACTOR_COLUMN = 'target_factor' # The column with your correction value

# ==============================================================================
# --- SCRIPT (No changes needed below this line) ---
# ==============================================================================

# Step 1: Load and Process Data
try:
    df = pd.read_csv(CSV_FILE_NAME)
except FileNotFoundError:
    print(f"Error: '{CSV_FILE_NAME}' not found.")
    exit()

print("Successfully loaded CSV. Columns found:", df.columns.tolist())

# Step 2: Calculate the Ideal Factor
try:
    # These are the factor values from the last table we made (linear correction).
    # This is our "base" map that we are now correcting.
    base_eth_points = np.array([0, 17, 34, 51, 68, 85])
    base_factor_points = np.array([0.05851, 0.06737, 0.07675, 0.08659, 0.09698, 0.10774])

    # For each row in your log, find what the base factor would have been.
    df['base_factor'] = np.interp(df[ETHANOL_COLUMN], base_eth_points, base_factor_points)

    # The new ideal factor is the base factor adjusted by your logged correction.
    df['ideal_factor'] = df['base_factor'] * df[CORRECTION_FACTOR_COLUMN]

except KeyError as e:
    print(f"\n--- ERROR ---")
    print(f"A column name in the CONFIGURATION section is incorrect: {e}")
    print("Please check the column names at the top of the script and try again.")
    exit()

# Step 3: Model the Trend with a Polynomial Fit
coeffs = np.polyfit(df[ETHANOL_COLUMN], df['ideal_factor'], 2)
p = np.poly1d(coeffs)

# Step 4: Generate the Final, Re-Tuned Table
blend_map = {0.0: 0, 0.2: 17, 0.4: 34, 0.6: 51, 0.8: 68, 1.0: 85}
final_table_data = []

for blend_factor, eth_pct in blend_map.items():
    new_tuned_factor = p(eth_pct)
    old_factor = np.interp(eth_pct, base_eth_points, base_factor_points) # Get old factor for table
    final_table_data.append({
        "Blend Factor (0 → 1)": f"{blend_factor:.2f}",
        "Ethanol Content": f"{eth_pct}%",
        "Old Factor": f"{old_factor:.5f}", # Added old factor column
        "Final Tuned Factor": f"{new_tuned_factor:.5f}"
    })

final_df = pd.DataFrame(final_table_data)

print("\n--- Final Tuned Factor Table with Comparison ---")
print(final_df.to_string(index=False))

# Step 5: Visualize the Results
x_fit = np.linspace(0, 90, 100)
y_fit = p(x_fit)

plt.figure(figsize=(12, 7))

# Plot the old, uncorrected factor points
plt.scatter(base_eth_points, base_factor_points, color='blue', s=100, label='Old Factor Points')

# Plot the newly calculated "ideal" points from your data log
plt.scatter(df[ETHANOL_COLUMN], df['ideal_factor'], alpha=0.5, label='Calculated Ideal Factor')

# Plot the new best-fit curve
plt.plot(x_fit, y_fit, color='r', linewidth=2.5, label='New Best-Fit Curve')

plt.title('Fuel Factor Comparison: Old vs. New')
plt.xlabel('Ethanol Percentage (%)')
plt.ylabel('Multiplicative Factor')
plt.grid(True)
plt.legend()
plt.savefig('final_factor_comparison_fit.png')
print("\nSaved plot 'final_factor_comparison_fit.png'")
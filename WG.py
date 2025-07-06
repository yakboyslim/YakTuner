"""
Wastegate (WG) Tuning Module for YAKtuner

This module analyzes engine log data to recommend adjustments to the wastegate base tables.
Key functionalities include:
- Filtering log data to isolate relevant operating conditions.
- Calculating the required wastegate duty cycle ('WGNEED') to correct for boost errors.
- Fitting a 3D surface to the scattered log data points (EFF, IFF, WGNEED) for both
  VVL0 and VVL1 camshaft profiles using scipy's griddata for robust interpolation.
- Applying a confidence interval to determine whether the newly fitted values are a
  statistically significant improvement over the existing table values.
- Calculating a recommended temperature compensation slope for the WG base table.
- Displaying the results in an interactive, color-coded table for user review.
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate
import tkinter as tk
from tkinter import simpledialog, messagebox, Toplevel, Frame
from pandastable import Table
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- Helper Functions ---

def _get_wg_tuning_parameters():
    """Shows dialogs to get user inputs for key WG tuning parameters."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    params = {
        'fudge': float(simpledialog.askstring("WG Inputs", "PUT fudge factor:", initialvalue="0.71")),
        'maxdelta': float(simpledialog.askstring("WG Inputs", "Maximum PUT delta:", initialvalue="10")),
        'minboost': float(simpledialog.askstring("WG Inputs", "Minimum Boost:", initialvalue="0"))
    }

    # Add a prompt for 3D visualization
    params['show_3d_plot'] = messagebox.askyesno(
        "3D Visualization",
        "Would you like to visualize the results in a 3D plot?\n(This can help in understanding the changes)"
    )

    return params

def _plot_3d_surfaces(title, wgxaxis, wgyaxis, old_map, new_map, log_data, WGlogic, changed_mask):
    """
    Creates an interactive 3D plot to visualize and compare surfaces.
    - Original WG map (wireframe)
    - Recommended WG map (surface)
    - Raw WGNEED data (scatter)
    - Markers for changed cells (scatter)
    """
    if log_data.empty:
        print(f"Skipping 3D plot for {title}: No log data available.")
        return

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid for the surface plots
    X, Y = np.meshgrid(wgxaxis, wgyaxis)

    # 1. Plot the raw WGNEED data points from the log
    ax.scatter(log_data['EFF'], log_data['IFF'], log_data['WGNEED'], c='red', marker='o', label='Raw WGNEED Data', depthshade=True, s=20)

    # 2. Plot the original map as a wireframe for reference
    ax.plot_wireframe(X, Y, old_map * 100.0, color='gray', alpha=0.7, label='Original Map')

    # 3. Plot the new recommended map as a colored surface
    ax.plot_surface(X, Y, new_map * 100.0, cmap='viridis', alpha=0.6, label='Recommended Map')

    # 4. Highlight the cells where changes were recommended by plotting markers
    # Get the coordinates (indices) of the changed cells from the boolean mask
    changed_y_indices, changed_x_indices = np.where(changed_mask)

    if changed_y_indices.size > 0:
        # Get the corresponding X, Y, and Z values for the markers
        x_coords = wgxaxis[changed_x_indices]
        y_coords = wgyaxis[changed_y_indices]
        z_coords = new_map[changed_y_indices, changed_x_indices] * 100.0 + 0.5  # Add a small Z-offset

        # Plot bright markers on top of the surface for changed cells
        ax.scatter(x_coords, y_coords, z_coords, c='magenta', marker='X', s=60, label='Changed Cells', depthshade=False)

    # --- Set axis labels based on the selected WG logic ---
    if WGlogic:
        x_label = 'Engine Speed (RPM)'
        y_label = 'PUT SP (Axis scaled by 10)'
    else:
        x_label = 'Engine Efficiency (EFF)'
        y_label = 'Intake Flow Factor (IFF)'

    # Setting labels and title
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_zlabel('Wastegate Duty Cycle (%)', fontsize=12)
    ax.invert_yaxis()  # Match the 2D table and plot orientation

    # Create "proxy artists" for a clean, manual legend that includes the new markers
    legend_elements = [Line2D([0], [0], color='gray', lw=2, label='Original Map'),
                       Patch(facecolor=plt.cm.viridis(0.5), edgecolor='k', label='Recommended Map'),
                       Line2D([0], [0], marker='o', color='w', label='Raw WGNEED Data', markerfacecolor='r', markersize=8),
                       Line2D([0], [0], marker='X', color='w', label='Changed Cells', markerfacecolor='magenta', markersize=10)]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.show(block=True)

def _prepare_and_filter_log(log, params, logvars, WGlogic, tempcomp, tempcompaxis):
    """Adds derived columns and filters the log data to isolate relevant tuning conditions."""
    # Determine SWG/FF logic and apply temperature correction
    if WGlogic:
        log['EFF'] = log['RPM']
        log['IFF'] = log['PUTSP'] * 10
        interp_func = interpolate.interp1d(tempcompaxis, tempcomp, kind='linear', fill_value='extrapolate')
        tempcorr = interp_func(log['AMBTEMP'])
    else:
        tempcorr = 0

    # Create derived values for analysis
    log['deltaPUT'] = log['PUT'] - log['PUTSP']
    log['WGNEED_uncorrected'] = log['WG_Final'] - log['deltaPUT'] * params['fudge']
    log['WGNEED'] = log['WGNEED_uncorrected'] - tempcorr

    # Filter log data to valid conditions
    if 'I_INH' in logvars:
        log = log[log['I_INH'] <= 0]
    else:
        minpedal = float(simpledialog.askstring("WG Inputs", "Recommend logging PUT I Inhibit.\nChoose minimum pedal to use:", initialvalue="50"))
        log = log[log['Pedal'] >= minpedal]

    if 'DV' in logvars:
        log = log[log['DV'] <= 50]
    else:
        messagebox.showwarning('Recommendation', 'Recommend logging DV position. Otherwise, DV may impact accuracy.')

    if 'BOOST' in logvars:
        log = log[log['BOOST'] >= params['minboost']]
    else:
        messagebox.showwarning('Recommendation', 'Recommend logging boost. Otherwise, logs are not trimmed for min boost.')

    log = log[abs(log['deltaPUT']) <= params['maxdelta']]
    log = log[log['WG_Final'] <= 98]
    return log

def _create_bins_and_labels(log_df, wgxaxis, wgyaxis):
    """Creates bin edges from axes and assigns each log entry to a grid cell (X, Y)."""
    wgxedges = np.zeros(len(wgxaxis) + 1)
    wgxedges[0] = wgxaxis[0]
    wgxedges[-1] = wgxaxis[-1] + 2
    for i in range(len(wgxaxis) - 1):
        wgxedges[i + 1] = (wgxaxis[i] + wgxaxis[i + 1]) / 2

    wgyedges = np.zeros(len(wgyaxis) + 1)
    wgyedges[0] = wgyaxis[0]
    wgyedges[-1] = wgyaxis[-1] + 2
    for i in range(len(wgyaxis) - 1):
        wgyedges[i + 1] = (wgyaxis[i] + wgyaxis[i + 1]) / 2

    log_df['X'] = pd.cut(log_df['EFF'], wgxedges, labels=False)
    log_df['Y'] = pd.cut(log_df['IFF'], wgyedges, labels=False)
    return log_df

def _plot_wg_data(log_VVL0, log_VVL1, wgxaxis, wgyaxis):
    """Displays a scatter plot of the filtered log data."""
    plt.figure(figsize=(12, 8))
    plt.scatter(log_VVL1['EFF'], log_VVL1['IFF'], s=abs(log_VVL1['WGNEED']), c=log_VVL1['deltaPUT'], marker='x', cmap='RdBu', label='VVL1')
    plt.scatter(log_VVL0['EFF'], log_VVL0['IFF'], s=abs(log_VVL0['WGNEED']), c=log_VVL0['deltaPUT'], marker='o', cmap='RdBu', label='VVL0', alpha=0.7)
    cbar = plt.colorbar()
    cbar.set_label('PUT - PUT SP (kPa)')
    plt.gca().invert_yaxis()
    plt.xlabel('Engine Efficiency (EFF)')
    plt.ylabel('Intake Flow Factor (IFF)')
    plt.title('Wastegate Duty Cycle Need vs. Operating Point')
    plt.grid(True)
    plt.xticks(wgxaxis, rotation=45)
    plt.yticks(wgyaxis)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

def _fit_surface(log_data, wgxaxis, wgyaxis):
    """
    Fits a 3D surface to the provided log data using scipy.interpolate.griddata.
    This replaces the previous 1D row/column pwlf fitting method.
    """
    if log_data.empty or len(log_data) < 3:
        return np.zeros((len(wgyaxis), len(wgxaxis)))

    points = log_data[['EFF', 'IFF']].values
    values = log_data['WGNEED'].values
    grid_x, grid_y = np.meshgrid(wgxaxis, wgyaxis)

    # Interpolate using a linear method, which is analogous to piecewise linear.
    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill any NaNs (points outside the convex hull of the data) using nearest neighbor.
    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    # If all points were outside the hull, return a zeroed array.
    if np.all(np.isnan(fitted_surface)):
        return np.zeros((len(wgyaxis), len(wgxaxis)))

    # Return the surface scaled to a 0-1 range, consistent with the old tables.
    return fitted_surface / 100.0

def _calculate_final_recommendations(log_data, blend, old_table, wgxaxis, wgyaxis, calculate_temp_coef=False):
    """
    Calculates the final recommended WG table by comparing the new fit with the old
    table and applying confidence intervals to each cell.
    """
    final_table = old_table.copy()
    total_coef, count_coef = 0, 0

    for i in range(len(wgxaxis)):
        for j in range(len(wgyaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            count = len(cell_data)

            # Only update cells with sufficient data points.
            if count > 3:
                # Fit a normal distribution to the WGNEED data in the cell.
                mean, std_dev = stats.norm.fit(cell_data['WGNEED'])
                # Get a 50% confidence interval.
                low_ci, high_ci = stats.norm.interval(0.5, loc=mean, scale=std_dev)

                # The old table values are fractions; scale them to compare with WGNEED.
                current_val_scaled = old_table[j, i] * 100.0

                # Decide whether to update the cell.
                # Update if the old value is outside the confidence interval of the new data.
                if np.isnan(current_val_scaled) or not (low_ci <= current_val_scaled <= high_ci):
                    # Average the fitted surface value with the cell's mean value.
                    new_val = (blend[j, i] + (mean / 100.0)) / 2
                    final_table[j, i] = new_val
                # Otherwise, keep the old value.

            # Optional: Calculate temperature compensation coefficient for VVL0.
            if calculate_temp_coef:
                temp_log = cell_data[cell_data['BOOST'] >= 8]
                if len(temp_log) > 2 and (temp_log['AMBTEMP'].max() > temp_log['AMBTEMP'].min() + 15):
                    coef = np.polyfit(temp_log['AMBTEMP'], temp_log['WGNEED_uncorrected'] / 100, 1)[0]
                    total_coef += coef * len(temp_log)
                    count_coef += len(temp_log)

    # Round the final table to the resolution of the ECU.
    final_table = np.round(final_table * 16384) / 16384

    if calculate_temp_coef:
        avg_coef = total_coef / count_coef if count_coef > 0 else "Not enough data"
        return final_table, avg_coef
    else:
        return final_table

class _ColoredTable(Table):
    """A pandastable Table subclass that colors cells based on value comparison."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

    def color_cells(self, new_array, old_array):
        self.resetColors()
        if new_array.shape != old_array.shape:
            return
        for r in range(new_array.shape[0]):
            for c in range(new_array.shape[1]):
                if new_array[r, c] > old_array[r, c]:
                    self.setRowColors(rows=[r], cols=[c], clr='#90EE90')  # Light Green
                elif new_array[r, c] < old_array[r, c]:
                    self.setRowColors(rows=[r], cols=[c], clr='#FFB6C1')  # Light Red
        self.redraw()

def _display_results_table(Res_1, Res_0, oldWG1, oldWG0, temp_comp_results_df):
    """Creates a Toplevel window to display the final results in colored tables."""
    W1 = Toplevel()
    W1.title("WG Table Recommendations")
    W1.minsize(600, 800)

    # --- VVL1 Table ---
    # Create a main container for this section
    vvl1_main_frame = Frame(W1)
    vvl1_main_frame.pack(fill='both', expand=True, padx=5, pady=5)

    # Place the label in the main container
    Label1 = tk.Label(vvl1_main_frame, text="VVL1 Recommended Table (Green=Higher, Red=Lower)", font=("Arial", 10, "bold"))
    Label1.pack(side='top', fill='x', pady=(0, 5))

    # Create a dedicated frame that will ONLY contain the pandastable widget
    table_frame1 = Frame(vvl1_main_frame)
    table_frame1.pack(fill='both', expand=True)

    # Create the table inside its dedicated frame, isolating it from other widgets
    pt1 = _ColoredTable(table_frame1, dataframe=Res_1, showtoolbar=True, showstatusbar=True)
    pt1.editable = False
    pt1.show()
    pt1.color_cells(Res_1.to_numpy(), oldWG1)


    # --- VVL0 Table ---
    vvl0_main_frame = Frame(W1)
    vvl0_main_frame.pack(fill='both', expand=True, padx=5, pady=5)

    Label0 = tk.Label(vvl0_main_frame, text="VVL0 Recommended Table", font=("Arial", 10, "bold"))
    Label0.pack(side='top', fill='x', pady=(0, 5))

    table_frame0 = Frame(vvl0_main_frame)
    table_frame0.pack(fill='both', expand=True)

    pt0 = _ColoredTable(table_frame0, dataframe=Res_0, showtoolbar=True, showstatusbar=True)
    pt0.editable = False
    pt0.show()
    pt0.color_cells(Res_0.to_numpy(), oldWG0)

    # --- Temperature Compensation Table ---
    # This new section displays the recommended temp comp table if one was generated.
    if temp_comp_results_df is not None:
        temp_comp_frame = Frame(W1)
        temp_comp_frame.pack(fill='both', expand=True, padx=5, pady=5)

        temp_label = tk.Label(temp_comp_frame, text="Recommended Temperature Compensation", font=("Arial", 10, "bold"))
        temp_label.pack(side='top', fill='x', pady=(0, 5))

        table_frame_temp = Frame(temp_comp_frame)
        table_frame_temp.pack(fill='both', expand=True)

        # Use a standard pandastable, as coloring isn't needed here.
        pt_temp = Table(table_frame_temp, dataframe=temp_comp_results_df, showtoolbar=True, showstatusbar=True)
        pt_temp.editable = False
        pt_temp.show()
    else:
        # If no recommendation could be made, inform the user.
        messagebox.showinfo(
            "Temperature Compensation",
            "Not enough data to calculate a recommended temperature compensation slope.",
            parent=W1
        )

    W1.mainloop()

# --- Main Function ---

def WG_tune(log, wgxaxis, wgyaxis, oldWG0, oldWG1, logvars, plot, WGlogic, tempcomp, tempcompaxis):
    """
    Main orchestrator for the WG tuning process.
    """
    # 1. Get user inputs for tuning parameters.
    params = _get_wg_tuning_parameters()

    # 2. Prepare and filter log data.
    log = _prepare_and_filter_log(log, params, logvars, WGlogic, tempcomp, tempcompaxis)

    # 3. Create bins and split data by VVL state.
    log = _create_bins_and_labels(log, wgxaxis, wgyaxis)
    log_VVL1 = log[log['VVL'] == 1].copy()
    log_VVL0 = log[log['VVL'] == 0].copy()

    # 4. Optionally plot the data for visual inspection.
    if plot:
        _plot_wg_data(log_VVL0, log_VVL1, wgxaxis, wgyaxis)

    # 5. Fit a 3D surface to the data for each VVL state.
    blend1 = _fit_surface(log_VVL1, wgxaxis, wgyaxis)
    blend0 = _fit_surface(log_VVL0, wgxaxis, wgyaxis)

    # 6. Calculate final recommendations based on the fit and confidence intervals.
    final_table_1 = _calculate_final_recommendations(log_VVL1, blend1, oldWG1, wgxaxis, wgyaxis)
    final_table_0, avg_coef = _calculate_final_recommendations(log_VVL0, blend0, oldWG0, wgxaxis, wgyaxis, calculate_temp_coef=True)

    # 7. Optionally visualize the results in 3D.
    if params['show_3d_plot']:
        # Create boolean masks to identify which cells have changed.
        changed_mask_1 = final_table_1 != oldWG1
        changed_mask_0 = final_table_0 != oldWG0

        # Visualize VVL1
        _plot_3d_surfaces(
            title="VVL1 3D Comparison (Changes Marked)",
            wgxaxis=wgxaxis,
            wgyaxis=wgyaxis,
            old_map=oldWG1,
            new_map=final_table_1,
            log_data=log_VVL1,
            WGlogic=WGlogic,
            changed_mask=changed_mask_1
        )
        # Visualize VVL0
        _plot_3d_surfaces(
            title="VVL0 3D Comparison (Changes Marked)",
            wgxaxis=wgxaxis,
            wgyaxis=wgyaxis,
            old_map=oldWG0,
            new_map=final_table_0,
            log_data=log_VVL0,
            WGlogic=WGlogic,
            changed_mask=changed_mask_0
        )

    # 8. Generate and prepare temperature compensation results.
    temp_comp_results_df = None
    if isinstance(avg_coef, (int, float)):
        # Calculate the intercept of the original temperature compensation table.
        _slope, original_intercept = np.polyfit(tempcompaxis, tempcomp, 1)

        # Calculate the new recommended table using the new slope and old intercept.
        new_tempcomp = (avg_coef * tempcompaxis) + original_intercept

        # Create a DataFrame for easy display.
        temp_df = pd.DataFrame({
            'Temperature': tempcompaxis,
            'Original Comp': tempcomp,
            'Recommended Comp': new_tempcomp
        })

        # Set 'Temperature' as the index and transpose the DataFrame to flip rows/columns.
        temp_comp_results_df = temp_df.set_index('Temperature').T.round(4)

    # 9. Prepare results as DataFrames and display them in a new window.
    exhlabels = [str(x) for x in wgxaxis]
    intlabels = [str(x) for x in wgyaxis]
    Res_1 = pd.DataFrame(final_table_1, columns=exhlabels, index=intlabels)
    Res_0 = pd.DataFrame(final_table_0, columns=exhlabels, index=intlabels)

    _display_results_table(Res_1, Res_0, oldWG1, oldWG0, temp_comp_results_df)

    return Res_1, Res_0
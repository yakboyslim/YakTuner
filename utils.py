# utils.py

import tkinter as tk
from tkinter import Toplevel, Frame
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pandastable import Table
import numpy as np

class ColoredTable(Table):
    """A pandastable Table subclass that colors cells based on change value."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.rowselectedcolor = None

    def color_cells(self, new_array, old_array, threshold=0.01):
        """Colors cells green for positive changes, red for negative."""
        self.resetColors()
        if new_array.shape != self.model.df.shape or old_array.shape != self.model.df.shape:
            return

        for r in range(new_array.shape[0]):
            for c in range(new_array.shape[1]):
                diff = new_array[r, c] - old_array[r, c]
                if diff > threshold:
                    self.setRowColors(rows=[r], cols=[c], clr='#90EE90')  # Light Green
                elif diff < -threshold:
                    self.setRowColors(rows=[r], cols=[c], clr='#FFB6C1')  # Light Red
        self.redraw()

def plot_3d_surface(title, xaxis, yaxis, old_map, new_map, log_data, changed_mask,
                    x_label, y_label, z_label, data_col_name, z_scale=1.0):
    """
    Generic function to create an interactive 3D plot for any tuning module.
    """
    if log_data.empty:
        return

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(xaxis, yaxis)

    # Aggregate data for plotting
    agg_data = log_data.groupby(['X', 'Y'])[data_col_name].agg(['mean', 'std']).reset_index().fillna(0)

    for _, row in agg_data.iterrows():
        x_idx, y_idx = int(row['X']), int(row['Y'])
        if x_idx < len(xaxis) and y_idx < len(yaxis):
            x_coord, y_coord = xaxis[x_idx], yaxis[y_idx]
            mean_val, std_val = row['mean'], row['std']
            ax.scatter(x_coord, y_coord, mean_val * z_scale, c='red', marker='o', s=20)
            ax.plot([x_coord, x_coord], [y_coord, y_coord],
                    [(mean_val - std_val) * z_scale, (mean_val + std_val) * z_scale],
                    marker="_", color='red', alpha=0.8)

    # Plot surfaces and changed cells
    ax.plot_wireframe(X, Y, old_map * z_scale, color='gray', alpha=0.7, label='Original Map')
    ax.plot_surface(X, Y, new_map * z_scale, cmap='viridis', alpha=0.6, label='Recommended Map')

    changed_y, changed_x = np.where(changed_mask)
    if changed_y.size > 0:
        x_coords, y_coords = xaxis[changed_x], yaxis[changed_y]
        z_coords = new_map[changed_y, changed_x] * z_scale + (0.01 * z_scale) # Z-offset
        ax.scatter(x_coords, y_coords, z_coords, c='magenta', marker='X', s=60, label='Changed Cells', depthshade=False)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_zlabel(z_label, fontsize=12)
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Original Map'),
        Patch(facecolor=plt.cm.viridis(0.5), edgecolor='k', label='Recommended Map'),
        Line2D([0], [0], marker='o', color='w', label='Mean Log Data', markerfacecolor='r', markersize=8),
        Line2D([0], [0], marker='_', color='r', label='Std. Dev. of Log Data', markersize=8, markeredgewidth=2),
        Line2D([0], [0], marker='X', color='w', label='Changed Cells', markerfacecolor='magenta', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show(block=True)
def WG_tune(log, wgxaxis, wgyaxis, oldWG0, oldWG1, logvars, plot, WGlogic):
    import numpy as np
    import pandas as pd
    from scipy import stats
    import tkinter as tk
    from tkinter import simpledialog
    from pandastable import Table
    import pwlf

    # Determine SWG/FF
    if WGlogic == 1:
        log['EFF'] = log['RPM']
        log['IFF'] = log['PUTSP'] * 10

    oldwgyaxis = wgyaxis.copy()
    oldwgxaxis = wgxaxis.copy()
    currentWG0 = oldWG0
    current = oldWG1

    # Get other inputs using tkinter dialog
    root = tk.Tk()
    root.withdraw()

    fudge = float(simpledialog.askstring("WG Inputs", "PUT fudge factor:", initialvalue="0.71"))
    minpedal = float(simpledialog.askstring("WG Inputs", "Minimum pedal (if no PUT_I_INHIBIT):", initialvalue="50"))
    maxdelta = float(simpledialog.askstring("WG Inputs", "Maximum PUT delta:", initialvalue="10"))
    minboost = float(simpledialog.askstring("WG Inputs", "Minimum Boost:", initialvalue="0"))

    # Create Derived Values
    log['deltaPUT'] = log['PUT'] - log['PUTSP']
    log['WGNEED'] = log['WG_Final'] - log['deltaPUT'] * fudge
    log['WGCL'] = log['WG_Final'] - log['WG_Base']

    # Create Trimmed datasets
    if 'I_INH' in logvars:
        log = log[log['I_INH'] <= 0]
    else:
        log = log[log['Pedal'] >= minpedal]

    if 'DV' in logvars:
        log = log[log['DV'] <= 50]

    if 'BOOST' in logvars:
        log = log[log['BOOST'] >= minboost]

    log = log[abs(log['deltaPUT']) <= maxdelta]
    log = log[log['WG_Final'] <= 98]
    log_WGopen = log.copy()

    if plot:
        # Plotting code would go here
        # Note: The original MATLAB plotting code uses UI components that would need
        # significant adaptation for Python. Consider using matplotlib or other Python
        # plotting libraries based on specific needs.
        pass

    # Create Bins
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

    exhlabels = [str(x) for x in wgxaxis]
    intlabels = [str(x) for x in wgyaxis]

    log_WGopen['X'] = pd.cut(log_WGopen['EFF'], wgxedges, labels=False)
    log_WGopen['Y'] = pd.cut(log_WGopen['IFF'], wgyedges, labels=False)

    log_VVL1 = log_WGopen[log_WGopen['VVL'] == 1]
    log_VVL0 = log_WGopen[log_WGopen['VVL'] == 0]

    # Initialize matrices
    SUM1 = np.zeros((len(wgyaxis), len(wgxaxis)))
    COUNT1 = np.zeros_like(SUM1)
    SUM0 = np.zeros_like(SUM1)
    COUNT0 = np.zeros_like(SUM1)
    columns1 = np.zeros((10, 16))
    rows1 = np.zeros_like(columns1)
    rows0 = np.zeros_like(columns1)
    columns0 = np.zeros_like(columns1)

    # Process VVL1 data

    current = oldWG1
    sigma = np.zeros_like(columns1)
    low = np.zeros_like(columns1)
    high = np.zeros_like(columns1)
    AVG1 = np.zeros_like(columns1)
    AVGtemp = np.zeros_like(columns1)

    for i in range(len(wgxaxis)):
        temp = log_VVL1.copy()
        temp = temp[temp.X == i]
        # if len(temp) > 1:
        fill = pd.DataFrame({"IFF": wgyaxis, "WGNEED": current[:, i]*100})
        temp = temp._append(fill, ignore_index=True)
        my_pwlf = pwlf.PiecewiseLinFit(temp.IFF, temp.WGNEED)
        my_pwlf.fit_with_breaks(wgyaxis)
        columns1[:, i] = my_pwlf.predict(wgyaxis)

    for j in range(len(wgyaxis)):
        temp = log_VVL1.copy()
        temp = temp[temp.Y == j]
        # if len(temp) > 1:
        fill = pd.DataFrame({"EFF": wgxaxis, "WGNEED": current[j, :]*100})
        temp = temp._append(fill, ignore_index=True)
        my_pwlf = pwlf.PiecewiseLinFit(temp.EFF, temp.WGNEED)
        my_pwlf.fit_with_breaks(wgxaxis)
        rows1[j, :] = my_pwlf.predict(wgxaxis)

    blend1 = (rows1+columns1)/200

    for i in range(len(wgxaxis)):
        for j in range(len(wgyaxis)):
            temp = log_VVL1[log_VVL1['X'] == i + 1]  # +1 to match MATLAB 1-based indexing
            temp = temp[temp['Y'] == j + 1]
            AVGtemp[j,i] = temp['WGNEED'].mean()
            COUNT1[j, i] = len(temp)
            # current[j, i] = interp2d(oldwgxaxis, oldwgyaxis, oldWG1)(wgxaxis[i], wgyaxis[j])[0]

            if COUNT1[j, i] > 3:
                # Fit normal distribution and get confidence intervals
                dist = stats.norm.fit(temp['WGNEED'])
                ci = stats.norm.interval(0.5, loc=dist[0], scale=dist[1])
                sigma[j, i] = dist[1]  # standard deviation
                low[j, i] = ci[0]
                high[j, i] = ci[1]

                if np.isnan(current[j, i]) or low[j, i] > current[j, i] or high[j, i] < current[j, i]:
                    AVG1[j, i] = (blend1[j, i] + AVGtemp[j,i] / 100) / 2
                else:
                    AVG1[j, i] = current[j, i]
            else:
                AVG1[j, i] = current[j, i]
    AVG1 = np.round(AVG1 * 16384) / 16384

    # Process VVL0 data

    current = oldWG0
    sigma = np.zeros_like(columns0)
    low = np.zeros_like(columns0)
    high = np.zeros_like(columns0)
    AVG0 = np.zeros_like(columns0)
    AVGtemp = np.zeros_like(columns0)

    for i in range(len(wgxaxis)):
        temp = log_VVL0.copy()
        temp = temp[temp.X == i]
        # if len(temp) > 1:
        fill = pd.DataFrame({"IFF": wgyaxis, "WGNEED": current[:, i]*100})
        temp = temp._append(fill, ignore_index=True)
        my_pwlf = pwlf.PiecewiseLinFit(temp.IFF, temp.WGNEED)
        my_pwlf.fit_with_breaks(wgyaxis)
        columns0[:, i] = my_pwlf.predict(wgyaxis)

    for j in range(len(wgyaxis)):
        temp = log_VVL0.copy()
        temp = temp[temp.Y == j]
        # if len(temp) > 1:
        fill = pd.DataFrame({"EFF": wgxaxis, "WGNEED": current[j, :]*100})
        temp = temp._append(fill, ignore_index=True)
        my_pwlf = pwlf.PiecewiseLinFit(temp.EFF, temp.WGNEED)
        my_pwlf.fit_with_breaks(wgxaxis)
        rows0[j, :] = my_pwlf.predict(wgxaxis)

    blend0 = (rows0+columns0)/200

    for i in range(len(wgxaxis)):
        for j in range(len(wgyaxis)):
            temp = log_VVL0[log_VVL0['X'] == i + 1]  # +1 to match MATLAB 1-based indexing
            temp = temp[temp['Y'] == j + 1]
            AVGtemp[j,i] = temp['WGNEED'].mean()
            COUNT0[j, i] = len(temp)
            # current[j, i] = interp2d(oldwgxaxis, oldwgyaxis, oldWG1)(wgxaxis[i], wgyaxis[j])[0]

            if COUNT0[j, i] > 3:
                # Fit normal distribution and get confidence intervals
                dist = stats.norm.fit(temp['WGNEED'])
                ci = stats.norm.interval(0.5, loc=dist[0], scale=dist[1])
                sigma[j, i] = dist[1]  # standard deviation
                low[j, i] = ci[0]
                high[j, i] = ci[1]

                if np.isnan(current[j, i]) or low[j, i] > current[j, i] or high[j, i] < current[j, i]:
                    AVG0[j, i] = (blend0[j, i] + AVGtemp[j,i] / 100) / 2
                else:
                    AVG0[j, i] = current[j, i]
            else:
                AVG0[j, i] = current[j, i]
    AVG0 = np.round(AVG0 * 16384) / 16384

    # Return results as pandas DataFrames
    Res_1 = pd.DataFrame(AVG1, columns=exhlabels, index=intlabels)
    Res_0 = pd.DataFrame(AVG0, columns=exhlabels, index=intlabels)

    class ColoredTable(Table):
        def __init__(self, parent=None, **kwargs):
            Table.__init__(self, parent, **kwargs)

        def color_cells(self, array1, array2):
            """Color cells based on comparison of two arrays"""

            # Reset all colors first
            self.resetColors()

            # Get the current DataFrame
            df = self.model.df

            # Ensure arrays are the same shape as the DataFrame
            if array1.shape != array2.shape or array1.shape != df.shape:
                raise ValueError("Arrays must have the same shape as the DataFrame")

            # Create color mapping
            colors = {
                'higher': '#90EE90',  # light green
                'lower': '#FFB6C1',  # light red
                'equal': '#FFFFFF'  # white
            }

            # Compare arrays and set colors
            for i in range(array1.shape[0]):
                for j in range(array1.shape[1]):
                    if array1[i, j] > array2[i, j]:
                        self.setRowColors(rows=[i], cols=[j], clr=colors['higher'])
                    elif array1[i, j] < array2[i, j]:
                        self.setRowColors(rows=[i], cols=[j], clr=colors['lower'])
                    else:
                        self.setRowColors(rows=[i], cols=[j], clr=colors['equal'])

            self.redraw()

    # Create the main window
    W1 = tk.Toplevel()
    W1.title("WG Tables")
    W1.minsize(500, 500)

    # Create a frame to hold the table
    frame1 = tk.Frame(W1)
    frame1.pack(fill='both', expand=True)
    frame0 = tk.Frame(W1)
    frame0.pack(fill='both', expand=True)

    tk.Label(frame1, text="VVl1")
    tk.Label(frame0, text="VVL0")

    # Create the table and add it to the frame
    pt1 = ColoredTable(frame1, dataframe=Res_1, showtoolbar=True, showstatusbar=True)
    pt1.editable = False
    pt1.show()
    pt1.color_cells(AVG1,oldWG1)
    pt1.rowselectedcolor = None
    pt0 = ColoredTable(frame0, dataframe=Res_0, showtoolbar=True, showstatusbar=True)
    pt0.editable = False
    pt0.show()
    pt0.color_cells(AVG0,oldWG0)
    pt0.rowselectedcolor = None

    def on_closing():
        W1.quit()
        W1.destroy()

    W1.protocol("WM_DELETE_WINDOW", on_closing)


    # Start the main loop
    W1.mainloop()

    return Res_1, Res_0
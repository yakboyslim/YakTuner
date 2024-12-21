def MAF_tune(log, mafxaxis, mafyaxis, maftables, combmodes, logvars):
    import numpy as np
    import pandas as pd
    from scipy import stats
    from tkinter import simpledialog
    import tkinter as tk
    from pandastable import Table
    import matplotlib.pyplot as plt
    import pwlf

    # Create input dialog for confidence
    root = tk.Tk()
    root.withdraw()
    conf = float(simpledialog.askstring("MAF Correction Inputs",
                                        "Confidence required to make change:",
                                        initialvalue="0.75"))
    maxcount = 100

    # Create Derived Values
    log['MAP'] = log['MAP'] * 10

    if 'LAM_DIF' not in logvars:
        log['LAM_DIF'] = 1/log['LAMBDA_SP'] - 1/log['LAMBDA']

    if "FAC_MFF_ADD_FAC_LAM_AD (%)" in logvars:
        log['LTFT'] = log["FAC_MFF_ADD_FAC_LAM_AD (%)"]

    if 'FAC_LAM_OUT' in logvars:
        log['STFT'] = log['FAC_LAM_OUT']

    log['ADD_MAF'] = log['STFT'] + log['LTFT'] - log['LAM_DIF']

    if 'MAF_COR' in logvars:
        log['ADD_MAF'] = log['ADD_MAF'] + log['MAF_COR']

    log = log[log['state_lam'] == 1]

    # Create axis labels
    xlabels = [str(x) for x in mafxaxis]
    ylabels = [str(y) for y in mafyaxis]

    # Create Bins
    xedges = np.zeros(len(mafxaxis) + 1)
    xedges[0] = 0
    xedges[-1] = np.inf
    for i in range(len(mafxaxis)-1):
        xedges[i+1] = (mafxaxis[i] + mafxaxis[i+1])/2

    yedges = np.zeros(len(mafyaxis) + 1)
    yedges[0] = 0
    yedges[-1] = np.inf
    for i in range(len(mafyaxis)-1):
        yedges[i+1] = (mafyaxis[i] + mafyaxis[i+1])/2

    # Discretize MAF
    log['X'] = pd.cut(log['RPM'], bins=xedges, labels=False)
    log['Y'] = pd.cut(log['MAP'], bins=yedges, labels=False)

    results = {}
    changes = {}

    for IDX in range(4):
        SUM = np.zeros((len(mafyaxis), len(mafxaxis)))
        COUNT = np.zeros_like(SUM)
        AVG = np.zeros_like(SUM)
        columns = np.zeros((12, 8))
        rows = np.zeros_like(columns)
        current = maftables[IDX]
        sigma = np.zeros_like(AVG)
        low = np.zeros_like(AVG)
        high = np.zeros_like(AVG)
        IDXmodes = np.where(combmodes == IDX)[0]

        if 'MAF_COR' in logvars:
            test = current.copy()
        else:
            test = np.zeros((len(mafyaxis), len(mafxaxis)))

        temp1 = log.copy()
        diffcmb = list(set(log['CMB']) - set(IDXmodes))
        temp1 = temp1[~temp1['CMB'].isin(diffcmb)]

        for i in range(len(mafxaxis)):
            temp = temp1[temp1['X'] == i]
            # if len(temp) > 1:
            fill = pd.DataFrame({"MAP": mafyaxis, "ADD_MAF": current[:, i]})
            temp = temp._append(fill, ignore_index=True)
            my_pwlf = pwlf.PiecewiseLinFit(temp.MAP, temp.ADD_MAF)
            my_pwlf.fit_with_breaks(mafyaxis)
            columns[:, i] = my_pwlf.predict(mafyaxis)

        for j in range(len(mafyaxis)):
            temp = temp1[temp1['Y'] == j]
            # if len(temp) > 1:
            fill = pd.DataFrame({"RPM": mafxaxis, "ADD_MAF": current[j, :]})
            temp = temp._append(fill, ignore_index=True)
            my_pwlf = pwlf.PiecewiseLinFit(temp.RPM, temp.ADD_MAF)
            my_pwlf.fit_with_breaks(mafxaxis)
            rows[j, :] = my_pwlf.predict(mafxaxis)


        blend = (columns + rows)/2
        interpfac = 0.25

        for i in range(len(mafxaxis)):
            for j in range(len(mafyaxis)):
                temp = temp1[(temp1['X'] == i) & (temp1['Y'] == j)]
                COUNT[j, i] = len(temp)
                if COUNT[j, i] > 3:
                    # Fit normal distribution and get confidence intervals
                    dist = stats.norm.fit(temp['ADD_MAF'])
                    ci = stats.norm.interval(0.5, loc=dist[0], scale=dist[1])
                    sigma[j, i] = dist[1]  # standard deviation
                    low[j, i] = ci[0]
                    high[j, i] = ci[1]

                    if low[j, i] > test[j, i]:
                        AVG[j, i] = (blend[j, i] * interpfac + low[j, i] * (1-interpfac))
                    elif high[j, i] < test[j, i]:
                        AVG[j, i] = (blend[j, i] * interpfac + high[j, i] * (1-interpfac))
                    else:
                        AVG[j, i] = test[j, i]
                else:
                    AVG[j, i] = test[j, i]

        COUNT = np.minimum(COUNT, maxcount)
        COUNT = COUNT/maxcount

        COUNT[np.isnan(COUNT)] = 0
        AVG[np.isnan(AVG)] = 0
        CHANGE = (AVG - test) * COUNT
        NEW = np.round((current + CHANGE) * 5.12) / 5.12

        field_name = f'IDX{IDX}'
        results[field_name] = pd.DataFrame(NEW, columns=xlabels, index=ylabels)
        changes[field_name] = CHANGE

    class ColoredTable(Table):
        def __init__(self, parent=None, **kwargs):
            Table.__init__(self, parent, **kwargs)

        def color_cells(self, array1):
            """Color cells based on comparison of two arrays"""

            # Reset all colors first
            self.resetColors()

            # Get the current DataFrame
            df = self.model.df

            # Ensure arrays are the same shape as the DataFrame
            if array1.shape != df.shape:
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
                    if array1[i, j] > 0:
                        self.setRowColors(rows=[i], cols=[j], clr=colors['higher'])
                    elif array1[i, j] < 0:
                        self.setRowColors(rows=[i], cols=[j], clr=colors['lower'])
                    else:
                        self.setRowColors(rows=[i], cols=[j], clr=colors['equal'])

            self.redraw()

    # Create the main window
    W1 = tk.Toplevel()
    W1.title("MAF Tables")

    # Create a frame to hold the table
    frame0 = tk.Frame(W1, width=450, height=200)
    frame1 = tk.Frame(W1, width=450, height=200)
    frame2 = tk.Frame(W1, width=450, height=200)
    frame3 = tk.Frame(W1, width=450, height=200)

    # Configure grid weights to make frames expand equally
    W1.grid_rowconfigure(0, weight=1)
    W1.grid_rowconfigure(1, weight=1)
    W1.grid_columnconfigure(0, weight=1)
    W1.grid_columnconfigure(1, weight=1)

    # Place frames in a 2x2 grid
    frame0.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    frame1.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    frame2.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    frame3.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    tk.Label(frame0, text="IDX0")
    tk.Label(frame1, text="IDX1")
    tk.Label(frame2, text="IDX2")
    tk.Label(frame3, text="IDX3")


    # Create the table and add it to the frame
    pt0 = ColoredTable(frame0, dataframe=results[f"IDX{0}"], showtoolbar=True, showstatusbar=True)
    pt0.editable = False
    pt0.show()
    pt0.color_cells(changes[f"IDX{0}"])
    pt0.rowselectedcolor = None

    pt1 = ColoredTable(frame1, dataframe=results[f"IDX{1}"], showtoolbar=True, showstatusbar=True)
    pt1.editable = False
    pt1.show()
    pt1.color_cells(changes[f"IDX{1}"])
    pt1.rowselectedcolor = None

    pt2 = ColoredTable(frame2, dataframe=results[f"IDX{2}"], showtoolbar=True, showstatusbar=True)
    pt2.editable = False
    pt2.show()
    pt2.color_cells(changes[f"IDX{2}"])
    pt2.rowselectedcolor = None

    pt3 = ColoredTable(frame3, dataframe=results[f"IDX{3}"], showtoolbar=True, showstatusbar=True)
    pt3.editable = False
    pt3.show()
    pt3.color_cells(changes[f"IDX{3}"])
    pt3.rowselectedcolor = None

    def on_closing():
        W1.quit()
        W1.destroy()

    W1.protocol("WM_DELETE_WINDOW", on_closing)


    # Start the main loop
    W1.mainloop()

    return results
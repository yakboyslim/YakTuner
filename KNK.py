def KNK(log, igxaxis, igyaxis, logvars, bin, IGNmaps, bin_path):
    import numpy as np
    import pandas as pd
    from scipy import stats
    import tkinter as tk
    from tkinter import simpledialog
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from pandastable import Table
    import BinRead
    matplotlib.use('TkAgg')

    # Create input dialog
    root = tk.Tk()
    root.withdraw()

    maxadv = float(simpledialog.askstring("Knock Inputs", "Maximum advance if no knock seen:", initialvalue="0.75"))
    conf = 1 - float(
        simpledialog.askstring("Knock Inputs", "Confidence Required to confirm knock:", initialvalue="0.75"))

    # Map selection dialog
    options = [str(i) for i in range(1, 6)] + ['No map switching']
    map_dialog = tk.Tk()
    map_dialog.withdraw()
    map = simpledialog.askinteger("Select Map", "Select a SP Map 1-5 (0 is no SP Map):", minvalue=0, maxvalue=5)

    # # Create dialog for variable selection
    # map_dialog = tk.Toplevel()
    # map_dialog.title(f"Select Map")
    # map_dialog.geometry("400x400")
    #
    # listbox = tk.Listbox(map_dialog, selectmode='single')
    # listbox.pack()
    #
    # options = [str(i) for i in range(1, 6)] + ['No map switching']
    #
    # for opt in options:
    #     listbox.insert(tk.END, opt)
    #
    # def on_select():
    #     selection = listbox.curselection()
    #     if selection:
    #         map = [selection[0] - 1]
    #     map_dialog.destroy()
    #
    # tk.Button(map_dialog, text="Select", command=on_select).pack()
    # map_dialog.wait_window()

    if map == 0:
        currentIG = [np.zeros((16, 16))]
    else:
        currentIG = IGNmaps[map - 1]

    # Create Derived Values
    log['MAP'] = log['MAP'] * 10

    # Read Axis Values
    xlabels = [str(x) for x in igxaxis]
    ylabels = [str(y) for y in igyaxis]

    # Create Bins
    xedges = [0] + [(igxaxis[i] + igxaxis[i + 1]) / 2 for i in range(len(igxaxis) - 1)] + [float('inf')]
    yedges = [0] + [(igyaxis[i] + igyaxis[i + 1]) / 2 for i in range(len(igyaxis) - 1)] + [float('inf')]

    # Initialize matrices
    SUM = np.zeros((len(igyaxis), len(igxaxis)))
    COUNT = np.zeros_like(SUM)
    AVG = np.zeros_like(SUM)

    # Create Trimmed Dataset
    log['knkoccurred'] = 0
    allcyl = np.column_stack((log['KNK1'], log['KNK2'], log['KNK3'], log['KNK4']))
    mincyl = np.min(allcyl, axis=1)
    outliercyl = stats.zscore(allcyl, axis=1)
    outliercyl = outliercyl * allcyl / mincyl[:, np.newaxis]
    outliercyl = np.nan_to_num(outliercyl)
    row, cyl = np.where(outliercyl)
    log.loc[row, 'singlecyl'] = cyl + 1
    log['KNKAVG'] = np.mean(allcyl, axis=1)

    # Calculate knock occurred
    for i in range(1, len(log)):
        if any([log['KNK1'].iloc[i - 1] - log['KNK1'].iloc[i] > 1,
                log['KNK2'].iloc[i - 1] - log['KNK2'].iloc[i] > 0,
                log['KNK3'].iloc[i - 1] - log['KNK3'].iloc[i] > 0,
                log['KNK4'].iloc[i - 1] - log['KNK4'].iloc[i] > 0]):
            log.loc[i, 'knkoccurred'] = 1

    # Discretize All data
    log['X'] = pd.cut(log['RPM'], bins=xedges, labels=range(1, len(igxaxis) + 1))
    log['Y'] = pd.cut(log['MAF'], bins=yedges, labels=range(1, len(igyaxis) + 1))

    # Process data
    KR = np.zeros_like(SUM)
    sigma = np.zeros_like(SUM)
    low = np.zeros_like(SUM)
    high = np.zeros_like(SUM)

    for i in range(len(igxaxis)):
        for j in range(len(igyaxis)):
            temp = log[(log['X'] == i + 1) & (log['Y'] == j + 1)]
            tempKNK = temp[temp['knkoccurred'] == 1]
            KR[j, i] = tempKNK['KNKAVG'].mean() if len(tempKNK) > 0 else 0
            COUNT[j, i] = len(temp)

            if COUNT[j, i] > 3:
                ci = norm.interval(1 - conf, loc=temp['KNKAVG'].mean(), scale=temp['KNKAVG'].std())
                sigma[j, i] = (ci[1] - ci[0]) / 2
                low[j, i] = ci[0]
                high[j, i] = ci[1]

                if high[j, i] < 0:
                    AVG[j, i] = (high[j, i] + KR[j, i]) / 2
                elif high[j, i] == 0 and igxaxis[i] > 2500 and igyaxis[j] > 700:
                    AVG[j, i] = min(maxadv * COUNT[j, i] / 100, maxadv)
                else:
                    AVG[j, i] = 0

    AVG = np.nan_to_num(AVG)
    AVG = np.minimum(AVG, maxadv)
    inter = np.ceil(AVG * 5.33333333) / 5.33333333
    NEW = np.round(inter * 2.666666666666667) / 2.666666666666667

    resarray = NEW + currentIG[0]
    Res_KNK = pd.DataFrame(resarray, columns=xlabels, index=ylabels)


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



    # Plotting

    plt.figure()
    plottemp = log[log['knkoccurred'] == 1]
    plt.scatter(plottemp['RPM'], plottemp['MAF'], s=abs(plottemp['KNKAVG']) * 100, c=plottemp['singlecyl'], cmap='Set1')
    plt.colorbar(label='Cylinder', ticks=[1, 2, 3, 4])
    plt.gca().invert_yaxis()
    plt.xlabel('RPM')
    plt.ylabel('MAF')
    plt.grid(True)
    plt.xticks(igxaxis)
    plt.yticks(igyaxis)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.set_ticks_position('top')
    plt.show(block=True)

    # Create the main window
    W1 = tk.Toplevel()
    W1.title("SP IGN Table")
    W1.minsize(500, 500)

    # Create a frame to hold the table
    frame1 = tk.Frame(W1)
    frame1.pack(fill='both', expand=True)

    tk.Label(frame1, text="VVl1")

    # Create the table and add it to the frame
    pt1 = ColoredTable(frame1, dataframe=Res_KNK, showtoolbar=True, showstatusbar=True)
    pt1.editable = False
    pt1.show()
    pt1.color_cells(NEW)
    pt1.rowselectedcolor = None

    def on_closing():
        W1.quit()
        W1.destroy()

    W1.protocol("WM_DELETE_WINDOW", on_closing)


    # Start the main loop
    W1.mainloop()

    return Res_KNK
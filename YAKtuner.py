# Read Log
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import pandas as pd

import BinRead
import KNK
import MAF
import WG

# Get Variable list
current_dir = os.getcwd()
varconv = pd.read_csv(os.path.join(current_dir, "variables.csv"), header=None, dtype=str)

# Create GUI window
root = ctk.CTk()
root.title("YakTuner")
root.geometry("400x200+500+500")

# Create checkboxes
cbx1 = ctk.CTkCheckBox(root, text="Tune WG?")
cbx1.place(x=20, y=20)

# cbx2 = ctk.CTkCheckBox(root, text="Edit WG Axis?")
# cbx2.place(x=40, y=50)

cbx6 = ctk.CTkCheckBox(root, text="SWG?")
cbx6.place(x=160, y=50)

cbx3 = ctk.CTkCheckBox(root, text="Tune MAF?")
cbx3.place(x=140, y=20)

cbx9 = ctk.CTkCheckBox(root, text="Tune Ignition?")
cbx9.place(x=260, y=20)

cbx7 = ctk.CTkCheckBox(root, text="S50")
cbx7.place(x=20, y=100)
cbx7.select()

cbx8 = ctk.CTkCheckBox(root, text="A05")
cbx8.place(x=100, y=100)

cbx11 = ctk.CTkCheckBox(root, text="V30")
cbx11.place(x=180, y=100)

cbx12 = ctk.CTkCheckBox(root, text="Custom FileStruct")
cbx12.place(x=260, y=100)

cbx5 = ctk.CTkCheckBox(root, text="Output results to CSV?")
cbx5.place(x=20, y=130)

cbx10 = ctk.CTkCheckBox(root, text="Reset Variables?")
cbx10.place(x=260, y=130)


def on_continue():
    root.quit()


continue_btn = ctk.CTkButton(root, text="CONTINUE", command=on_continue)
continue_btn.place(x=150, y=160)

root.mainloop()

# Get checkbox values
WGtune = cbx1.get()
MAFtune = cbx3.get()
IGtune = cbx9.get()
# plot = cbx2.get()
plot = False
save = cbx5.get()
WGlogic = cbx6.get()
S50 = cbx7.get()
A05 = cbx8.get()
V30 = cbx11.get()
custom = cbx12.get()
var_reset = cbx10.get()

root.destroy()

varconv = varconv.to_numpy()

# Open Files
file_paths = filedialog.askopenfilenames(
    title="Select Log CSV File",
    filetypes=[("CSV files", "*.csv")]
)

if len(file_paths) > 0:
    log = pd.read_csv(file_paths[0])
    log = log.iloc[:, :-1]  # Remove last column

    # Combine multiple logs if selected
    for file_path in file_paths[1:]:
        temp_log = pd.read_csv(file_path)
        temp_log = temp_log.iloc[:, :-1]
        log = pd.concat([log, temp_log], ignore_index=True)

# Select bin file
bin_path = filedialog.askopenfilename(
    title="Select Bin File",
    filetypes=[("Binary files", "*.bin")]
)

cust_struct = pd.read_csv(os.path.join(current_dir, "structure.csv"), header=None, dtype=str)

with open(bin_path, 'rb') as bin_file:
    # Parse bin
    if S50:
        address = [0x24B642, 0x24B62A, 0x24BB66, 0x21974A, 0x2196FC, 0x2199F6, 0x219B36, 0x23CDBC, 0x23CE5A]
    elif A05:
        address = [0x277EB8, 0x277EA0, 0x27837C, 0x23D0E0, 0x23D092, 0x23D368, 0x23D4A8, 0x267A3E, 0x267ADE]
    elif V30:
        address = [0x22D926, 0x22D90E, 0x22DE5C, 0x212E76, 0x212E28, 0x213134, 0x213274, 0x230634, 0x2306D2]
    elif custom:
        address = [int(hex_str, 16) for hex_str in cust_struct[1][:9].tolist()]
    else:
        messagebox.showerror("Error", "Must select File Structure")
        exit()

    rows = [1, 1, 1, 1, 1, 10, 10, 1, 1]
    cols = [39, 12, 8, 10, 16, 16, 16, 16, 16]
    offset = [0, 0, 0, 0, 0, 32768, 32768, 0, 0]
    res = [1, 12.06, 1, 16384, 16384, 16384, 16384, 23.59071274298056, 1]

    if WGlogic:
        res[3] = 1 / 0.082917524986648
        res[4] = 1

    prec = ["uint8", "uint16", "uint16", "uint16", "uint16", "uint16", "uint16", "uint16", "uint16"]
    req = [address, rows, cols, offset, res, prec]
    output = BinRead.bin_read(bin_file, req)

    combmodes = np.ravel(output[0])
    mafyaxis = np.ravel(output[1])
    mafxaxis = np.ravel(output[2])
    wgyaxis = np.ravel(output[3])
    wgxaxis = np.ravel(output[4])
    currentWG0 = output[5]
    currentWG1 = output[6]
    igyaxis = np.ravel(output[7])
    igxaxis = np.ravel(output[8])



    # Continue with MAF addresses
    if MAF:
        if S50:
            address = [0x24B669, 0x24B6C9, 0x24B729, 0x24B789]
        elif A05:
            address = [0x277EDF, 0x277F3F, 0x277F9F, 0x277FFF]
        elif V30:
            address = [0x22D94D, 0x22D9AD, 0x22DA0D, 0x22DA6D]
        elif custom:
            address = [int(hex_str, 16) for hex_str in cust_struct[1][9:13].tolist()]

        rows = [12, 12, 12, 12]
        cols = [8, 8, 8, 8]
        offset = [128, 128, 128, 128]
        res = [5.12, 5.12, 5.12, 5.12]
        prec = ["uint8", "uint8", "uint8", "uint8"]
        mafreq = [address, rows, cols, offset, res, prec]
        maftables = BinRead.bin_read(bin_file, mafreq)

   # Continue with SP IGN Maps
    if KNK:
        if S50:
            address = [0x27CF1A, 0x27D01A, 0x27D11A, 0x27D21A, 0x27D31A]
        elif A05:
            address = [0x2AFE7A, 0x2AFF9A, 0x2B00BA, 0x2B01DA, 0x2B02FA]
        elif V30:
            address = [0x13CF1A, 0x13D01A, 0x13D11A, 0x13D21A, 0x13D31A]
        elif custom:
            address = [int(hex_str, 16) for hex_str in cust_struct[1][13:17].tolist()]
        rows = [16, 16, 16, 16, 16]
        cols = [16, 16, 16, 16, 16]
        offset = [95, 95, 95, 95, 95]
        res = [2.666666666667, 2.666666666667, 2.666666666667, 2.666666666667, 2.666666666667]
        prec = ["uint8", "uint8", "uint8", "uint8", "uint8"]
        IGNreq = [address, rows, cols, offset, res, prec]
        IGNmaps = BinRead.bin_read(bin_file, IGNreq)  # Assuming BinRead function exists or is imported

# Convert Variables
logvars = log.columns.tolist()
missing_vars = []

if var_reset:
    missing_vars = list(range(1, varconv.shape[1]))
else:
    for i in range(1, varconv.shape[1]):
        if any(var == varconv[0, i] for var in logvars):
            log = log.rename(columns={varconv[0, i]: varconv[1, i]})
        else:
            missing_vars.append(i)

if missing_vars:
    for i in missing_vars:
        # Create dialog for variable selection
        var_window = tk.Toplevel()
        var_window.title(f"Select variable for: {varconv[2, i]}")
        var_window.geometry("400x400")

        listbox = tk.Listbox(var_window, selectmode='single')
        listbox.pack()

        options = [varconv[0, i]] + logvars
        for opt in options:
            listbox.insert(tk.END, opt)


        def on_select():
            selection = listbox.curselection()
            if selection and selection[0] > 0:
                varconv[0, i] = logvars[selection[0] - 1]
                log = log.rename(columns={varconv[0, i]: varconv[1, i]})
            var_window.destroy()


        tk.Button(var_window, text="Select", command=on_select).pack()
        var_window.wait_window()

    pd.DataFrame(varconv).to_csv(os.path.join(current_dir, "variables.csv"), header=False, index=False)

logvars = log.columns.tolist()

 #Tunes
if WGtune:
    Res_1, Res_0 = WG.WG_tune(log, wgxaxis, wgyaxis, currentWG0, currentWG1, logvars, plot, WGlogic)
    if save:
        Res_1.to_csv(os.path.join(os.path.dirname(file_paths[0]), "VVL1 Results.csv"))
        Res_0.to_csv(os.path.join(os.path.dirname(file_paths[0]), "VVL0 Results.csv"))

if MAFtune:
    MAFresults = MAF.MAF_tune(log, mafxaxis, mafyaxis, maftables, combmodes, logvars)
    if save:
        for idx in range(4):
            MAFresults[f"IDX{idx}"].to_csv(
                os.path.join(os.path.dirname(file_paths[0]), f"MAF_STD[{idx}] Results.csv")
            )

if IGtune:
    Res_KNK = KNK.KNK(log, igxaxis, igyaxis, logvars, bin_file, IGNmaps, bin_path)
    if save:
        Res_KNK.to_csv(os.path.join(os.path.dirname(file_paths[0]), "KNK Results.csv"))

# YakWG
----
## What's it do?
This program will open log files and determine areas of WOT and suggest WG Feedforward values based on the WG Final and PUT difference.
Additionally it can tune MAF_STD tables, which can help with fuel trims

----
## How to use WG Tuner

1. Click the YakTuner.exe
1. Follow the prompts, selecting Tune WG. Plots are optional, and a bad idea if you have more than a single pull log.
2. Change first row of variables to match your logs if necessary.(I_INH is optional)
2. Copy the x-axis and y-axis scalars from you WG tables into the WG x-axis and WG y-axis fields (Either what you have stock, or what you want to use if shifting axis values)
5. Select your log. Can select multiple at once as long as they are from an identical PID list.
6. Choose FF or SWG logic
7. Choose fudge factor and trim values. Defaults should work fine.
8. The program executes and will show a few plots (if selected), and open two output tables. Only cells for which data was found in the logs will have recommended values. Feel free to enter these values, but as always use caution and common sense. Smooth as necessary into surrounding cells.

----
## How to use MAF Tuner

1. Copy your IP_FAC_MAF_COR[0] table (usually the first of 4 tables titled "Global MAF_STND correction factor (High-res-interpolation) with selectable array by ID_IDX_CMB_MOD_SEL") into a csv file. An example with all zeros is provided in the install folder. Ensure the axis values are correct as well.
2. Click the YakTuner.exe
1. Follow the prompts, selecting Tune MAF.
2. Change first row of variables to match your logs if necessary. Engine State is not a standard PID, but is required for this to work.
5. Select your log. Can select multiple at once as long as they are from an identical PID list.
6. Select the CSV file containig your current MAF_STD table.
7. Choose confidence factors. Defaults should work fine.
8. The program executes and will open and output tables. Only cells for which data was found to justify a change will be highlighted, the others are simply a copy of stock values. Feel free to enter these values, but as always use caution and common sense. This table is often non-intuitive.

----
## How to install
1. Download an installer from the releases tab.
2. Open this file, and approve any security requests.
The installer and program were compiled using MATLAB. On first install you may need to install the MATLAB runtime environment from the internet.

## Required PIDS for MAF Tuning
### state_eng|"Engine operating state"
#### S50
Eng State,x,%01.0f,0xd000c5cb,1,FALSE,0,6,-1000,1000,0,TRUE
#### A05
Eng State,x,%01.0f,0xd0012ac7,1,FALSE,0,6,-1000,1000,0,TRUE

## Optional PIDS recommended for WG tuning
### lv_inh_put_ctl_i|"Flag to inhibit PUT Controller I share"
#### S50
PUT I Inhibit,x,%01.0f,0xd0000b76,1,FALSE,0,1,-1000,1000,0,TRUE
#### A05
PUT I Inhibit,x,%01.0f,0xd00005b8,1,FALSE,0,1,-1000,1000,0,TRUE

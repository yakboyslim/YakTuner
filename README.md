# YakWG
----
## What's it do?
This program will open log files and determine areas of WOT and suggest WG Feedforward values based on the WG Final and PUT difference.

----
## How to use

1. Click the WG_Tuner.exe
1. Select 'Yes' to edit variables and axis if it is the first time, or you need to make changes
2. Change first row of variables to match your logs.(I_INH is optional)
2. Copy the x-axis and y-axis scalars from you WG tables into the x-axis and y-axis fields (Either what you have stock, or what you want to use if shifting axis values)
5. Select your log. Can select multiple at once.
6. Choose FF or SWG logic
7. Choose fudge factor and trim values. Defaults should work fine.
8. The program executes and will show a few plots, and open two output tables. Only cells for which data was found in the logs will have recommended values. Feel free to enter these values, but as always use caution and common sense. Smooth as necessary into surrounding cells.

----
## How to install
1. Download an installer from the releases tab.
2. Open this file, and approve any security requests.
The installer and program were compiled using MATLAB. On first install you may need to install the MATLAB runtime environment from the internet.

## Optional PIDS
### lv_inh_put_ctl_i|"Flag to inhibit PUT Controller I share"
#### S50
PUT I Inhibit,x,%01.0f,0xd0000b76,1,FALSE,0,1,-1000,1000,0,TRUE
#### A05
PUT I Inhibit,x,%01.0f,0xd00005b8,1,FALSE,0,1,-1000,1000,0,TRUE

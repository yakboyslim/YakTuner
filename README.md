# YakTUNER
----
## What's it do?
Opens logs and bins. Suggests tune changes for WG, MAF (for tuning out fuel trims) and IGN.
This program does not "tune for you". It is a tool, to give data based recommendations that you can CHOOSE to apply to your tune.
As always, tuning is done at your own risk.

### WG
The program takes logs and trims the dataset to only the portion that would be used to adjust the WG.
It then considers both over/under boost and the closed loop portion of the boost controller. Using both a 2-d piecewise fit, and binned averages it calculates what it believes the WG feed forward table should have been.
It then uses a confidence interval to decide if there is a statistical basis to apply the change. If there is it reccommends a change to the table.

### MAF
This works similar to the WG program, except uses data about fuel trims and rich/lean of target lambda to reccommend changes to the MAF correction tables. The important thing to note with this tool is it requires logs of ALL types of driving - idling, cruising, open throttle. And the logs MUST be done with the car fully warm (oil up to temp and stabilized)
This level of tuning _could_ reduce fuel trims, which _may_ allow the car to handle transient fuel requests better... maybe.
This is not a tool to correct fueling issues. The tuning explained in the guide, as well as any MPI tuning must be done FIRST before using this tool.

## KNK

This program takes logs and looks for instances of knock. It then determines whether that knock is across the engine (real knock that timing will fix) or one cylinder (something else).
It then uses confidence intervals to determine if timing should be lowered in that cell. It will then either output a list of recommended reductions, or an updated SP timing map.
Since this tool throws away single cylinder events it cannot replace looking at logs in datazap, MegalogViewer, etc. It is simply a tool to help identify areas where timing should be reduced for the fuel used.
It has a function that may recommend advancing timing, but only rarely with a lot of data and no knock observed. Also this tool does not consider MBT tables, etc, so is not really a tool to hunt "optimal timing".

----

## Required PIDS
In "PID LISTS" folder.

----

## Installation
Install by downloading from releases. The two files (YAKtuner.exe and variables.csv) can be placed anywhere, but must be in the same folder. Run by double clicking the executable.
----

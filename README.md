# YAKtuner User Guide

---

Welcome to YAKtuner! This guide will walk you through setting up and using the application to analyze your engine data logs and generate recommendations for your tune.

## 1. What is YAKtuner?

YAKtuner is an automated tuning assistant designed to process data logs from your vehicle's ECU. It compares your log data against the tables in your current tune file (`.bin`) and provides data-driven recommendations for improvement. It can help you dial in your **Wastegate (WG)**, **Mass Airflow (MAF)**, **Fueling (MFF)**, **Ignition Timing (KNK)**, and **Low-Pressure Fuel Pump (LPFP)** tables.

## 2. Getting Started: Required Files

Before you begin, make sure you have the following files in the same folder as `YAKtuner.exe`:

1.  **YAKtuner.exe**: The main application.
2.  **map_definitions.csv**: A critical configuration file that tells the application where to find the maps (e.g., `wgpid0`, `maftable1`) inside your `.bin` file for your specific firmware.
3.  **variables.csv**: This file links the column headers from your data logs (e.g., "Engine Speed (1/min)") to the standard names the application uses internally (e.g., "RPM").

You will also need:

* **Data Log Files (.csv)**: One or more data logs from your vehicle. For best results, these should cover a wide range of operating conditions (idle, cruise, and wide-open throttle pulls).
* **Your Tune File (.bin)**: The exact binary tune file that was loaded on the ECU when the data logs were recorded.

## 3. Running YAKtuner: The Workflow

Running the application is a straightforward process.

### Step 1: Configure Your Tune

When you launch YAKtuner, you'll be greeted with the main settings window.

* **Tuning Modules**: Check the boxes for the modules you want to run.
    * **Tune WG?**: Enables the Wastegate tuner. You can also select **Use SWG Logic?** if your tune uses "Simple Wastegate" logic.
    * **Tune MAF? / Tune MFF?**: Enables the fuel correction tuners.
    * **Tune Ignition?**: Enables the knock and timing advance tuner.
    * **Tune LPFP PWM?**: Enables the Low-Pressure Fuel Pump tuner. Select **2WD** or **4WD** for the correct table.
* **Ignition Tuner Settings**: When **Tune Ignition?** is checked, this section becomes active.
    * **Max Advance**: The maximum degrees of timing the tuner is allowed to add in knock-free cells. A safe starting point is 0.75.
    * **SP Map**: The base ignition map you want to apply corrections to (1-5). 6 is the Flex Fuel modifier map, and 0 will apply corrections to a zeroed-out table (so you could manually apply them to the correct base table).
* **Firmware Version**: Select the firmware your `.bin` file is based on (e.g., S50, A05, V30). This is crucial for finding the correct map addresses.
* **Other Options**:
    * **Save Results to CSV?**: If checked, the recommended tables will be saved as `.csv` files in the same folder as your logs.
    * **Reset Variable Mappings?**: Check this if you've changed logging software or want to force the application to re-map all your log headers.

### Step 2: Select Your Files

After clicking "CONTINUE", you will be prompted to:

1.  Select one or more log files (`.csv`).
2.  Select your single binary tune file (`.bin`).

### Step 3: Map Log Variables (If Necessary)

If this is your first time running the app or a variable name in your log doesn't match the `variables.csv` file, a new window will pop up. It will ask you to select the correct column from your log file that corresponds to the required internal variable. This mapping is saved automatically for future runs.

The application will now process your data and display the results.

## 4. Understanding the Tuning Modules

Each module analyzes a different aspect of your engine's performance.

### WG (Wastegate) Tuner

* **Synopsis**: This module fine-tunes your wastegate duty cycle tables to make your actual boost pressure match your target boost pressure more accurately, reducing over- or under-boosting.
* **How it Works**: It calculates the difference between your target and actual boost pressure (**deltaPUT**). It then determines the ideal wastegate duty cycle (**WGNEED**) that would have been required to hit the target perfectly. It fits a 3D surface to this "needed" data and compares it to your old table. If the new value is statistically significant, it recommends an update.
* **Interpreting the Results**: The 3D plot shows your original map (gray wireframe), the recommended map (colored surface), and your log data (red dots). You want the colored surface to pass through the middle of the red dots. The final tables are colored green for an increase in duty cycle and red for a decrease.
* **Limitations & Best Practices**: This tuner works best with steady-state data, however any data will work, even part throttle. It filters out data where boost pressure is fluctuating wildly. The temperature compensation feature requires logs from a wide range of ambient temperatures to be accurate.

### MAF & MFF (Fueling) Tuners

* **Synopsis**: These module corrects your fueling tables based on your fuel trims. MAF uses the GLOBAL_STD_MAF correction tables to increase or decrease modelled airflow, while MFF uses the mass fuel flow correction tables.
* **How it Works**: They look at your Short-Term (STFT) and Long-Term (LTFT) fuel trims. If the ECU is consistently adding or removing fuel (e.g., LTFT is +5%), it assumes the engine isn't getting the amount of air it thinks it is. The tuners calculate the necessary correction to your MAF or MFF tables to bring fuel trims closer to zero.
* **Interpreting the Results**: The 3D plots and result tables show the recommended corrections.
* **Limitations & Best Practices**: If you have other issues like incorrect injector scaling or a fuel pressure problem, these can be misinterpreted as an airflow issue. It's best to solve mechanical or base configuration issues before relying heavily on these tuners.

### KNK (Ignition) Tuner

* **Synopsis**: This module analyzes knock sensor data to recommend safer and more optimal ignition timing. It will recommend retarding timing where knock is detected and can carefully advance timing in knock-free regions.
* **How it Works**: It detects a "knock event" whenever the ECU's internal knock counters decrease (which happens after the ECU pulls timing). It then applies a statistically-driven retard in those cells. In cells with many data points but no knock, it will add a small amount of timing, weighted by its confidence and capped by your **Max Advance** setting.
* **Interpreting the Results**: The initial plot shows where knock events occurred and on which cylinder. The final table shows the recommended ignition map, with red cells indicating retard and green cells indicating advance.
* **Limitations & Best Practices**: This is a powerful but sensitive tool. It cannot distinguish between true engine knock and false knock from other engine bay noises. It works best with logs that contain both knock-free operation and some knock events. If your logs have no knock, it will only ever advance timing, which may not be safe. The module also has no concept of max brake torque and will happily advance past it if knock is not seen. Always review the recommendations carefully, or even set the **Max Advance** to zero.

### LPFP (Low-Pressure Fuel Pump) Tuner

* **Synopsis**: This module optimizes the duty cycle of your low-pressure fuel pump. The goal is to ensure the pump hits target fuel pressue without large fluctuations across a wide variety of conditions.
* **How it Works**: It looks at the relationship between the requested fuel flow, the target fuel pressure, and the PWM duty cycle required to achieve it. It filters for stable operating points and fits a surface to determine the most efficient duty cycle for any given condition.
* **Interpreting the Results**: The output is a recommended PWM table. The 3D plot shows how the recommended surface fits the actual logged data points.
* **Limitations & Best Practices**: This tuner relies on the ECU's calculated "Fuel Flow Setpoint" (FF_SP), which is an estimate. The results are a strong recommendation for optimizing the system but should be reviewed with an understanding of your specific fuel system's capabilities.

## 5. Saving and Applying Your New Tune

If you checked "Save Results to CSV?", you will find the output files in the same folder as your logs. You can open these in any spreadsheet program.

To apply the changes, you must manually enter the recommended values from the on-screen tables or the saved CSV files into your tuning software (e.g., TunerPro, etc.) and flash the new `.bin` file to your ECU.

Happy Tuning!
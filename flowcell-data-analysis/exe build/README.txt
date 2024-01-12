Osmotex 13mm Flow Cell Data Analysis and Visualization
Version 0.1.2
Author: Holly Huang <holly.huang@myantx.com>

- Flow cell data analysis python script bundled in exe file
- Build with PyInstaller v2.41.0 / Windows only

1) Copy raw data and comment file into a raw data folder
2) Select output folder to store excel summary sheets and plots
3) You can choose to analyze a single experiment OR analyze all files in a given folder. Those two operations are independant of each other

For analyzing a single experiment: 
1) Select relevant data files and double check the display name is the correct one
2) Click "generate data summary" to output an excel sheet containing information on experiment condition and figures of merit
3) Click "flow and current plots" to generate plots of:
-cycle average
-snapshot
-flow vs water column height
4) The console will prompt user to change signal track and height delta for plotting if desired (default signal=1 and deltah=0)
5) Click "exit" to quit the program or continue selecting different file for processing

For comparison between multiple experiments: 
1) Select folder containing raw data and comment files
2) Click "generate folder summary" to output an excel sheet containing information on:
- experiment name
- figures of merit
- pulse and cycle averages
- mean flow, current, and power
3) Click "genrate boxplot" to generate boxplot comparing all experiments in the folder
4) The console will prompt user to change height delta for plotting if desired (default deltah=0)
5) The console will then prompt user to input shortened sample name for plotting
6) Click "exit" to quit the program or continue selecting different folder for processing
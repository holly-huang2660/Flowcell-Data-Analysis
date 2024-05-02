from ui import UserInterface
from flowcalc import FlowCalculator
from dataplot import DataPlot
import openpyxl

# GUI for processing data
# User is able to select a single file to process or select a folder to summarize all results inside
# single file would output a summary Excel file and 8 figures (4 for each cell)
# flow snapshot, current snapshot, flow avg, and current avg
# folder can output a summary Excel file with calculated flow parameters
# folder can also output comparison boxplot
# click exit when you are done

ui = UserInterface()

# Note: currently manually setting y-axis for flow avg plot, change value in dataplot.py
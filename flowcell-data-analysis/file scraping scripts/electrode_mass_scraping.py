import os
import pandas as pd
import numpy as np

# folder_loc = r'\\xrcc.com\dfs\Projects\Myant X Research Portfolio\Osmotex\2. Experimental\13mmFlowCellTest_Data'
summary_loc = r'C:\Users\hhuang\Downloads\flow cell mass and resistance summary'
#
# # copy all excel files into new folder for processing
# for (roots, dirs, files) in os.walk(folder_loc, topdown=True):
#     for file in files:
#         if '.xlsx' in file:
#             xlsx_file = f'{roots}\\{file}'
#             shutil.copy(xlsx_file, summary_loc)

files = os.listdir(summary_loc)
file_list = [f'{summary_loc}\\{n}' for n in os.listdir(summary_loc) if 'Resistance' or 'Weights' in n]
df_list = []
for file in file_list:
    exp_name = file.split('\\')[-1]
    # initial parse the entire sheet, if the sheet is not named 'mass' then check if it's using the default 'Sheet1'
    try:
        df = pd.read_excel(file, sheet_name='Mass', header=None)
    except ValueError:
        df = pd.read_excel(file, sheet_name='Sheet1', header=None)

    # identify which row and column the flow cell mass is recorded, assuming it's a WYPO system
    try:
        index = np.where(df == 'W')
        row_start = index[0][0]
        column_start = index[1][0]
    except IndexError:
        index = np.where(df == 'R')
        row_start = index[0][0]
        column_start = index[1][0]

    # second go at actually grabbing the data
    try:
        df1 = pd.read_excel(file,
                           sheet_name='Mass',
                           header=None,
                           usecols=range(column_start, column_start+5),
                           names=['W1','Y1','P1','O1','IEM Coated Electrode'],
                           skiprows=lambda x: x not in [row_start+1])
        df2 = pd.read_excel(file,
                           sheet_name='Mass',
                           header=None,
                           usecols=range(column_start, column_start+5),
                           names=['W2','Y2','P2','O2','Carbon Electrode'],
                           skiprows=lambda x: x not in [row_start+2])
    except ValueError:
        df1 = pd.read_excel(file,
                           sheet_name='Sheet1',
                           header=None,
                           usecols=range(column_start, column_start+5),
                           names=['W1','Y1','P1','O1','IEM Coated Electrode'],
                           skiprows=lambda x: x not in [row_start+1])
        df2 = pd.read_excel(file,
                           sheet_name='Sheet1',
                           header=None,
                           usecols=range(column_start, column_start+5),
                           names=['W2','Y2','P2','O2','Carbon Electrode'],
                           skiprows=lambda x: x not in [row_start+2])

    df = pd.concat([df1, df2], axis=1)
    df.insert(0, 'Date', exp_name)
    df_list.append(df)
output_df = pd.concat(df_list)
output_loc = r'C:\Users\hhuang\Downloads\flow cell mass output'
output_df.to_csv(f'{output_loc}/output_df.csv')
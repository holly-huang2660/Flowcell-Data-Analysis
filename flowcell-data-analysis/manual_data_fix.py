import pandas as pd
import numpy as np

folder_path = r'C:\Users\hhuang\PycharmProjects\Osmotex\flowcell-data-analysis\data comparison\20250121 weekly update\archive'
files = ['250116_K_CC-4mil-SefarPX64Ag353mm-MP40_EntekPE-EO-1-23_CC-8mil-SefarPX64Ag353mm-MP40_3mM_PO',
         '250116_K_CC-4mil-SefarPX64Ag353mm-MP40_EntekPE-EO-1-23_CC-8mil-SefarPX64Ag353mm-MP40_3mM_WY']

header = ["time",  # time series
          "appv",  # set voltage (V)
          "sli1",  # flow rate (uL/min)
          "vtot1",  # measured voltage for cell 1
          "vcur1",  # current for cell 1
          "sli2",  # flow rate (uL/min)
          "vtot2",  # measured voltage for cell 2
          "vcur2",  # current for cell 2
          "deltah",  # water column height
          "signal"  # signal for different voltage waveforms
          ]

file_path = f'{folder_path}/{files[1]}'

raw_data = pd.read_csv(filepath_or_buffer=file_path,
                       engine="python",
                       sep="  | ",
                       names=header,
                       comment="#",
                       on_bad_lines='skip'
                       )

mask = ((raw_data['signal'] == 0) & (raw_data['appv'] != 0))
raw_data.loc[mask, 'signal'] = 1

#raw_data.to_csv(f'{folder_path}/temp2', sep=' ', index=False)

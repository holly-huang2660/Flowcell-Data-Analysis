import os
import shutil

import pandas as pd
import numpy as np

folder_loc = r'C:\Users\hhuang\Myant Research Center of Canada Inc\MRCC-MyantCH - Documents\10-19 Administration\12 Meetings'
summary_loc = r'C:\Users\hhuang\Downloads\meeting ppt files'

# copy all ppt files into new folder for processing
for (roots, dirs, files) in os.walk(folder_loc, topdown=True):
    for file in files:
        if 'HH' in file:
            xlsx_file = f'{roots}\\{file}'
            shutil.copy(xlsx_file, summary_loc)


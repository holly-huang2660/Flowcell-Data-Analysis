import pandas as pd

# ====================== merge new FoM data ====================================
file = r'C:\Users\hhuang\PycharmProjects\Osmotex\flowcell-data-analysis\raw data\data_summary_to_merge.xlsx'
df = pd.read_excel(file, sheet_name='to merge')
data_2023 = pd.read_excel(file, sheet_name='2023 data')
data_2024 = pd.read_excel(file, sheet_name='2024 data')

merge_cols = ['Experiment File', 'flow cell', 'signal track']
data_cols = ['appv',
             'pressure flow (L/h/m^2)',
             'neg pulse flow (L/h/m^2)',
             'pos pulse flow (L/h/m^2)',
             'hydraulic permeability',
             'pulse pressure (m)',
             'cycle pressure (m)',
             'neg pulse current (A/m^2)',
             'pos pulse current (A/m^2)',
             'pulse energy consumption (Wh/L)',
             'cycle energy consumption (Wh/L)',
             'work in (W/m^2)',
             'work out (W/m^2)',
             'cycle efficiency']
new_data = pd.concat([data_2023, data_2024])
new_data = new_data[merge_cols + data_cols]
df.drop(columns=data_cols, inplace=True)

merged_df = df.merge(new_data, how='left', on=merge_cols)
merged_df.to_csv(r'C:\Users\hhuang\PycharmProjects\Osmotex\flowcell-data-analysis\raw data\output\summary\temp.csv',
                 na_rep='n/a', encoding='utf-8-sig')

from textwrap import wrap
import pandas as pd
from flowcalc import FlowCalculator
import matplotlib.pyplot as plt
import numpy as np
from tkinter import simpledialog, messagebox
from pathlib import Path
from scipy import stats
import seaborn as sns

FONT_SIZE = 10
COLOR = "crest"
FIG_SIZE = (6, 4)


class DataPlot:
    def __init__(self, file_path):
        # Processing the first file
        self.df, self.pulse_df, self.file_name = self.data_preprocessing(exp=file_path[0])

        pulse_df_list = [self.pulse_df]

        # Processing for multiple files
        if len(file_path) > 1:
            for exp in file_path[1:]:
                df, pulse_df, file_name = self.data_preprocessing(exp=exp)

                # Get data for plotting pulse and cycle flow vs water column [m]
                pulse_df_list.append(pulse_df)
                self.pulse_df = pd.concat(pulse_df_list, ignore_index=True)

                # Get raw data for avg and snapshot plot
                merged = pd.merge(self.df, df, how='left', on=['signal', 'deltah', 'appv', 'cnt', 'cyc'])
                self.df = merged

        # Text for plot titles, auto-wrap ones that are too long
        self.title_text = self.file_name[0]
        self.title_text = self.title_text[:-3]

        # invert values for plotting
        # self.df["appv"] = self.df["appv"] * -1
        # self.df["flow1"] = self.df["flow1"] * -1
        # self.df["flow2"] = self.df["flow2"] * -1
        # self.df["cur1"] = self.df["cur1"] * -1
        # self.df["cur2"] = self.df["cur2"] * -1

        # Initiate default values for plotting and ask if user wants to change
        self.signal = 1
        self.deltah = 0
        self.prompt_user_input()

        # Get column names
        self.col_names = [name for name in self.df.columns if
                          ('flow' in name) or (('cur' in name) and ('vcur' not in name))]

    def data_preprocessing(self, exp):
        # Processing the first file
        flow_calculator = FlowCalculator(file_path=exp)
        df = flow_calculator.raw_data
        pulse_df = flow_calculator.pulse_cycle_calculator()
        file_name = [exp.split("/")[-1]]

        cell_pair = exp.split('/')[-1].split('_')[-1]
        old_cols = ['flow1', 'flow2', 'cur1', 'cur2']
        new_cols = {name: f'{name}_{cell_pair}' for name in old_cols}
        df.drop(columns=['time', 'rel time', 'sli1', 'sli2', 'vcur1', 'vcur2', 'vtot1', 'vtot2', 'pwr1', 'pwr2'],
                inplace=True)
        df.rename(columns=new_cols, inplace=True)
        df.drop(df[df['signal'] == 0].index, inplace=True)
        return df, pulse_df, file_name

    def prompt_user_input(self):
        signal_list = self.df["signal"].unique()

        user_input = messagebox.askyesno("Plotting",
                                         "Change the signal track (default=1) and height delta (default=0) for "
                                         "plotting?")
        if user_input:
            self.signal = simpledialog.askinteger("Signal Track", "Enter signal track: ")
            if self.signal not in signal_list:
                messagebox.showinfo("Info", f"signal = {self.signal} doesn't exist, try again.")
                self.signal = 1  # reset to default
                self.prompt_user_input()
            else:
                deltah_list = self.df.loc[(self.df["signal"] == self.signal)]["deltah"].unique()
                self.deltah = simpledialog.askfloat("Height", "Enter height delta: ")
                if self.deltah not in deltah_list:
                    messagebox.showinfo("Info",
                                        f"deltah = {self.deltah} doesn't exist for signal = {self.signal}, try again.")
                    self.deltah = 0  # reset to default
                    self.prompt_user_input()

    def flowcell_averages(self):
        # add average and standard deviation columns across all 4 flow cells to the dataframe
        flow_col_names = [name for name in self.df.columns if 'flow' in name]
        cur_col_names = [name for name in self.df.columns if 'cur' in name]

        self.df['flow_avg'] = self.df[flow_col_names].mean(axis=1)
        self.df['flow_std'] = self.df[flow_col_names].std(axis=1)

        self.df['cur_avg'] = self.df[cur_col_names].mean(axis=1)
        self.df['cur_std'] = self.df[cur_col_names].std(axis=1)

        # calculate 95% confidence interval with t-test
        # write margin of error to new columns
        alpha = 0.05
        n_samples = self.df.shape[0]
        t_value = stats.t.ppf(1 - alpha / 2, n_samples - 1)
        self.df['flow_ci'] = self.df['flow_std'] / np.sqrt(n_samples) * t_value
        self.df['cur_ci'] = self.df['cur_std'] / np.sqrt(n_samples) * t_value

    def cycle_avg_plot(self, figure_folder="figures"):
        # Add calculated column
        self.flowcell_averages()
        df_filter = self.df.loc[((self.df["signal"] == self.signal)
                                 & (self.df["deltah"] == self.deltah)
                                 & (self.df["cyc"] > 1)
                                 )]
        df_filter = df_filter.copy()

        flow1_df = df_filter.groupby(["signal", "deltah", "appv", "cnt"], as_index=False)["flow_avg"].mean()
        flow2_df = df_filter.groupby(["signal", "deltah", "appv", "cnt"], as_index=False)["flow_std"].mean()
        cur1_df = df_filter.groupby(["signal", "deltah", "appv", "cnt"], as_index=False)["cur_avg"].mean()
        cur2_df = df_filter.groupby(["signal", "deltah", "appv", "cnt"], as_index=False)["cur_std"].mean()

        df = flow1_df.merge(flow2_df).merge(cur1_df).merge(cur2_df)

        Path(f"{figure_folder}/flow").mkdir(parents=True, exist_ok=True)
        Path(f"{figure_folder}/current").mkdir(parents=True, exist_ok=True)
        appv_abs = df_filter['appv'].abs().unique()[0]

        # Loop through all four variables of interest
        y_var = {'flow_avg': 'flow_std', 'cur_avg': 'cur_std'}
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE, layout="tight")
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE, layout="tight")

        pos_pulse = df.loc[(df["appv"] == appv_abs)]
        neg_pulse = df.loc[(df["appv"] == -appv_abs)]

        for y, std in y_var.items():
            if "flow" in y:
                ax1.plot(y, '.-', data=pos_pulse, label=f"{appv_abs} V", markersize=2, linewidth=1, color='#1f77b4')
                ax1.plot(y, '.-', data=neg_pulse, label=f"-{appv_abs} V", markersize=2, linewidth=1, color='#ff7f0e')

                ax1.errorbar(pos_pulse.index, pos_pulse[y], fmt='.', yerr=pos_pulse[std], capsize=2, markersize=2,
                             linewidth=1, color='#1f77b4', alpha=0.3)
                ax1.errorbar(neg_pulse.index, neg_pulse[y], fmt='.', yerr=neg_pulse[std], capsize=2, markersize=2,
                             linewidth=1, color='#ff7f0e', alpha=0.3)
                plt.ylabel(r"Avg Flow [L/h/$m^{2}$]", fontsize=FONT_SIZE)
            elif "cur" in y:
                ax2.plot(y, '.-', data=pos_pulse, label=f"{appv_abs} V", markersize=2, linewidth=1, color='#1f77b4')
                ax2.plot(y, '.-', data=neg_pulse, label=f"-{appv_abs} V", markersize=2, linewidth=1, color='#ff7f0e')

                ax2.errorbar(pos_pulse.index, pos_pulse[y], fmt='.', yerr=pos_pulse[std], capsize=2, markersize=2,
                             linewidth=1, color='#1f77b4', alpha=0.3)
                ax2.errorbar(neg_pulse.index, neg_pulse[y], fmt='.', yerr=neg_pulse[std], capsize=2, markersize=2,
                             linewidth=1, color='#ff7f0e', alpha=0.3)
                plt.ylabel(r"Avg Current [A/$m^{2}$]", fontsize=FONT_SIZE)

        for ax in (ax1, ax2):
            ax.legend(loc="upper right", fontsize=FONT_SIZE, labelspacing=0.2)
            ax.set_title('\n'.join(wrap(self.title_text, 50)), loc='left', fontsize=FONT_SIZE)
            ax.set_xlabel("Time [s]", fontsize=FONT_SIZE)

            # Auto-scale y-axis
            # Find max value to be used as y-axis limits (+10% margin), force symmetrical axis
            ymax = max([abs(n) for n in ax.get_ylim()])
            ax.set_ylim(ymax * -1, ymax)
            ax.grid(visible=True)

        fig1.savefig(f"{figure_folder}/flow/{self.title_text}_avg_{y}_sig={self.signal}_deltah={self.deltah}.png",
                     transparent=True)
        fig2.savefig(f"{figure_folder}/current/{self.title_text}_avg_{y}_sig={self.signal}_deltah={self.deltah}.png",
                     transparent=True)

        # plt.show()
        plt.close()

        # write to temp file for troubleshooting
        Path(f"{figure_folder}/temp").mkdir(parents=True, exist_ok=True)
        df.to_excel(f"{figure_folder}/temp/cycle_average_temp.xlsx")

    def snapshot_plot(self, figure_folder="figures"):
        # Add calculated column
        self.flowcell_averages()

        # filter
        df = self.df.loc[((self.df["signal"] == self.signal)
                          & (self.df["deltah"] == self.deltah)
                          )]
        df = df.copy()
        # construct relative time (min) column
        # df['rel time'] = df.reset_index()['time'].diff().dt.total_seconds().fillna(0).cumsum().values
        # df["rel time"] = df["rel time"] / 60

        # Check for magnitude of applied voltage
        appv_abs = [n for n in df['appv'].abs().unique() if n != 0]
        appv_abs = appv_abs[0]


        # group data by voltage, assign NaN values to avoid connecting line between cycles
        zero_v = df.copy()
        zero_v[zero_v["appv"] != 0] = np.nan
        pos_v = df.copy()
        pos_v[pos_v["appv"] != appv_abs] = np.nan
        neg_v = df.copy()
        neg_v[neg_v["appv"] != -appv_abs] = np.nan

        # Variables of interest
        y_var = {'flow_avg': 'flow_std', 'cur_avg': 'cur_std'}

        Path(f"{figure_folder}/flow").mkdir(parents=True, exist_ok=True)
        Path(f"{figure_folder}/current").mkdir(parents=True, exist_ok=True)

        for y, std in y_var.items():
            # configurate plot size
            fig, ax = plt.subplots(figsize=FIG_SIZE, layout="tight")

            # plot the average flow and current
            ax.plot(pos_v.index, pos_v[y], '.-', label=f"+{appv_abs} V", markersize=2, linewidth=1, color='#1f77b4')
            ax.plot(zero_v.index, zero_v[y], 'k--', label="0 V", markersize=2, linewidth=1)
            ax.plot(neg_v.index, neg_v[y], '.-', label=f"-{appv_abs} V", markersize=2, linewidth=1, color='#ff7f0e')

            # Auto-scale y-axis
            # Find max value to be used as y-axis limits (+10% margin), force symmetrical axis
            ymax = max([abs(n) for n in ax.get_ylim()])
            ax.set_ylim(ymax * -1, ymax)

            # plot the standard deviation
            ax.errorbar(pos_v.index, pos_v[y], fmt='.', yerr=pos_v[std], capsize=2, markersize=2,
                        linewidth=1, color='#1f77b4', alpha=0.1)
            ax.errorbar(neg_v.index, neg_v[y], fmt='.', yerr=neg_v[std], capsize=2, markersize=2,
                        linewidth=1, color='#ff7f0e', alpha=0.1)

            ax.legend(loc="upper right", fontsize=FONT_SIZE, labelspacing=0.2)
            ax.set_title('\n'.join(wrap(self.title_text, 50)), loc='left', fontsize=FONT_SIZE)

            if "flow" in y:
                plt.ylabel(r"Flow [L/h/$m^{2}$]", fontsize=FONT_SIZE)
            elif "cur" in y:
                plt.ylabel(r"Current [A/$m^{2}$]", fontsize=FONT_SIZE)

            ax.set_xlabel("Time [s]", fontsize=FONT_SIZE)

            if "flow" in y:
                fig.savefig(
                    f"{figure_folder}/flow/{self.title_text}_snapshot_{y}_sig={self.signal}_deltah={self.deltah}.png",
                    transparent=True)
            elif "cur" in y:
                fig.savefig(
                    f"{figure_folder}/current/{self.title_text}_snapshot_{y}_sig={self.signal}_deltah={self.deltah}.png",
                    transparent=True)
            # plt.show()
            plt.close()

        # write to temp file for troubleshooting
        Path(f"{figure_folder}/temp").mkdir(parents=True, exist_ok=True)
        df.to_excel(f"{figure_folder}/temp/all_cycle_snapshot_temp.xlsx")

    def cycle_avg_by_flowcell(self, figure_folder="figures"):
        df_filter = self.df.loc[((self.df["signal"] == self.signal)
                                 & (self.df["deltah"] == self.deltah)
                                 & (self.df["cyc"] > 1)
                                 )]
        df_filter = df_filter.copy()

        # col_names = [name for name in self.df.columns if ('flow' in name) or (('cur' in name) and ('vcur' not in name))]
        col_names = self.col_names

        avg_list = []
        std_list = []
        for col in col_names:
            avg_df = df_filter.groupby(["signal", "deltah", "appv", "cnt"], as_index=False)[col].mean()
            avg_df.rename(columns={n: n + '_avg' for n in col_names}, inplace=True)
            std_df = df_filter.groupby(["signal", "deltah", "appv", "cnt"], as_index=False)[col].std()
            std_df.rename(columns={n: n + '_std' for n in col_names}, inplace=True)
            avg_list.append(avg_df)
            std_list.append(std_df)
        avg = avg_list[0]
        std = std_list[0]
        for n in range(1, len(avg_list)):
            avg = avg.merge(avg_list[n])
            std = std.merge(std_list[n])

        df = avg.merge(std)

        fig1, ax1 = plt.subplots(figsize=FIG_SIZE, layout="tight")
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE, layout="tight")
        for y in col_names:
            if "1" in y:
                label_text = y[-2]
            else:
                label_text = y[-1]

            if "flow" in y:
                markers, caps, bars = ax1.errorbar(df.index, df[f'{y}_avg'], fmt='.-', yerr=df[f'{y}_std'],
                                                   label=label_text,
                                                   capsize=2, markersize=5, linewidth=1)
                [bar.set_alpha(0.5) for bar in bars]
                [cap.set_alpha(0.5) for cap in caps]
                ax1.set_ylabel(r"Avg Flow [L/h/$m^{2}$]", fontsize=FONT_SIZE)

            elif "cur" in y:
                markers, caps, bars = ax2.errorbar(df.index, df[f'{y}_avg'], fmt='.-', yerr=df[f'{y}_std'],
                                                   label=label_text,
                                                   capsize=2, markersize=5, linewidth=1, alpha=1)
                [bar.set_alpha(0.5) for bar in bars]
                [cap.set_alpha(0.5) for cap in caps]
                ax2.set_ylabel(r"Avg Current [A/$m^{2}$]", fontsize=FONT_SIZE)

        # Plot formatting
        for ax in (ax1, ax2):
            ax.legend(loc="upper right", fontsize=FONT_SIZE, labelspacing=0.2)
            ax.set_xlabel("Time [s]", fontsize=FONT_SIZE)
            ax.set_title('\n'.join(wrap(self.title_text, 50)), loc='left', fontsize=FONT_SIZE)
            ax.grid(visible=True)

            # Auto-scale y-axis
            ymax = max([abs(n) for n in ax.get_ylim()])
            ax.set_ylim(ymax * -1, ymax)
            ax.set_xlim(0, 120)

        Path(f"{figure_folder}/cycle average").mkdir(parents=True, exist_ok=True)

        fig1.savefig(f"{figure_folder}/cycle average/flowcell_flow_avg_{self.title_text}.png",
                     transparent=True)
        fig2.savefig(f"{figure_folder}/cycle average/flowcell_cur_avg_{self.title_text}.png",
                     transparent=True)

        # plt.show()
        plt.close()

        # write to temp file for troubleshooting
        Path(f"{figure_folder}/temp").mkdir(parents=True, exist_ok=True)
        df.to_excel(f"{figure_folder}/temp/cycle_average_by_flowcell_temp.xlsx")

    def snapshot_by_flowcell(self, figure_folder="figures"):
        df_filter = self.df.loc[((self.df["signal"] == self.signal)
                                 & (self.df["deltah"] == self.deltah)
                                 )]
        df = df_filter.copy()

        # construct relative time (min) column
        # df['rel time'] = df.reset_index()['time'].diff().dt.total_seconds().fillna(0).cumsum().values
        # df["rel time"] = df["rel time"] / 60
        # df.drop(df[df["rel time"] > 25].index, inplace=True)

        # # Check for magnitude of applied voltage
        # appv_abs = [n for n in df['appv'].abs().unique() if n != 0]
        # appv_abs = appv_abs[0]

        # # group data by voltage, assign NaN values to avoid connecting line between cycles
        # zero_v = df.copy()
        # zero_v[zero_v["appv"] != 0] = np.nan
        # pos_v = df.copy()
        # pos_v[pos_v["appv"] != appv_abs] = np.nan
        # neg_v = df.copy()
        # neg_v[neg_v["appv"] != -appv_abs] = np.nan

        # Variables of interest
        col_names = [name for name in df.columns if ('flow' in name) or (('cur' in name) and ('vcur' not in name))]

        fig1, ax1 = plt.subplots(figsize=FIG_SIZE, layout="tight")
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE, layout="tight")
        for y in col_names:
            if "1" in y:
                label_text = y[-2]
            else:
                label_text = y[-1]

            if "flow" in y:
                ax1.plot(y, '.-', data=df, markersize=5, linewidth=1, label=label_text, alpha=0.5)
                # ax1.plot('rel time', y, 'b.', data=zero_v, markersize=2, linewidth=1)
                # ax1.plot('rel time', y, '.-', data=pos_v, markersize=2, linewidth=1)
                # ax1.plot('rel time', y, '.-', data=neg_v, markersize=2, linewidth=1, label=label_text)

                ax1.set_ylabel(r"Flow [L/h/$m^{2}$]", fontsize=FONT_SIZE)

            elif "cur" in y:
                ax2.plot(y, '.-', data=df, markersize=5, linewidth=1, label=label_text, alpha=0.5)
                # ax2.plot('rel time', y, 'b.', data=zero_v, markersize=2, linewidth=1)
                # ax2.plot('rel time', y, '.-', data=pos_v, markersize=2, linewidth=1)
                # ax2.plot('rel time', y, '.-', data=neg_v, markersize=2, linewidth=1, label=label_text)

                ax2.set_ylabel(r"Current [A/$m^{2}$]", fontsize=FONT_SIZE)

        # Plot formatting
        for ax in (ax1, ax2):
            ax.legend(loc="upper right", fontsize=FONT_SIZE, labelspacing=0.2)
            ax.set_xlabel("Time", fontsize=FONT_SIZE)
            ax.set_title('\n'.join(wrap(self.title_text, 50)), loc='left', fontsize=FONT_SIZE)
            ax.grid(visible=True)

            # Auto-scale y-axis
            ymax = max([abs(n) for n in ax.get_ylim()])
            ax.set_ylim(ymax * -1, ymax)

        fig1.savefig(f"{figure_folder}/flowcell_flow_snapshot_{self.title_text}.png", transparent=True)
        fig2.savefig(f"{figure_folder}/flowcell_cur_snapshot_{self.title_text}.png", transparent=True)

        # plt.show()
        plt.close()

    def flow_vs_water_column(self, figure_folder='figures'):
        df = self.pulse_df.loc[(self.pulse_df['signal'] == self.signal)]
        df = df.copy()
        df['deltah'] = df['deltah'] * 100  # convert height to cm
        sns.set_style("whitegrid")

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=FIG_SIZE, layout='tight', sharex='all', sharey='all')

        # format marker differently for system 1 vs 2
        for cell in list(df['flow cell'].unique()):
            cell_df = df.loc[(df['flow cell'] == cell)]
            if cell in 'POWY':
                sns.regplot(data=cell_df, x='deltah', y='cycle flow',
                            line_kws={'linestyle': '--'}, marker='o', x_ci=None, ci=None, label=cell, ax=ax1)
        for cell in list(df['flow cell'].unique()):
            cell_df = df.loc[(df['flow cell'] == cell)]
            if cell in 'GBRK':
                sns.regplot(data=cell_df, x='deltah', y='cycle flow',
                            line_kws={'linestyle': '--'}, marker='^', x_ci=None, ci=None, label=cell, ax=ax1)

        ax1.axhline(y=0, color='black', linestyle='--')
        ax1.set_ylabel('Cycle Flow [L/h/m^2]')
        ax1.set_xlabel('Water Column [cm]')
        ax1.set_title('\n'.join(wrap(self.title_text, 60)), loc='left', fontsize=FONT_SIZE)
        ax1.legend()

        for cell in list(df['flow cell'].unique()):
            cell_df = df.loc[(df['flow cell'] == cell)]
            if cell in 'POWY':
                sns.regplot(data=cell_df, x='deltah', y='pulse flow (-v)',
                            line_kws={'linestyle': '--'}, marker='o', x_ci=None, ci=None, label=cell, ax=ax2)
        for cell in list(df['flow cell'].unique()):
            cell_df = df.loc[(df['flow cell'] == cell)]
            if cell in 'GBRK':
                sns.regplot(data=cell_df, x='deltah', y='pulse flow (-v)',
                            line_kws={'linestyle': '--'}, marker='^', x_ci=None, ci=None, label=cell, ax=ax2)

        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.set_ylabel('Negative Pulse Flow [L/h/m^2]')
        ax2.set_xlabel('Water Column [cm]')
        ax2.legend()

        fig.savefig(f"{figure_folder}/flow_vs_deltah_{self.title_text}.png", transparent=True)

import os.path
import sys
from textwrap import wrap

import pandas as pd
import seaborn as sns
from dataplot import DataPlot
from flowcalc import FlowCalculator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from os import listdir
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilenames, askdirectory
from tkinter import simpledialog, messagebox
from pathlib import Path

FONT = ("calibri", 14, "normal")
BUTTON_FONT = ("calibri", 12, "normal")
FG = "#004b00"
BG = "#a7bbc6"


class UserInterface:
    def __init__(self):
        self.folder_file_list = None
        self.folder_path = None
        self.file_path = None
        self.output_folder = None

        # UI elements
        self.window = Tk()
        self.window.title("Osmotex 13mm Flow Cell Data Analysis & Visualization")
        self.window.config(padx=30, pady=30, bg=BG)

        self.label_output = Label(text="Select output folder: ", bg=BG, font=FONT)
        self.label_output_name = Label(text="<output-folder>", bg=BG, font=FONT)
        self.button_output = Button(text="Browse folder", font=BUTTON_FONT, command=self.get_output_folder)

        self.label_file = Label(text="Select multiple data files\nfor a single experiment:\n(PO & WY or GB & RK)",
                                bg=BG, font=FONT)
        self.label_file_name = Label(text="<file-name>", bg=BG, font=FONT)
        self.label_folder = Label(text="Select raw data folder: ", bg=BG, font=FONT)
        self.label_folder_name = Label(text="<data-folder-path>", bg=BG, font=FONT)

        self.button_file = Button(text="Browse file", font=BUTTON_FONT, command=self.get_file)
        self.button_file_summary = Button(text="Generate Data Summary", font=BUTTON_FONT, command=self.file_summary)
        self.file_sum_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)
        self.button_circuit_summary = Button(text="Equivalent Circuit R(RC)", font=BUTTON_FONT,
                                             command=self.circuit_summary)
        self.circuit_sum_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)
        self.button_file_plot = Button(text="Flow & Current Plots", font=BUTTON_FONT, command=self.file_plot)
        self.file_plot_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)
        self.button_rawdata = Button(text="Export Raw Data", font=BUTTON_FONT, command=self.export_rawdata)
        self.rawdata_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)

        self.button_folder = Button(text="Browse folder", font=BUTTON_FONT, command=self.get_folder)
        self.button_folder_summary = Button(text="Folder & Circuit Summary", font=BUTTON_FONT,
                                            command=self.folder_summary)
        self.folder_sum_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)
        self.button_folder_boxplot = Button(text="Generate Boxplot", font=BUTTON_FONT, command=self.boxplot_plot)
        self.boxplot_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)
        self.button_folder_plot = Button(text="Folder Flow & Current Plots", font=BUTTON_FONT, command=self.folder_plot)
        self.folderplot_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)

        self.exit = Button(text="Exit", font=BUTTON_FONT, command=self.close_window)

        # Place elements
        self.label_output.grid(row=0, column=0, padx=5, pady=5, columnspan=2)
        self.button_output.grid(row=0, column=2, padx=5, pady=5, columnspan=2)
        self.label_output_name.grid(row=1, column=0, columnspan=5)

        self.linebreak1 = Label(text="======================================", font=FONT, bg=BG)
        self.linebreak1.grid(row=2, column=0, columnspan=5)

        self.label_file.grid(row=3, column=0, padx=5, pady=5, columnspan=2, sticky='w')
        self.button_file.grid(row=3, column=2, padx=5, pady=5, columnspan=2)
        self.label_file_name.grid(row=4, column=0, columnspan=5)
        self.button_file_summary.grid(row=5, column=0, pady=15)
        self.file_sum_check.grid(row=5, column=1, sticky="w")
        self.button_file_plot.grid(row=5, column=2, pady=15)
        self.file_plot_check.grid(row=5, column=3, sticky="w")
        self.button_circuit_summary.grid(row=6, column=0, pady=15)
        self.circuit_sum_check.grid(row=6, column=1, sticky="w")
        self.button_rawdata.grid(row=6, column=2, pady=15)
        self.rawdata_check.grid(row=6, column=3, sticky="w")

        self.linebreak2 = Label(text="======================================", font=FONT, bg=BG)
        self.linebreak2.grid(row=7, column=0, columnspan=5)

        self.label_folder.grid(row=8, column=0, padx=5, pady=5, columnspan=2)
        self.button_folder.grid(row=8, column=2, padx=5, pady=5, columnspan=2)
        self.label_folder_name.grid(row=9, column=0, columnspan=5)
        self.button_folder_summary.grid(row=10, column=0, pady=15)
        self.folder_sum_check.grid(row=10, column=1, sticky="w")
        self.button_folder_boxplot.grid(row=10, column=2, pady=15)
        self.boxplot_check.grid(row=10, column=3, sticky="w")

        self.button_folder_plot.grid(row=11, column=0, pady=15)
        self.folderplot_check.grid(row=11, column=1, sticky="w")

        self.exit.grid(row=12, column=0, columnspan=5, padx=10, pady=10)

        self.window.mainloop()

    def close_window(self):
        sys.exit(0)

    def get_file(self):
        # prompt user to select file in folder
        # show an "Open" dialog box and return the path to the selected file

        self.file_sum_check.config(text="")
        self.file_plot_check.config(text="")

        file_path = askopenfilenames(title="Select data files")
        self.file_path = [f for f in file_path if ("." not in f and os.path.isfile(f))]

        file_name = []
        for file in self.file_path:
            file_name.append(file.split("/")[-1])
        display_text = '\n'.join(wrap('\n'.join(file_name), 60))

        self.label_file_name.config(text=display_text)

    def file_summary(self):
        # Make folder if it doesn't exist
        Path(f"{self.output_folder}/summary").mkdir(parents=True, exist_ok=True)
        # Get experiment name
        exp_name = self.file_path[0].split("/")[-1][:-3]
        self.export_data_summary(file_list=self.file_path, summary_name=f"data summary {exp_name}.xlsx")

        print(f"File summary generated. File is located in {self.output_folder}/summary")
        self.file_sum_check.config(text="✓")

    def export_rawdata(self):
        # Make folder if it doesn't exist
        Path(f"{self.output_folder}/raw data").mkdir(parents=True, exist_ok=True)

        # Get experiment date
        exp_date = self.file_path[0].split("/")[-1].split("_")[0]
        with pd.ExcelWriter(f"{self.output_folder}/raw data/{exp_date} raw data.xlsx") as writer:
            for file in self.file_path:
                flow_calculator = FlowCalculator(file_path=file)
                df = flow_calculator.raw_data
                df.to_excel(writer, sheet_name=f'{exp_date}_{flow_calculator.cell_1+flow_calculator.cell_2}', index=False)

        print(f"Raw data exported. File is located in {self.output_folder}/raw data")
        self.rawdata_check.config(text="✓")

    def circuit_summary(self):
        # Make folder if it doesn't exist
        Path(f"{self.output_folder}/summary").mkdir(parents=True, exist_ok=True)
        # Get experiment name
        exp_name = self.file_path[0].split("/")[-1][:-3]
        self.export_circuit_summary(file_list=self.file_path, summary_name=f"circuit summary {exp_name}.xlsx")

        print(f"Equivalent circuit fitted generated. File is located in {self.output_folder}/summary")
        self.circuit_sum_check.config(text="✓")

    def file_plot(self):
        # Make folder if it doesn't exist
        Path(f"{self.output_folder}/figures").mkdir(parents=True, exist_ok=True)

        signal, deltah, auto_ylim, flow_ymax, cur_ymax, auto_name = self.prompt_user_input()
        data_plot = DataPlot(file_path=self.file_path, manual_flowcell_name=auto_name)
        # change attribute to user input
        data_plot.signal = signal
        data_plot.deltah = deltah

        # Calculates and save figures to figure folder
        data_plot.snapshot_by_flowcell(figure_folder=f"{self.output_folder}/figures",
                                       auto_ylim=auto_ylim, flow_ylim=flow_ymax, cur_ylim=cur_ymax)
        data_plot.flow_vs_water_column(figure_folder=f"{self.output_folder}/figures")
        data_plot.cycle_avg_by_flowcell(figure_folder=f'{self.output_folder}/figures',
                                        auto_ylim=auto_ylim, flow_ylim=flow_ymax, cur_ylim=cur_ymax)
        print(f"Plotting finished, figures are located in {self.output_folder}/figures")
        self.file_plot_check.config(text="✓")

    def get_output_folder(self):
        self.output_folder = askdirectory(title="Select Folder")
        self.label_output_name.config(text='\n'.join(wrap(self.output_folder, 60)))

    def get_folder(self):
        self.boxplot_check.config(text="")
        self.folder_sum_check.config(text="")

        # GUI to prompt user to select folder
        self.folder_path = askdirectory(title="Select Folder")
        # Find all data files in the selected folder
        self.folder_file_list = [f"{self.folder_path}/{f}" for f in listdir(f"{self.folder_path}")
                                 if ("." not in f and os.path.isfile(f"{self.folder_path}/{f}"))]

        self.label_folder_name.config(text='\n'.join(wrap(self.folder_path, 60)))

    def folder_plot(self):
        # Make folder if it doesn't exist
        Path(f"{self.output_folder}/figures").mkdir(parents=True, exist_ok=True)

        # remove flow cell designation
        exp_names = [f[:-3] for f in self.folder_file_list]
        exp_names = list(set(exp_names))  # find unique

        signal, deltah, auto_ylim, flow_ymax, cur_ymax, auto_name = self.prompt_user_input()

        for exp in exp_names:
            # find unique sets of experiments
            file_path = [f for f in self.folder_file_list if exp in f]
            data_plot = DataPlot(file_path=file_path)

            # change attribute to user input
            data_plot.signal = signal
            data_plot.deltah = deltah

            data_plot.snapshot_by_flowcell(figure_folder=f'{self.output_folder}/figures',
                                           auto_ylim=auto_ylim, flow_ylim=flow_ymax, cur_ylim=cur_ymax)
            data_plot.flow_vs_water_column(figure_folder=f'{self.output_folder}/figures')
            data_plot.cycle_avg_by_flowcell(figure_folder=f'{self.output_folder}/figures',
                                            auto_ylim=auto_ylim, flow_ylim=flow_ymax, cur_ylim=cur_ymax)
            plt.close()

        print(f"Plotting finished, figures are located in {self.output_folder}/figures")
        self.folderplot_check.config(text="✓")

    def folder_summary(self):
        # Make folder if it doesn't exist
        Path(f"{self.output_folder}/summary").mkdir(parents=True, exist_ok=True)

        folder_name = self.folder_path.split("/")[-1]
        self.export_data_summary(file_list=self.folder_file_list, summary_name=f"{folder_name} folder summary.xlsx")
        self.export_circuit_summary(file_list=self.folder_file_list, summary_name=f"{folder_name} circuit summary.xlsx")

        print(f"Folder summary generated. File is located in {self.output_folder}/summary")
        self.folder_sum_check.config(text="✓")

    def export_data_summary(self, file_list, summary_name):
        material_list = {
            'file': [],
            'membrane': [],
            'electrode': [],
            'iem': []
        }
        flow_list = []
        eo_list = []
        fom_list = []

        for file in file_list:
            flow_calculator = FlowCalculator(file_path=file)
            mf = flow_calculator.mean_flow_calculator()
            eo = flow_calculator.pulse_cycle_calculator()
            fom = flow_calculator.FOM_calculator()

            # remove flow cell designation from file name
            exp_name = file.split('/')[-1][:-3]
            print(file.split('/')[-1])

            mf.insert(0, "file", exp_name)
            eo.insert(0, "file", exp_name)
            fom.insert(0, "file", exp_name)

            flow_list.append(mf)
            eo_list.append(eo)
            fom_list.append(fom)

            material_list['file'].append(exp_name)
            material_list['membrane'].append(flow_calculator.params['membrane'])
            material_list['electrode'].append(flow_calculator.params['electrode'])
            material_list['iem'].append(flow_calculator.params['iem'])

        flow_df = pd.concat(flow_list, ignore_index=True)
        eo_flow_df = pd.concat(eo_list, ignore_index=True)
        fom_df = pd.concat(fom_list, ignore_index=True)
        # circuit_df = pd.concat(circuit_list, ignore_index=True)
        material_df = pd.DataFrame(material_list)
        material_df.drop_duplicates('file', ignore_index=True, inplace=True)

        with pd.ExcelWriter(f"{self.output_folder}/summary/{summary_name}") as writer:
            material_df.to_excel(writer, sheet_name="experiment setup", index=False)
            fom_df.to_excel(writer, sheet_name="figures of merit", index=False)
            flow_df.to_excel(writer, sheet_name="flow and power summary", index=False)
            eo_flow_df.to_excel(writer, sheet_name="pulse and cycle calc", index=False)

    def export_circuit_summary(self, file_list, summary_name):
        material_list = {
            'file': [],
            'membrane': [],
            'electrode': [],
            'iem': []
        }

        circuit_list = []

        for file in file_list:
            flow_calculator = FlowCalculator(file_path=file)
            circuit = flow_calculator.circuit_fitting()

            # remove flow cell designation from file name
            exp_name = file.split('/')[-1][:-3]
            print(file.split('/')[-1])

            circuit.insert(0, "file", exp_name)
            circuit_list.append(circuit)

            material_list['file'].append(exp_name)
            material_list['membrane'].append(flow_calculator.params['membrane'])
            material_list['electrode'].append(flow_calculator.params['electrode'])
            material_list['iem'].append(flow_calculator.params['iem'])

        circuit_df = pd.concat(circuit_list, ignore_index=True)
        circuit_df.sort_values(by=['signal'], inplace=True)
        material_df = pd.DataFrame(material_list)
        material_df.drop_duplicates('file', ignore_index=True, inplace=True)

        with pd.ExcelWriter(f"{self.output_folder}/summary/{summary_name}") as writer:
            material_df.to_excel(writer, sheet_name="experiment setup", index=False)
            circuit_df.to_excel(writer, sheet_name="equivalent circuit", index=False)

    def boxplot_plot(self):
        """
        The box extends from the first quartile (Q1) to the third quartile (Q3) of the data, with a line at the median.
        The whiskers extend from the box to the farthest data point lying within 1.5x the inter-quartile range (IQR)
        from the box. Flier points are those past the end of the whiskers.
                 Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
                              |-----:-----|
              o      |--------|     :     |--------|    o  o
                              |-----:-----|
            flier             <----------->            fliers
                                   IQR
        :return:
        """
        # write graph to figures folder
        Path(f"{self.output_folder}/figures/boxplot").mkdir(parents=True, exist_ok=True)

        list1 = []

        keep_flowcell = messagebox.askyesno("Plotting",
                                            "Analyze flow cell pairs individually?\n"
                                            "Default is to combine all flow cell pairs as a single experiment")

        for file in self.folder_file_list:
            # Calculate figures of merit
            flow_calculator = FlowCalculator(file_path=file)
            df1 = flow_calculator.FOM_calculator()

            # remove flow cell designation from file name
            if keep_flowcell:
                exp_name = file.split("/")[-1]
            else:
                exp_name = file.split("/")[-1][:-3]
            df1.insert(0, "file", exp_name)
            list1.append(df1)

        df = pd.concat(list1, ignore_index=True)
        exp_list = df["file"].unique()

        # Shorten experiment name
        plot_label = {}
        user_input = messagebox.askyesno("Plotting",
                                         "Change sample names? Default is <sample YYMMDD>")
        if user_input:
            for name in exp_list:
                user_input = simpledialog.askstring("Plot sample name",
                                                    f"Please enter label for experiment <{name}>:\n")
                # shorten original file name for plotting, if no input detected use the default
                if len(user_input) > 0:
                    plot_label[name] = '\n'.join(wrap(user_input, 15))
                else:
                    exp_date = name.split('_')[0]
                    plot_label[name] = f"sample {exp_date}"
        else:
            for name in exp_list:
                exp_date = name.split('_')[0]
                plot_label[name] = f"sample {exp_date}"

        # Create new column with abbreviated name
        df["sample"] = df["file"].map(plot_label, na_action="ignore")

        folder_name = self.folder_path.split("/")[-1]

        # prompt user for graph title
        graph_title = simpledialog.askstring("Plot title", f"Please enter plot title:\n")

        # Plot for comparing flow
        self.fom_boxplot(df=df, fom_list=[col for col in df.columns if ('pulse flow' in col or 'cycle flow' in col)],
                         plot_name=f"{folder_name} flow boxplot", y_label=r"Flow [L/h/$m^{2}$]",
                         graph_title=graph_title)

        # Plot for comparing permeability
        self.fom_boxplot(df=df, fom_list=['hydraulic permeability'],
                         plot_name=f"{folder_name} perm boxplot", y_label="Hydraulic Permeability",
                         graph_title=graph_title)

        # Plot for comparing pressure (hydrostatic head, m)
        self.fom_boxplot(df=df, fom_list=[col for col in df.columns if ('pressure' in col and 'flow' not in col)],
                         plot_name=f"{folder_name} pressure boxplot", y_label=r"Water Column [m]",
                         graph_title=graph_title)

        # Plot for comparing power consumption
        self.fom_boxplot(df=df, fom_list=[col for col in df.columns if 'energy' in col],
                         plot_name=f"{folder_name} power boxplot", y_label="Power [Wh/L]",
                         graph_title=graph_title)

        # Plot for comparing current density
        self.fom_boxplot(df=df, fom_list=[col for col in df.columns if 'current' in col],
                         plot_name=f"{folder_name} current boxplot", y_label=r"Current [A/$m^{2}$]",
                         graph_title=graph_title)

        # Plot for comparing equivalent circuit resistance
        # self.fom_boxplot(df=df, fom_list=['Resistance (Ohms)'],
        #                  plot_name=f"{folder_name} eq circuit boxplot", y_label="Equivalent RC Circuit - Resistance",
        #                  graph_title=graph_title)

        self.boxplot_check.config(text="✓")
        print(f"Boxplot completed, output can be located at {self.output_folder}/figures/boxplot")

    def fom_boxplot(self, df, fom_list, plot_name, y_label, graph_title):
        sns.set_style("whitegrid")
        # Config figure size based on number of files in folder
        fig_width = len(self.folder_file_list)
        fig_height = 4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # Manually configurate box plot legend
        color_list = sns.color_palette()
        handles = []

        # handle missing data or 'n/a'
        for fom in fom_list:
            df.drop(df[df[fom] == 'n/a'].index, inplace=True)

        for n in range(len(fom_list)):
            sns.boxplot(df, x="sample", y=fom_list[n], ax=ax)

            # Manually configurate box plot legend
            handle = mpatches.Patch(facecolor=color_list[n], edgecolor='black', label=fom_list[n])
            handles.append(handle)

        # Put a legend to the right of the current axis
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(wrap=True)
        plt.ylabel(y_label)

        plt.title(graph_title, fontsize=12)

        fig.savefig(f"{self.output_folder}/figures/boxplot/{plot_name}.png",
                    bbox_inches='tight',
                    transparent=True)

        plt.close()

    def prompt_user_input(self):
        # Ask user if they want to change the signal track and delta height
        user_input = messagebox.askyesno("Plotting",
                                         "Change the signal track (default=1) and height delta (default=0) for "
                                         "plotting?")
        if user_input:
            signal = simpledialog.askfloat("Signal Track", "Enter signal track: ")
            deltah = simpledialog.askfloat("Height", "Enter height delta: ")
            if deltah == 0:  # if it's zero set to int instead of float
                deltah = 0
            if signal == 1:
                signal = 1
        else:
            signal = 1
            deltah = 0

        # Ask user if they want to manually set y-limits for plotting
        user_input = messagebox.askyesno("Plotting",
                                         "Manually set y limits for plotting? Click no to automatically set scale")
        if user_input:
            auto_ylim = False
            flow_ymax = simpledialog.askinteger("Y limit", "Y limit for flow (L/h/m^2):")
            cur_ymax = simpledialog.askinteger("Y limit", "Y limit for current (A/m^2):")
        else:
            auto_ylim = True
            flow_ymax = 10
            cur_ymax = 100

        # Ask user if they want to manually name the flow cells for plotting
        user_input = messagebox.askyesno("Plotting",
                                         "Manually change flow cell names? Click no to automatically name based on "
                                         "file name")
        if user_input:
            auto_name = True
        else:
            auto_name = False

        return signal, deltah, auto_ylim, flow_ymax, cur_ymax, auto_name

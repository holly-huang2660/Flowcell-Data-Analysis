import os.path
import sys
from textwrap import wrap

import pandas as pd
import seaborn as sns
from dataplot import DataPlot
from flowcalc import FlowCalculator
import matplotlib.pyplot as plt
from os import listdir
from tkinter import *
from tkinter.filedialog import askopenfilenames, askdirectory
from tkinter import simpledialog, messagebox
from pathlib import Path

FONT = ("calibri", 14, "normal")
BUTTON_FONT = ("calibri", 12, "normal")
FG = "#004b00"
BG = "#a7bbc6"


class UserInterface:
    def __init__(self):
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

        self.label_file = Label(text="Select multiple data files for one experiment: ", bg=BG, font=FONT)
        self.label_file_name = Label(text="<file-name>", bg=BG, font=FONT)
        self.label_folder = Label(text="Select raw data folder: ", bg=BG, font=FONT)
        self.label_folder_name = Label(text="<data-folder-path>", bg=BG, font=FONT)

        self.button_file = Button(text="Browse file", font=BUTTON_FONT, command=self.get_file)
        self.button_file_summary = Button(text="Generate Data Summary", font=BUTTON_FONT, command=self.file_summary)
        self.file_sum_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)
        self.button_file_plot = Button(text="Flow & Current Plots", font=BUTTON_FONT, command=self.file_plot)
        self.file_plot_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)

        self.button_folder = Button(text="Browse folder", font=BUTTON_FONT, command=self.get_folder)
        self.button_folder_summary = Button(text="Generate Folder Summary", font=BUTTON_FONT,
                                            command=self.folder_summary)
        self.folder_sum_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)
        self.button_folder_boxplot = Button(text="Generate Boxplot", font=BUTTON_FONT, command=self.boxplot_plot)
        self.boxplot_check = Label(text="", fg=FG, bg=BG, font=BUTTON_FONT)

        self.exit = Button(text="Exit", font=BUTTON_FONT, command=self.close_window)

        # Place elements
        self.label_output.grid(row=0, column=0, padx=5, pady=5, columnspan=2)
        self.button_output.grid(row=0, column=2, padx=5, pady=5, columnspan=2)
        self.label_output_name.grid(row=1, column=0, columnspan=4)

        self.linebreak1 = Label(text="======================================", font=FONT, bg=BG)
        self.linebreak1.grid(row=2, column=0, columnspan=4)

        self.label_file.grid(row=3, column=0, padx=5, pady=5, columnspan=2)
        self.button_file.grid(row=3, column=2, padx=5, pady=5, columnspan=2)
        self.label_file_name.grid(row=4, column=0, columnspan=4)
        self.button_file_summary.grid(row=5, column=0, pady=15)
        self.file_sum_check.grid(row=5, column=1, sticky="w")
        self.button_file_plot.grid(row=5, column=2, pady=15)
        self.file_plot_check.grid(row=5, column=3, sticky="w")

        self.linebreak2 = Label(text="======================================", font=FONT, bg=BG)
        self.linebreak2.grid(row=6, column=0, columnspan=4)

        self.label_folder.grid(row=7, column=0, padx=5, pady=5, columnspan=2)
        self.button_folder.grid(row=7, column=2, padx=5, pady=5, columnspan=2)
        self.label_folder_name.grid(row=8, column=0, columnspan=4)
        self.button_folder_summary.grid(row=9, column=0, pady=15)
        self.folder_sum_check.grid(row=9, column=1, sticky="w")
        self.button_folder_boxplot.grid(row=9, column=2, pady=15)
        self.boxplot_check.grid(row=9, column=3, sticky="w")

        self.exit.grid(row=10, column=0, columnspan=4, padx=10, pady=10)

        self.window.mainloop()

    def close_window(self):
        sys.exit(0)

    def get_file(self):
        # prompt user to select file in folder
        # show an "Open" dialog box and return the path to the selected file

        self.file_sum_check.config(text="")
        self.file_plot_check.config(text="")

        self.file_path = askopenfilenames(title="Select data files")

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

        print("File summary generated")
        self.file_sum_check.config(text="✓")

    def file_plot(self):
        # Make folder if it doesn't exist
        Path(f"{self.output_folder}/figures").mkdir(parents=True, exist_ok=True)

        # Calculates and save figures to figure folder
        data_plot = DataPlot(file_path=self.file_path)
        data_plot.cycle_avg_plot(figure_folder=f"{self.output_folder}/figures")
        data_plot.snapshot_plot(figure_folder=f"{self.output_folder}/figures")

        # Only plot cycle average by flow cell if multiple data files were selected
        if len(self.file_path) > 1:
            data_plot.cycle_avg_by_flowcell(figure_folder=f"{self.output_folder}/figures")


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
        self.label_folder_name.config(text='\n'.join(wrap(self.folder_path, 60)))

    def folder_summary(self):
        # ignore any files with extensions
        file_list = [f"{self.folder_path}/{f}" for f in listdir(f"{self.folder_path}")
                     if ("." not in f and os.path.isfile(f"{self.folder_path}/{f}"))]
        folder_name = self.folder_path.split("/")[-1]
        self.export_data_summary(file_list=file_list, summary_name=f"{folder_name} folder summary.xlsx")
        print("Folder summary generated")
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
            eo = flow_calculator.eo_flow_calculator()
            fom = flow_calculator.FOM_calculator()

            # remove flow cell designation from file name
            exp_name = file.split('/')[-1][:-3]
            print(exp_name)

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
        material_df = pd.DataFrame(material_list)
        material_df.drop_duplicates('file', ignore_index=True, inplace=True)

        with pd.ExcelWriter(f"{self.output_folder}/summary/{summary_name}") as writer:
            material_df.to_excel(writer, sheet_name="experiment setup", index=False)
            fom_df.to_excel(writer, sheet_name="figures of merit", index=False)
            flow_df.to_excel(writer, sheet_name="flow and power summary", index=False)
            eo_flow_df.to_excel(writer, sheet_name="pulse and cycle calc", index=False)

    def boxplot_plot(self):
        # df = pd.read_excel("canonical samples/boxplot_data.xlsx")
        # loop through all files in a folder
        file_list = [f for f in listdir(f"{self.folder_path}")
                     if ("." not in f and os.path.isfile(f"{self.folder_path}/{f}"))]
        list1 = []
        deltah = 0
        user_input = messagebox.askyesno("Plotting",
                                         "Change the height delta (default=0) for boxplot?")
        if user_input:
            deltah = simpledialog.askfloat("Height", "Enter height delta: ")
            if deltah not in [0, 0.1, 0.2]:
                messagebox.showinfo("Info", "Height delta can only be 0, 0.1, or 0.2 m. Resetting to 0 m")
                deltah = 0

        for file in file_list:
            file_path = f"{self.folder_path}/{file}"
            flow_calculator = FlowCalculator(file_path=file_path)
            df1 = flow_calculator.boxplot_calculator(deltah=deltah)

            # remove flow cell designation from file name
            exp_name = file[:-3]

            df1.insert(0, "file", exp_name)
            list1.append(df1)

        df = pd.concat(list1, ignore_index=True)
        exp_list = df["file"].unique()

        plot_label = {}
        for name in exp_list:
            # user_input = input(f"Please enter label for file <{name}>:\n")
            user_input = simpledialog.askstring("Plot sample name",
                                                f"Please enter label for experiment <{name}>:\n")
            # shorten original file name for plotting
            if len(user_input) > 0:
                plot_label[name] = user_input
            else:
                plot_label[name] = name

        df["sample"] = df["file"].map(plot_label, na_action="ignore")

        # print(df.columns)
        # print(df.head())

        # invert values for plotting
        df["net eo flow"] = df["net eo flow"] * -1
        df["net current"] = df["net current"] * -1

        # remove cycle 0 and 1
        df.drop(df[df["cycle"] < 2].index, inplace=True)

        # write to temp file for troubleshooting
        Path(f"{self.output_folder}/temp").mkdir(parents=True, exist_ok=True)
        df.to_excel(f"{self.output_folder}/temp/temp.xlsx")

        # write graph to figures folder
        Path(f"{self.output_folder}/figures/boxplot").mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

        plot_list = {"net eo flow": r"Net EO Flow [L/h/$m^{2}$]",
                     "total flow": r"Total Flow [L/h/$m^{2}$]",
                     "net current": r"Net Current [A/$m^{2}$]",
                     "total current": r"Total Current [A/$m^{2}$]"
                     }

        fig_width = len(file_list)
        fig_height = 4

        for y_name, y_label in plot_list.items():
            plt.figure(figsize=(fig_width, fig_height))
            sns.boxplot(df, x="sample", y=y_name, legend="full")
            plt.xticks(wrap=True)
            plt.ylabel(y_label)
            plt.savefig(f"{self.output_folder}/figures/boxplot/boxplot {y_name} deltah={deltah}.png",
                        bbox_inches='tight',
                        transparent=True)
            plt.close()

        self.boxplot_check.config(text="✓")

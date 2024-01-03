import sys
import numpy as np
import pandas as pd
import math
from sklearn import linear_model
from scipy.optimize import curve_fit
from circuit_models import CircuitModel

default_resistor = 47
default_diameter = 13 * 10 ** -3


class FlowCalculator:
    def __init__(self, file_path):
        self.file_path = file_path

        # get the comment file and write experiment conditions to the output summary sheet
        comment_file = self.file_path + ".comments"
        with open(comment_file, "r") as comment:
            text = comment.readlines()
        text = [n.replace("\n", "") for n in text]
        self.comment_df = pd.DataFrame(text)

        # Dictionary of parameters parsed from the comment file
        self.params = self.parse_comments()

        # Get the name of the cell from the file title (PO or WY)
        self.file_name = self.file_path.split("/")[-1]
        cell_pair = self.file_name.split("_")[-1]
        self.cell_1 = cell_pair[0]
        self.cell_2 = cell_pair[1]
        self.flowcell_list = [self.cell_1, self.cell_2]

        # 
        self.header = ["time",  # time series
                       "appv",  # set voltage (V)
                       "sli1",  # flow rate (uL/min)
                       "vtot1",  # actual voltage?
                       "vcur1",  # current?
                       "sli2",  # flow rate (uL/min)
                       "vtot2",  # actual voltage?
                       "vcur2",  # current?
                       "deltah",
                       "signal"
                       ]

        self.raw_data = pd.read_csv(filepath_or_buffer=self.file_path,
                                    engine="python",
                                    sep="  ",
                                    names=self.header,
                                    comment="#",
                                    on_bad_lines='skip'
                                    )

        # raw data preprocessing:
        # add cycle and phase count
        # drop ignored ranges from comment file
        # add calculated columns

        print(self.params)

        self.cycle_count()
        self.action_ignore()
        self.calc_columns()

        # Format time column for plotting, drop rows that cannot be parsed
        self.raw_data["time"] = pd.to_datetime(self.raw_data["time"].str.replace("_", " "),
                                               errors='coerce')
        self.raw_data.dropna(axis=0, subset=['time'], ignore_index=True, inplace=True)

        # Set up empty dictionary for the calculated results
        self.EO_flow_dict = {"flow cell": [],
                             "signal": [],
                             "appv": [],
                             "deltah": [],
                             'pressure flow (v=0)': [],
                             'pulse flow (+v)': [],
                             'pulse flow (-v)': [],
                             'cycle flow': [],
                             "eo flow (+v)": [],
                             "eo flow (-v)": [],
                             "pulse energy (+v)": [],
                             "pulse energy (-v)": [],
                             "cycle energy": []
                             }

    def parse_comments(self):
        params = {'period': None,
                  'diameter': None,
                  'resistor': None,
                  'membrane': None,
                  'electrode': None,
                  'iem': None,
                  'ignoredphases': [],
                  'ignoredcycles': [],
                  'ignoredranges': [],
                  'ignoredsignals': [],
                  'signaldefs': [],
                  'slinames': []
                  }
        comment_file = f"{self.file_path}.comments"
        with open(comment_file, "r") as comments:
            for line in comments:
                if line.startswith('U3_PERIOD'):
                    params['period'] = int(line.split('=')[1])
                if line.startswith('CHANNELS'):
                    # hacky hunt for sensor identifiers
                    for word in line.split():
                        if word.startswith('/tmp/sli'):
                            params['slinames'].append(word.split('/')[2])
                if line.startswith('Diameter'):
                    val = line.partition(":")[2].split()
                    if len(val) > 1 and val[1] != 'mm':
                        print('Error: Unexpected diameter unit')
                        sys.exit(1)
                    num = val[0].partition('mm')[0]  # if no space before mm
                    params['diameter'] = float(num) * 10 ** -3
                    print('Diameter specified: %f m' % params['diameter'])
                if line.startswith('Resistor'):
                    val = line.partition(":")[2].split()
                    if len(val) > 1 and val[1] != 'Ohm' and val[1] != 'ohm':
                        print('Error: Unexpected resistor unit')
                        sys.exit(1)
                    num = val[0].partition('ohm')[0]  # if no space before ohm
                    params['resistor'] = float(num)
                    print('Resistor specified: %f ohm' % params['resistor'])

                # Parse materials from comment file
                if line.startswith('Membrane'):
                    params['membrane'] = line.split(":")[1].strip()
                if line.startswith('Electrode'):
                    params['electrode'] = line.split(":")[1].strip()
                if line.startswith('Ion'):
                    params['iem'] = line.split(":")[1].strip()

                # Parse actions from comment file
                if line.startswith('Ignore-phase'):
                    phaselist = line.partition(":")[2].split()
                    for i in phaselist:
                        params['ignoredphases'].append(int(i))
                if line.startswith('Ignore-cycle'):
                    cyclelist = line.partition(":")[2].split()
                    for i in cyclelist:
                        params['ignoredcycles'].append(int(i))
                if line.startswith('Ignore-range'):
                    rangelist = line.partition(":")[2].split()
                    for i in rangelist:
                        first, last = i.split('-')
                        params['ignoredranges'].append((int(first), int(last)))
                if line.startswith('Ignore-signal'):
                    signallist = line.partition(":")[2].split()
                    for i in signallist:
                        params['ignoredsignals'].append(int(i))
                if line.startswith('Signal:'):
                    signallist = line.partition(":")[2].split()
                    params['signaldefs'].append(signallist)

        if len(params['slinames']) == 0:
            print('Comment file has no CHANNELS in configuration header?')
            sys.exit(1)

        if params['diameter'] is None:
            print('Diameter not specified, using default value %f m' % default_diameter)
            params['diameter'] = default_diameter

        if params['resistor'] is None:
            print('Resistor not specified, using default value %f ohm' % default_resistor)
            params['resistor'] = default_resistor

        return params

    def cycle_count(self):
        # Modifies self.raw_data when called
        # Add two columns to the raw data file - cycle count and phase change count
        # assuming -0.5 / 0.5 V cycles
        appv = list(self.raw_data["appv"])
        sig = list(self.raw_data["signal"])
        cnt_list = []
        cyc_list = []
        cyc = 0
        per_phase_cnt = 0
        first_phase = 0

        for i in range(len(appv)):
            if sig[i] == 0:
                first_phase = 0
                per_phase_cnt = 0
                cyc = 0
            else:
                if appv[i] != 0 and first_phase == 0:
                    first_phase = appv[i]

                if appv[i] != appv[i - 1] and appv[i] == first_phase:
                    cyc += 1

                if appv[i] != appv[i - 1]:
                    per_phase_cnt = 0
                else:
                    per_phase_cnt += 1

            cnt_list.append(per_phase_cnt)
            cyc_list.append(cyc)

        self.raw_data["cnt"] = cnt_list
        self.raw_data["cyc"] = cyc_list

    def action_ignore(self):
        # Modifies self.raw_data when called
        # Drop ranges specified in the comment file

        df = self.raw_data
        if len(self.params["ignoredranges"]) > 0:
            for n in self.params["ignoredranges"]:
                range_start = n[0]
                range_end = n[1]
                df.drop(index=range(range_start, range_end), inplace=True)

        # mark_ignored_phases(m, self.params['ignoredphases'])
        if len(self.params["ignoredphases"]) > 0:
            for cnt in self.params["ignoredphases"]:
                df.drop(df[df['cnt'] == cnt].index, inplace=True)

        # mark_ignored_cycles(m, self.params['ignoredcycles'])
        if len(self.params["ignoredcycles"]) > 0:
            for cyc in self.params["ignoredcycles"]:
                df.drop(df[df['cyc'] == cyc].index, inplace=True)

        if len(self.params["ignoredsignals"]) > 0:
            for signal in self.params["ignoredsignals"]:
                df.drop(df[df['signal'] == signal].index, inplace=True)

    def calc_columns(self):
        unit_conversion = 1e-6 * 60  # convert unit from uL/min to L/hr
        cell_area = self.params['diameter'] ** 2 * math.pi / 4  # cell area with 13 mm diameter cell

        df = self.raw_data

        # specific flow in units of L/hour/m^2
        df['flow1'] = df["sli1"] * unit_conversion / cell_area
        df['flow2'] = df["sli2"] * unit_conversion / cell_area

        # current density in units of A/m^2
        df['cur1'] = df["vcur1"] / self.params["resistor"] / cell_area
        df['cur2'] = df["vcur2"] / self.params["resistor"] / cell_area

        # power consumption in Watt/m^2 (P = I*V)
        df['pwr1'] = df['cur1'] * df["vtot1"]
        df['pwr2'] = df['cur2'] * df["vtot2"]

    def boxplot_calculator(self, deltah=0):
        # Calculates mean flow and mean current for each cycle
        df = self.raw_data.copy()
        df = df.loc[(
                (df["signal"] != 0)
                & (df["deltah"] == deltah)
        )]

        flow1_df = df.groupby(["signal", "deltah", "appv", "cyc"], as_index=False)["flow1"].mean()
        flow2_df = df.groupby(["signal", "deltah", "appv", "cyc"], as_index=False)["flow2"].mean()

        flow1_df.insert(0, "flow cell", self.cell_1)
        flow2_df.insert(0, "flow cell", self.cell_2)
        flow1_df.rename(columns={"flow1": "mean flow"}, inplace=True)
        flow2_df.rename(columns={"flow2": "mean flow"}, inplace=True)

        cur1_df = df.groupby(["signal", "deltah", "appv", "cyc"], as_index=False)["cur1"].mean()
        cur2_df = df.groupby(["signal", "deltah", "appv", "cyc"], as_index=False)["cur2"].mean()

        cur1_df.insert(0, "flow cell", self.cell_1)
        cur2_df.insert(0, "flow cell", self.cell_2)
        cur1_df.rename(columns={"cur1": "current"}, inplace=True)
        cur2_df.rename(columns={"cur2": "current"}, inplace=True)

        flow_df = pd.concat([flow1_df, flow2_df], ignore_index=True)
        cur_df = pd.concat([cur1_df, cur2_df], ignore_index=True)
        cycle_df = flow_df.merge(cur_df)

        # Boxplot dataframe
        cyc_dict = {"flow cell": [],
                    "cycle": [],
                    "net eo flow": [],
                    "total flow": [],
                    "net current": [],
                    "total current": []
                    }

        # List of unique conditions to filter for
        cyc_list = cycle_df["cyc"].unique()

        for flowcell in self.flowcell_list:
            mean_flow_zero = cycle_df.loc[((cycle_df["appv"] == 0)
                                           & (cycle_df["flow cell"] == flowcell))]["mean flow"].mean()
            for cyc in cyc_list:
                new_df = cycle_df.loc[((cycle_df["flow cell"] == flowcell) & (cycle_df["cyc"] == cyc))]

                mean_flow_pos = new_df.loc[(new_df["appv"] > 0)]["mean flow"].mean()
                mean_flow_neg = new_df.loc[(new_df["appv"] < 0)]["mean flow"].mean()

                EO_flow_pos = (mean_flow_pos - mean_flow_zero)
                EO_flow_neg = (mean_flow_neg - mean_flow_zero)
                net_EO_flow = (EO_flow_pos + EO_flow_neg) / 2
                total_flow = (abs(EO_flow_pos) + abs(EO_flow_neg)) / 2

                current_pos = new_df.loc[(new_df["appv"] > 0)]["current"].mean()
                current_neg = new_df.loc[(new_df["appv"] < 0)]["current"].mean()

                net_current = (current_pos + current_neg)
                total_current = (abs(current_pos) + abs(current_neg))

                cyc_dict["flow cell"].append(flowcell)
                cyc_dict["cycle"].append(cyc)
                cyc_dict["net eo flow"].append(round(net_EO_flow, 3))
                cyc_dict["total flow"].append(round(total_flow, 3))
                cyc_dict["net current"].append(round(net_current, 3))
                cyc_dict["total current"].append(round(total_current, 3))

        boxplot_df = pd.DataFrame(cyc_dict)
        return boxplot_df

    def mean_flow_calculator(self):
        # Calculates mean flow for each condition
        # Returns separate data frame file called flow_df to be used
        # in subsequent calculations for eo flow and hydraulic resistance

        df = self.raw_data.copy()
        df = df.loc[((df["signal"] != 0) & ((df["cyc"] > 1) | (df["appv"] == 0)))]

        flow1_df = df.groupby(["signal", "deltah", "appv"], as_index=False)["flow1"].mean()
        flow2_df = df.groupby(["signal", "deltah", "appv"], as_index=False)["flow2"].mean()

        flow1_df.insert(0, "flow cell", self.cell_1)
        flow2_df.insert(0, "flow cell", self.cell_2)
        flow1_df.rename(columns={"flow1": "mean flow"}, inplace=True)
        flow2_df.rename(columns={"flow2": "mean flow"}, inplace=True)

        pwr1_df = df.groupby(["signal", "deltah", "appv"], as_index=False)["pwr1"].mean()
        pwr2_df = df.groupby(["signal", "deltah", "appv"], as_index=False)["pwr2"].mean()

        pwr1_df.insert(0, "flow cell", self.cell_1)
        pwr2_df.insert(0, "flow cell", self.cell_2)
        pwr1_df.rename(columns={"pwr1": "power"}, inplace=True)
        pwr2_df.rename(columns={"pwr2": "power"}, inplace=True)

        pwr_df = pd.concat([pwr1_df, pwr2_df])
        flow_df = pd.concat([flow1_df, flow2_df])
        flow_df = flow_df.merge(pwr_df)

        return flow_df

    def eo_flow_calculator(self):
        # Takes flow_df from mean_flow_calculator and performs the following calculations:
        # Returns a new dataframe
        flow_df = self.mean_flow_calculator()

        # List of unique conditions to filter for
        signal_list = list(flow_df["signal"].unique())
        delta_h_list = list(flow_df["deltah"].unique())
        voltage_list = list(flow_df['appv'].abs().unique())
        voltage_list = [n for n in voltage_list if n > 0]

        for flowcell in self.flowcell_list:
            for signal in signal_list:
                for delta_h in delta_h_list:
                    for voltage in voltage_list:
                        new_df = flow_df.loc[(
                                (flow_df["deltah"] == delta_h)
                                & (flow_df["signal"] == signal)
                                & (flow_df["flow cell"] == flowcell)
                        )]
                        pressure_flow = new_df.loc[(new_df["appv"] == 0)]["mean flow"].mean()
                        pulse_flow_pos = new_df.loc[(new_df["appv"] == voltage)]["mean flow"].mean()
                        pulse_flow_neg = new_df.loc[(new_df["appv"] == -voltage)]["mean flow"].mean()
                        cycle_flow = (pulse_flow_pos + pulse_flow_neg) / 2

                        eo_flow_pos = pulse_flow_pos - pressure_flow
                        eo_flow_neg = pulse_flow_neg - pressure_flow

                        pulse_energy_pos = new_df.loc[(new_df["appv"] == voltage)]["power"].mean() / pulse_flow_pos
                        pulse_energy_neg = new_df.loc[(new_df["appv"] == -voltage)]["power"].mean() / pulse_flow_neg
                        cycle_energy = (pulse_energy_pos + pulse_energy_neg) / 2

                        # Only write to dictionary if it's not empty
                        if not np.isnan(pulse_flow_pos):
                            self.EO_flow_dict["signal"].append(signal)
                            self.EO_flow_dict["appv"].append(voltage)
                            self.EO_flow_dict["deltah"].append(delta_h)
                            self.EO_flow_dict["flow cell"].append(flowcell)
                            self.EO_flow_dict["pressure flow (v=0)"].append(round(pressure_flow, 3))
                            self.EO_flow_dict["pulse flow (+v)"].append(round(pulse_flow_pos, 3))
                            self.EO_flow_dict["pulse flow (-v)"].append(round(pulse_flow_neg, 3))
                            self.EO_flow_dict["cycle flow"].append(round(cycle_flow, 3))
                            self.EO_flow_dict["eo flow (+v)"].append(round(eo_flow_pos, 3))
                            self.EO_flow_dict["eo flow (-v)"].append(round(eo_flow_neg, 3))
                            self.EO_flow_dict["pulse energy (+v)"].append(round(pulse_energy_pos, 3))
                            self.EO_flow_dict["pulse energy (-v)"].append(round(pulse_energy_neg, 3))
                            self.EO_flow_dict["cycle energy"].append(round(cycle_energy, 3))

        eo_flow_df = pd.DataFrame(self.EO_flow_dict)
        return eo_flow_df

    def FOM_calculator(self):
        """
        Figures of merit:
        1) Pulse Flow (h=0)
        2) Cycle Flow (h=0)
        3) Hydrodynamic resistance/permeability (linear fit) (V=0)
        4) Pulse Pressure (height at Pulse flow=0)
        5) Cycle pressure (height at Cycle flow=0)
        6) Pulse Energy consumption [Wh /L] (Pulse Flow , h=0)
        7) Cycle Energy consumption [Wh/L] (Pulse Flow , h=0)

        h: water column height [m]
        V: applied voltage [V]

        :return:
        """
        FOM = {'flow cell': [],
               'signal': [],
               'appv': [],
               'pressure flow (h=0)': [],
               'neg pulse flow (h=0)': [],
               'pos pulse flow (h=0)': [],
               'cycle flow (h=0)': [],
               'hydraulic permeability': [],
               'pulse pressure (m)': [],
               'cycle pressure (m)': [],
               'pulse energy consumption (Wh/L)': [],
               'cycle energy consumption (Wh/L)': [],
               }

        flow_df = self.eo_flow_calculator()
        good_fit = 0.7  # cut-off for R^2 value

        # List of unique conditions to filter for
        signal_list = list(flow_df["signal"].unique())
        voltage_list = list(flow_df['appv'].unique())

        for flowcell in self.flowcell_list:
            for signal in signal_list:
                for voltage in voltage_list:
                    new_df = flow_df.loc[(
                            (flow_df["appv"] == voltage)
                            & (flow_df["signal"] == signal)
                            & (flow_df["flow cell"] == flowcell)
                    )]

                    pressure_flow = new_df.loc[(new_df["deltah"] == 0)]["pressure flow (v=0)"].mean()
                    pulse_flow_neg = new_df.loc[(new_df["deltah"] == 0)]["pulse flow (-v)"].mean()
                    pulse_flow_pos = new_df.loc[(new_df["deltah"] == 0)]["pulse flow (+v)"].mean()
                    pulse_energy_neg = new_df.loc[(new_df["deltah"] == 0)]["pulse energy (-v)"].mean()
                    cycle_flow = new_df.loc[(new_df["deltah"] == 0)]["cycle flow"].mean()
                    cycle_energy = new_df.loc[(new_df["deltah"] == 0)]["cycle energy"].mean()

                    # Only write to dictionary if it's not empty
                    if not np.isnan(pulse_flow_neg):
                        # hydraulic permeability (slope of linear fit vs deltah, v = 0)
                        x = new_df["deltah"].values
                        y1 = new_df["pressure flow (v=0)"].values

                        # calculate hydraulic permeability by linear regression
                        length = len(x)
                        x = x.reshape(length, 1)
                        y1 = y1.reshape(length, 1)
                        reg = linear_model.LinearRegression(fit_intercept=False)
                        reg.fit(x, y1)
                        score1 = reg.score(x, y1)
                        if score1 >= good_fit:
                            hydraulic_perm = reg.coef_[0][0]
                            hydraulic_perm = round(hydraulic_perm, 3)

                        else:
                            hydraulic_perm = 'n/a'

                        # pulse pressure (x-intercept of linear fit negative pulse flow vs deltah)
                        y2 = new_df["pulse flow (-v)"].values
                        y2 = y2.reshape(length, 1)
                        reg = linear_model.LinearRegression()
                        reg.fit(x, y2)
                        slope = reg.coef_[0][0]
                        intercept = reg.intercept_[0]
                        score2 = reg.score(x, y2)
                        if (score2 >= good_fit) and (slope > 0):
                            pulse_pressure = -intercept / slope
                            pulse_pressure = round(pulse_pressure, 3)
                        else:
                            pulse_pressure = 'n/a'

                        # cycle pressure (x-intercept of linear fit cycle_flow vs deltah)
                        y3 = new_df["cycle flow"].values
                        y3 = y3.reshape(length, 1)
                        reg = linear_model.LinearRegression()
                        reg.fit(x, y3)
                        intercept = reg.intercept_[0]
                        slope = reg.coef_[0][0]
                        score3 = reg.score(x, y3)
                        if (score3 >= good_fit) and (slope > 0):
                            cycle_pressure = -intercept / slope
                            round(cycle_pressure, 3)
                        else:
                            cycle_pressure = 'n/a'

                        FOM['flow cell'].append(flowcell)
                        FOM['signal'].append(signal)
                        FOM['appv'].append(voltage)
                        FOM['pressure flow (h=0)'].append(pressure_flow)
                        FOM['neg pulse flow (h=0)'].append(round(pulse_flow_neg, 3))
                        FOM['pos pulse flow (h=0)'].append(round(pulse_flow_pos, 3))
                        FOM['cycle flow (h=0)'].append(round(cycle_flow, 3))
                        FOM['pulse energy consumption (Wh/L)'].append(round(pulse_energy_neg, 3))
                        FOM['cycle energy consumption (Wh/L)'].append(round(cycle_energy, 3))
                        FOM['hydraulic permeability'].append(hydraulic_perm)
                        FOM['pulse pressure (m)'].append(pulse_pressure)
                        FOM['cycle pressure (m)'].append(cycle_pressure)

        FOM = pd.DataFrame(FOM)

        # merge equivalent circuit fitting results
        circuit = self.circuit_fitting()
        FOM = FOM.merge(circuit)

        return FOM

    def circuit_fitting(self):
        cell_area = self.params['diameter'] ** 2 * math.pi / 4  # cell area with 13 mm diameter cell
        raw_data = self.raw_data
        circuit_results = pd.DataFrame()

        exp_df = raw_data.loc[((raw_data['signal'] != 0)
                               & (raw_data['deltah'] == 0)
                               & (raw_data['cyc'] > 1))]
        exp_df = exp_df.copy()

        # construct relative time (seconds) column
        exp_df["rel time"] = exp_df["time"] - exp_df["time"].iloc[0]
        exp_df["rel time"] = exp_df["rel time"].dt.total_seconds()

        exp_voltage = exp_df['appv'].to_numpy()
        exp_time = exp_df['rel time'].to_numpy()
        xdata = pd.DataFrame({'time': exp_time, 'appv': exp_voltage})

        # Convert specific current back to total current (A) for fitting by multiplying by cell area
        exp_cur1 = np.multiply(exp_df['cur1'].to_numpy(), cell_area)
        exp_cur2 = np.multiply(exp_df['cur2'].to_numpy(), cell_area)

        # import circuit model class
        circuit_model = CircuitModel(time=exp_time)
        popt1, pcov1 = curve_fit(f=circuit_model.circuit_RC, xdata=xdata, ydata=exp_cur1, p0=[100, 1],
                                 bounds=(0, [np.inf, np.inf]))
        cell1_R, cell1_C = popt1[0], popt1[1]
        popt2, pcov2 = curve_fit(f=circuit_model.circuit_RC, xdata=xdata, ydata=exp_cur2, p0=[100, 1],
                                 bounds=(0, [np.inf, np.inf]))
        cell2_R, cell2_C = popt2[0], popt2[1]

        # evaluate goodness of fit
        sim1_cur = circuit_model.circuit_RC(xdata=xdata, resistance=cell1_R, capacitance=cell1_C)
        sim2_cur = circuit_model.circuit_RC(xdata=xdata, resistance=cell2_R, capacitance=cell2_C)

        sim1_fit = self.r_squared_calc(exp_cur1, sim1_cur)
        sim2_fit = self.r_squared_calc(exp_cur2, sim2_cur)

        if sim1_fit < 0.7:
            cell1_R, cell1_C = 'n/a', 'n/a'

        if sim2_fit < 0.7:
            cell2_R, cell2_C = 'n/a', 'n/a'

        results = pd.DataFrame({'flow cell': self.flowcell_list,
                                'Resistance (Ohms)': [cell1_R, cell2_R],
                                'Capacitance (F)': [cell1_C, cell2_C],
                                })
        circuit_results = pd.concat([circuit_results, results])

        return circuit_results

    def r_squared_calc(self, y, y_fit):
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

import sys
import numpy as np
import pandas as pd
import math
from sklearn import linear_model
from scipy.optimize import curve_fit
from circuit_models import CircuitModel

default_resistor = 47
default_diameter = 13 * 10 ** -3
# default date time format: '%Y-%m-%d_%H:%M:%S' (e.g. 2024-01-01_09:00:00)
# for older files (before 2023) use format: '%H:%M:%S'
date_format = '%Y-%m-%d_%H:%M:%S'


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
                                    sep="  | ",
                                    names=self.header,
                                    comment="#",
                                    on_bad_lines='skip'
                                    )
        # print(self.params)

        # Format time column for plotting, drop rows that cannot be parsed and duplicate rows to keep to 1 point/sec
        self.raw_data['time'] = pd.to_datetime(self.raw_data["time"], format=date_format,
                                               errors="coerce")
        self.raw_data.dropna(axis=0, subset=['time'], inplace=True, ignore_index=True)
        self.raw_data.drop_duplicates(subset=['time'], inplace=True, ignore_index=True)

        # raw data preprocessing:
        # add cycle and phase count
        self.cycle_count()
        # drop ignored ranges from comment file
        self.action_ignore()
        # add calculated columns
        self.calc_columns()

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
                    # print('Diameter specified: %f m' % params['diameter'])
                if line.startswith('Resistor'):
                    val = line.partition(":")[2].split()
                    if len(val) > 1 and val[1] != 'Ohm' and val[1] != 'ohm':
                        print('Error: Unexpected resistor unit')
                        sys.exit(1)
                    num = val[0].partition('ohm')[0]  # if no space before ohm
                    params['resistor'] = float(num)
                    # print('Resistor specified: %f ohm' % params['resistor'])

                # Parse materials from comment file
                if line.startswith('Membrane'):
                    text = [f.strip() for f in line.split(":")[1:]]
                    params['membrane'] = ' '.join(text)
                if line.startswith('Electrode'):
                    text = [f.strip() for f in line.split(":")[1:]]
                    params['electrode'] = ' '.join(text)
                if line.startswith('Ion'):
                    text = [f.strip() for f in line.split(":")[1:]]
                    params['iem'] = ' '.join(text)

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
            # print('Diameter not specified, using default value %f m' % default_diameter)
            params['diameter'] = default_diameter

        if params['resistor'] is None:
            # print('Resistor not specified, using default value %f ohm' % default_resistor)
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

        # Add in relative time column
        df['rel time'] = df.reset_index()['time'].diff().dt.total_seconds().fillna(0).cumsum().values

        # specific flow in units of L/hour/m^2
        df['flow1'] = df["sli1"] * unit_conversion / cell_area
        df['flow2'] = df["sli2"] * unit_conversion / cell_area

        # current density in units of A/m^2
        df['cur1'] = df["vcur1"] / self.params["resistor"] / cell_area
        df['cur2'] = df["vcur2"] / self.params["resistor"] / cell_area

        # power consumption in Watt/m^2 (P = I*V)
        df['pwr1'] = df['cur1'] * df["vtot1"]
        df['pwr2'] = df['cur2'] * df["vtot2"]

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

        cur1_df = df.groupby(["signal", "deltah", "appv"], as_index=False)["cur1"].mean()
        cur2_df = df.groupby(["signal", "deltah", "appv"], as_index=False)["cur2"].mean()

        cur1_df.insert(0, "flow cell", self.cell_1)
        cur2_df.insert(0, "flow cell", self.cell_2)
        cur1_df.rename(columns={"cur1": "current"}, inplace=True)
        cur2_df.rename(columns={"cur2": "current"}, inplace=True)

        pwr_df = pd.concat([pwr1_df, pwr2_df])
        cur_df = pd.concat([cur1_df, cur2_df])
        flow_df = pd.concat([flow1_df, flow2_df])
        flow_df = flow_df.merge(pwr_df).merge(cur_df)

        return flow_df

    def pulse_cycle_calculator(self):
        # Takes flow_df from mean_flow_calculator and performs the following calculations:
        # Returns a new dataframe
        flow_df = self.mean_flow_calculator()

        # Set up empty dictionary for the calculated results
        EO_flow_dict = {"flow cell": [],
                        "signal": [],
                        "appv": [],
                        "deltah": [],
                        'duty cycle': [],
                        'pressure flow (v=0)': [],
                        'pulse flow (+v)': [],
                        'pulse flow (-v)': [],
                        'cycle flow': [],
                        "eo flow (+v)": [],
                        "eo flow (-v)": [],
                        "pulse energy (+v)": [],
                        "pulse energy (-v)": [],
                        "cycle energy": [],
                        "streaming current (v=0)": [],
                        "cycle current": [],
                        "cycle abs current": [],
                        "cycle power": []
                        }

        # List of unique conditions to filter for
        signal_list = list(flow_df["signal"].unique())
        delta_h_list = list(flow_df["deltah"].unique())

        for flowcell in self.flowcell_list:
            for signal in signal_list:
                for delta_h in delta_h_list:
                    new_df = flow_df.loc[(
                            (flow_df["deltah"] == delta_h)
                            & (flow_df["signal"] == signal)
                            & (flow_df["flow cell"] == flowcell)
                    )]

                    # Only write to dictionary if it's not empty
                    if not new_df.empty:
                        # list voltages applied
                        # If symmetrical record only the magnitude appv
                        # else record in the format of 'pos v/neg v'
                        appv = [appv for appv in list(new_df['appv'].unique()) if appv != 0]
                        if sum(appv) == 0:
                            appv = abs(appv[0])
                        else:
                            appv = '/'.join([str(n) for n in appv])

                        # calculate duty cycle as % of time NEGATIVE voltage is applied, ignoring the first cycle
                        df = self.raw_data.copy()
                        df = df.loc[((df["signal"] == signal) & (df["deltah"] == delta_h) & (df["cyc"] > 1))]
                        duty_cycle = df.loc[(df['appv'] < 0)]['appv'].count() / df.loc[(df['appv'] != 0)][
                            'appv'].count()

                        # Because the data is in 1 sec intervals
                        # Integral of the variable over time is equivalent to the mean
                        # Trapezoidal rule (b-a)*[y(a)+y(b)]/2 = 1/2*[y(a) + y(b)]

                        pressure_flow = new_df.loc[(new_df["appv"] == 0)]["mean flow"].mean()
                        pulse_flow_pos = new_df.loc[(new_df["appv"] > 0)]["mean flow"].mean()
                        pulse_flow_neg = new_df.loc[(new_df["appv"] < 0)]["mean flow"].mean()

                        eo_flow_pos = pulse_flow_pos - pressure_flow
                        eo_flow_neg = pulse_flow_neg - pressure_flow

                        pulse_energy_pos = new_df.loc[(new_df["appv"] > 0)]["power"].mean() / pulse_flow_pos
                        pulse_energy_neg = new_df.loc[(new_df["appv"] < 0)]["power"].mean() / pulse_flow_neg

                        streaming_current = new_df.loc[(new_df["appv"] == 0)]["current"].mean()
                        pulse_cur_pos = new_df.loc[(new_df["appv"] > 0)]["current"].mean()
                        pulse_cur_neg = new_df.loc[(new_df["appv"] < 0)]["current"].mean()
                        cycle_mean_cur = (pulse_cur_pos + pulse_cur_neg)
                        cycle_mean_cur_abs = abs(pulse_cur_pos) + abs(pulse_cur_neg)

                        # modify cycle calculation  for duty cycle != 50%
                        cycle_flow = pulse_flow_pos * (1 - duty_cycle) + pulse_flow_neg * duty_cycle
                        cycle_energy = pulse_energy_pos * (1 - duty_cycle) + pulse_energy_neg * duty_cycle
                        cycle_pwr = np.trapz(new_df["power"])

                        # Write to dictionary
                        EO_flow_dict["signal"].append(signal)
                        EO_flow_dict["appv"].append(appv)
                        EO_flow_dict["deltah"].append(delta_h)
                        EO_flow_dict['duty cycle'].append(round(duty_cycle, 3))
                        EO_flow_dict["flow cell"].append(flowcell)
                        EO_flow_dict["pressure flow (v=0)"].append(round(pressure_flow, 3))
                        EO_flow_dict["pulse flow (+v)"].append(round(pulse_flow_pos, 3))
                        EO_flow_dict["pulse flow (-v)"].append(round(pulse_flow_neg, 3))
                        EO_flow_dict["cycle flow"].append(round(cycle_flow, 3))
                        EO_flow_dict["eo flow (+v)"].append(round(eo_flow_pos, 3))
                        EO_flow_dict["eo flow (-v)"].append(round(eo_flow_neg, 3))
                        EO_flow_dict["pulse energy (+v)"].append(round(pulse_energy_pos, 3))
                        EO_flow_dict["pulse energy (-v)"].append(round(pulse_energy_neg, 3))
                        EO_flow_dict["cycle energy"].append(round(cycle_energy, 3))
                        EO_flow_dict["streaming current (v=0)"].append(streaming_current)
                        EO_flow_dict["cycle current"].append(round(cycle_mean_cur, 3))
                        EO_flow_dict["cycle abs current"].append(round(cycle_mean_cur_abs, 3))
                        EO_flow_dict["cycle power"].append(round(cycle_pwr, 3))

        eo_flow_df = pd.DataFrame(EO_flow_dict)
        return eo_flow_df

    def FOM_calculator(self):
        """
        Figures of merit:
        1) Pulse Flow (h=0)
        2) Cycle Flow (h=0)
        3) Hydrodynamic resistance/permeability (linear fit) (V=0)
        4) Pulse Pressure (height at Pulse flow=0)
        5) Cycle pressure (height at Cycle flow=0)
        6) Pulse Energy consumption [Wh/L] (Pulse Flow , h=0)
        7) Cycle Energy consumption [Wh/L] (Pulse Flow , h=0)

        h: water column height [m]
        V: applied voltage [V]

        :return:
        """
        FOM = {'flow cell': [],
               'signal': [],
               'appv': [],
               'duty cycle': [],
               'pressure flow (L/h/m^2)': [],
               'neg pulse flow (L/h/m^2)': [],
               'pos pulse flow (L/h/m^2)': [],
               'cycle flow (L/h/m^2)': [],
               'hydraulic permeability': [],
               'pulse pressure (m)': [],
               'cycle pressure (m)': [],
               'pulse energy consumption (Wh/L)': [],
               'cycle energy consumption (Wh/L)': [],
               'cycle current (A/m^2)': [],
               'cycle abs current (A/m^2)': [],
               'cycle abs flow (L/h/m^2)': [],
               'work in (W/m^2)': [],
               'work out (W/m^2)': [],
               'cycle efficiency': []
               }

        flow_df = self.pulse_cycle_calculator()
        good_fit = 0.8  # cut-off for R^2 value

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

                    duty_cycle = new_df.loc[(new_df["deltah"] == 0)]["duty cycle"].mean()

                    pressure_flow = new_df.loc[(new_df["deltah"] == 0)]["pressure flow (v=0)"].mean()
                    pulse_flow_neg = new_df.loc[(new_df["deltah"] == 0)]["pulse flow (-v)"].mean()
                    pulse_flow_pos = new_df.loc[(new_df["deltah"] == 0)]["pulse flow (+v)"].mean()
                    pulse_energy_neg = new_df.loc[(new_df["deltah"] == 0)]["pulse energy (-v)"].mean()
                    cycle_flow = new_df.loc[(new_df["deltah"] == 0)]["cycle flow"].mean()
                    cycle_energy = new_df.loc[(new_df["deltah"] == 0)]["cycle energy"].mean()
                    cycle_cur = new_df.loc[(new_df["deltah"] == 0)]["cycle current"].mean()
                    cycle_abs_cur = new_df.loc[(new_df["deltah"] == 0)]["cycle abs current"].mean()
                    cycle_abs_flow = abs(pulse_flow_pos) + abs(pulse_flow_neg)

                    # efficiency calculation
                    # work in: integral( I (A/m^2) * V) -> W/m^2
                    work_in = new_df.loc[(new_df["deltah"] == 0)]["cycle power"].mean()

                    # Only write to dictionary if it's not empty
                    if not np.isnan(pulse_flow_neg):
                        # hydraulic permeability (slope of linear fit vs deltah, v = 0)
                        x = new_df["deltah"].values
                        y1 = new_df["pressure flow (v=0)"].values

                        # calculate hydraulic permeability by linear regression
                        length = len(x)
                        if length > 1:
                            x = x.reshape(length, 1)
                            y1 = y1.reshape(length, 1)
                            reg = linear_model.LinearRegression(fit_intercept=False)
                            reg.fit(x, y1)
                            score1 = reg.score(x, y1)

                            # check the fit is good and slope is positive
                            if (score1 >= good_fit) and (reg.coef_[0][0] > 0):
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
                                if pulse_pressure < 0:
                                    pulse_pressure = 0
                                # check that it's a reasonable number (10 meter water column max)
                                elif 0 < pulse_pressure < 10:
                                    pulse_pressure = round(pulse_pressure, 3)
                                else:
                                    pulse_pressure = 'n/a'
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

                                if cycle_pressure < 0:
                                    cycle_pressure = 0
                                    work_out = 0
                                    max_eff = 0
                                # check that it's a reasonable number (10 meter water column max)
                                elif 0 < cycle_pressure < 10:
                                    # max efficiency calculation
                                    # work out : J * P, J = J_max / 2 and P = P_max / 2
                                    # work out : max cycle flow (m^3/m^2/s) * max cycle pressure (Pa) / 4 -> W/m^2
                                    work_out = (abs(cycle_flow) / 1000 / 3600) * (cycle_pressure * 9.81 * 1000) / 4
                                    max_eff = work_out / work_in
                                    cycle_pressure = round(cycle_pressure, 3)
                                else:
                                    cycle_pressure = 'n/a'
                                    work_out = 'n/a'
                                    max_eff = 'n/a'
                            else:
                                cycle_pressure = 'n/a'
                                work_out = 'n/a'
                                max_eff = 'n/a'
                        else:
                            hydraulic_perm = 'n/a'
                            pulse_pressure = 'n/a'
                            cycle_pressure = 'n/a'
                            work_out = 'n/a'
                            max_eff = 'n/a'

                        FOM['flow cell'].append(flowcell)
                        FOM['signal'].append(signal)
                        FOM['appv'].append(voltage)
                        FOM['duty cycle'].append(round(duty_cycle, 3))
                        FOM['pressure flow (L/h/m^2)'].append(round(pressure_flow, 3))
                        FOM['neg pulse flow (L/h/m^2)'].append(round(pulse_flow_neg, 3))
                        FOM['pos pulse flow (L/h/m^2)'].append(round(pulse_flow_pos, 3))
                        FOM['cycle flow (L/h/m^2)'].append(round(cycle_flow, 3))
                        FOM['pulse energy consumption (Wh/L)'].append(round(pulse_energy_neg, 3))
                        FOM['cycle energy consumption (Wh/L)'].append(round(cycle_energy, 3))
                        FOM['hydraulic permeability'].append(hydraulic_perm)
                        FOM['pulse pressure (m)'].append(pulse_pressure)
                        FOM['cycle pressure (m)'].append(cycle_pressure)
                        FOM['cycle current (A/m^2)'].append(round(cycle_cur, 3))
                        FOM['cycle abs current (A/m^2)'].append(round(cycle_abs_cur, 3))
                        FOM['cycle abs flow (L/h/m^2)'].append(round(cycle_abs_flow, 3))
                        FOM['work in (W/m^2)'].append(work_in)
                        FOM['work out (W/m^2)'].append(work_out)
                        FOM['cycle efficiency'].append(max_eff)

        FOM = pd.DataFrame(FOM)

        # merge equivalent circuit fitting results
        # circuit = self.circuit_fitting()
        # FOM = FOM.merge(circuit)

        return FOM

    def circuit_fitting(self):
        cell_area = self.params['diameter'] ** 2 * math.pi / 4  # cell area with 13 mm diameter cell
        raw_data = self.raw_data
        circuit_results = pd.DataFrame()

        # List of unique conditions to filter for
        signal_list = list(raw_data['signal'].unique())
        signal_list.pop(0)  # drop signal 0

        for signal in signal_list:
            exp_df = raw_data.loc[((raw_data['signal'] == signal) & (raw_data['deltah'] == 0))]
            if not exp_df.empty:
                exp_df = exp_df.copy()

                # construct relative time (seconds) column
                exp_df["rel time"] = exp_df["time"] - exp_df["time"].iloc[0]
                exp_df["rel time"] = exp_df["rel time"].dt.total_seconds()

                exp_voltage = exp_df['appv'].to_numpy()
                exp_time = exp_df['rel time'].to_numpy()
                xdata = pd.DataFrame({'time': exp_time, 'appv': exp_voltage})

                # Convert specific current back to total current (A/m^2) for fitting by multiplying by cell area
                exp_cur1 = np.multiply(exp_df['cur1'].to_numpy(), cell_area)
                exp_cur2 = np.multiply(exp_df['cur2'].to_numpy(), cell_area)

                # import circuit model class R(RC)
                circuit_model = CircuitModel(time=exp_time)

                # add in try except block to catch runtime error for curve_fit
                try:
                    popt1, pcov1 = curve_fit(f=circuit_model.circuit_RRC, xdata=xdata, ydata=exp_cur1,
                                             p0=[100, 100, 1],
                                             bounds=(0, [np.inf, np.inf, np.inf]))

                    popt2, pcov2 = curve_fit(f=circuit_model.circuit_RRC, xdata=xdata, ydata=exp_cur2,
                                             p0=[100, 100, 1],
                                             bounds=(0, [np.inf, np.inf, np.inf]))
                except RuntimeError:
                    print("Optimal parameters not found, set values to n/a")
                    cell1_R1, cell1_R2, cell1_C = 'n/a', 'n/a', 'n/a'
                    cell2_R1, cell2_R2, cell2_C = 'n/a', 'n/a', 'n/a'
                else:
                    cell1_R1, cell1_R2, cell1_C = popt1[0], popt1[1], popt1[2]
                    cell2_R1, cell2_R2, cell2_C = popt2[0], popt2[1], popt2[2]

                    # evaluate goodness of fit
                    sim1_cur = circuit_model.circuit_RRC(xdata=xdata, R1=cell1_R1, R2=cell1_R2, C=cell1_C)
                    sim2_cur = circuit_model.circuit_RRC(xdata=xdata, R1=cell2_R1, R2=cell2_R2, C=cell2_C)

                    sim1_fit = self.r_squared_calc(exp_cur1, sim1_cur)
                    sim2_fit = self.r_squared_calc(exp_cur2, sim2_cur)

                    if sim1_fit < 0.7:
                        cell1_R1, cell1_R2, cell1_C = 'n/a', 'n/a', 'n/a'
                    if sim2_fit < 0.7:
                        cell2_R1, cell1_R2, cell2_C = 'n/a', 'n/a', 'n/a'

                results = pd.DataFrame({'flow cell': self.flowcell_list,
                                        'signal': [signal, signal],
                                        'R1 (Ohms)': [cell1_R1, cell2_R1],
                                        'R2 (Ohms)': [cell1_R2, cell2_R2],
                                        'Capacitance (F)': [cell1_C, cell2_C],
                                        })
                circuit_results = pd.concat([circuit_results, results], ignore_index=True)
        return circuit_results

    def r_squared_calc(self, y, y_fit):
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

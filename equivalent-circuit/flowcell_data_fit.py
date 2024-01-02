from textwrap import wrap
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from flowcalc import FlowCalculator
from circuit_models import CircuitModel

# Constants for flow cell experiments
# cell area with 13 mm diameter cell [m^2]
cell_area = (13 / 1000) ** 2 * np.pi / 4
# fit data at specified height delta
deltah = 0


def r_squared_calc(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def get_experiment_data(filepath):
    #raw_file = askopenfilename()
    raw_file = filepath
    if ".comments" in raw_file:
        raw_file.replace(".comments", "")
    file_name = raw_file.split('/')[-1]
    print(file_name)
    flow_calculator = FlowCalculator(file_path=raw_file)
    raw_data = flow_calculator.raw_data

    # filter for signal and deltah
    # exp_df = raw_data.loc[((raw_data['signal'] != 0) & (raw_data['deltah'] == deltah) & (raw_data['cyc'] > 1))]
    exp_df = raw_data.loc[((raw_data['signal'] != 0) & (raw_data['deltah'] == deltah))]
    exp_df = exp_df.copy()

    # mean flow at zero appv
    passive_flow_1 = raw_data.loc[((raw_data["appv"] == 0) & (raw_data['deltah'] == deltah))]["flow1"].mean()
    passive_flow_2 = raw_data.loc[((raw_data["appv"] == 0) & (raw_data['deltah'] == deltah))]["flow2"].mean()
    # print(f'Passive flow: cell 1 = {round(passive_flow_1,3)}; cell 2 = {round(passive_flow_2,3)}')

    # construct relative time (seconds) column
    exp_df["rel time"] = exp_df["time"] - exp_df["time"].iloc[0]
    exp_df["rel time"] = exp_df["rel time"].dt.total_seconds()

    exp_voltage = exp_df['appv'].to_numpy()
    exp_time = exp_df['rel time'].to_numpy()
    exp_cur1 = exp_df['cur1'].to_numpy()
    exp_cur2 = exp_df['cur2'].to_numpy()
    exp_flow1 = exp_df['flow1'].to_numpy()
    exp_flow2 = exp_df['flow2'].to_numpy()

    xdata = pd.DataFrame({'time': exp_time, 'appv': exp_voltage})
    ydata = pd.DataFrame({'cur1': exp_cur1, 'cur2': exp_cur2, 'flow1': exp_flow1, "flow2": exp_flow2})
    return file_name, xdata, ydata


# noinspection PyTupleAssignmentBalance
def fit_exp_flow_data(file_name, xdata, ydata):
    # Parsing input data
    exp_time = xdata['time']
    exp_voltage = xdata['appv']

    # Convert specific flow back to total flow (L/h) for fitting by multiplying by cell area
    exp_flow1 = np.multiply(ydata['flow1'], cell_area)
    exp_flow2 = np.multiply(ydata['flow2'], cell_area)

    # import circuit model class
    circuit_model = CircuitModel(time=exp_time)

    # Set up graphs
    fig, (ax1, ax2) = plt.subplots(figsize=(6.5, 5), nrows=2, ncols=1, layout="constrained")
    ax1.set_title(f"{file_name}\nflow cell 1", loc="left")
    ax2.set_title("flow cell 2", loc="left")

    for ax, y in {ax1: exp_flow1, ax2: exp_flow2}.items():
        # Use non-linear least squares to fit a function f to data
        # assumes ydata = f(xdata, *params) + eps
        # Plotting simulated response with fitted paramater from simple RC model
        popt1, pcov1 = curve_fit(f=circuit_model.circuit_RC,
                                 xdata=xdata,
                                 ydata=y,
                                 p0=[1000, 0.1],
                                 bounds=(0, [np.inf, np.inf]))
        RC_fit_R, RC_fit_C = popt1[0], popt1[1]
        print(f"RC Fitted results: R = {round(RC_fit_R, 2)} Ohms, C = {round(RC_fit_C, 2)}F")
        sim1_flow = circuit_model.circuit_RC(xdata=xdata, resistance=RC_fit_R, capacitance=RC_fit_C)
        sim1_fit = r_squared_calc(y, sim1_flow)
        print(r"RC Model R^2=", round(sim1_fit, 3))
        specific_sim_f1 = np.divide(sim1_flow, cell_area)

        popt2, pcov2 = curve_fit(f=circuit_model.circuit_RRC_diode,
                                 xdata=xdata,
                                 ydata=y,
                                 p0=[100, 100, 100, 0.1],
                                 bounds=(0, [np.inf, np.inf, np.inf, np.inf]))
        RRC_fit_R1, RRC_fit_R2, RRC_fit_R3, RRC_fit_C = popt2[0], popt2[1], popt2[2], popt2[3]
        print(f"Diode Model Fitted results: R1 = {round(RRC_fit_R1, 2)} Ohms, "
              f"R2 = {round(RRC_fit_R2, 2)} Ohms, "
              f"C = {round(RRC_fit_C, 2)}F")
        sim2_flow = circuit_model.circuit_RRC_diode(xdata=xdata, R1=RRC_fit_R1, R2=RRC_fit_R2, R3=RRC_fit_R3, C=RRC_fit_C)
        sim2_fit = r_squared_calc(y, sim2_flow)
        print(r"Diode Model R^2=", round(sim2_fit, 3))
        specific_sim_f2 = np.divide(sim2_flow, cell_area)

        # =========================== Plotting ==========================================================
        ax.plot(exp_time, exp_voltage, 'r--', alpha=0.5, label="appv")
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 600)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('Voltage [V]')

        secondary_ax = ax.twinx()
        secondary_ax.plot(exp_time, np.divide(y, cell_area), '-', label="flow")
        secondary_ax.plot(exp_time, specific_sim_f1, '.', markersize=2, label="RC model")
        secondary_ax.plot(exp_time, specific_sim_f2, '.', markersize=2, label="Diode model")
        secondary_ax.set_ylabel(r'Flow [L/h/$m^2$]')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = secondary_ax.get_legend_handles_labels()
        secondary_ax.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.show()


# noinspection PyTupleAssignmentBalance
def fit_exp_current_data(file_name, xdata, ydata):
    # set up empty dictionary for storing fitting results
    fit_results = {'Experiment': [file_name, file_name],
                   'Flow Cell': [file_name.split('_')[-1][0], file_name.split('_')[-1][1]],
                   'R1': [], 'R2': [], 'R3': [], 'C': [], 'Model Fit - R^2': []}

    # Parsing input data
    exp_time = xdata.time
    exp_voltage = xdata.appv

    # Convert specific current back to total current (A) for fitting by multiplying by cell area
    exp_cur1 = np.multiply(ydata['cur1'], cell_area)
    exp_cur2 = np.multiply(ydata['cur2'], cell_area)

    # import circuit model class
    circuit_model = CircuitModel(time=exp_time)
    fit_circuit = circuit_model.circuit_RRC_diode
    initial_guesses = [100, 100, 10, 1]  # for R1, R2, R3, and C


    # Set up graphs
    fig, (ax1, ax2) = plt.subplots(figsize=(6.5, 5), nrows=2, ncols=1, layout="constrained")
    file_name = "\n".join(wrap(file_name, 60))
    ax1.set_title(f"{file_name}\nflow cell 1", loc="left")
    ax2.set_title("flow cell 2", loc="left")

    for ax, y in {ax1: exp_cur1, ax2: exp_cur2}.items():
        # Use non-linear least squares to fit a function f to data
        # assumes ydata = f(xdata, *params) + eps
        # Plotting simulated response with fitted paramater from simple RC model
        popt1, pcov1 = curve_fit(f=fit_circuit,
                                 xdata=xdata,
                                 ydata=y,
                                 p0=initial_guesses,
                                 bounds=(0, [np.inf]*len(initial_guesses)))
        RC_fit_R1, RC_fit_R2, RC_fit_R3, RC_fit_C = popt1[0], popt1[1], popt1[2], popt1[3]
        print(f"RRC-Diode Fitted results: R1 = {round(RC_fit_R1, 2)} Ohms, "
              f"R2 = {round(RC_fit_R2, 2)} Ohms, "
              f"R3 = {round(RC_fit_R3, 2)} Ohms, "
              f"C = {round(RC_fit_C, 2)}F")

        sim1_cur = fit_circuit(xdata=xdata, R1=RC_fit_R1, R2=RC_fit_R2, R3=RC_fit_R3, C=RC_fit_C)
        sim1_fit = r_squared_calc(y, sim1_cur)
        print(r"RRC-Diode Model R^2=", round(sim1_fit, 3))

        fit_results['R1'].append(round(RC_fit_R1, 2))
        fit_results['R2'].append(round(RC_fit_R2, 2))
        fit_results['R3'].append(round(RC_fit_R3, 2))
        fit_results['C'].append(round(RC_fit_C, 2))
        fit_results['Model Fit - R^2'].append(round(sim1_fit, 3))

        # =========================== Plotting ===========================================================
        specific_sim_c1 = np.divide(sim1_cur, cell_area)
        ax.plot(exp_time, exp_voltage, 'r--', alpha=0.5, label="appv")
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 600)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('Voltage [V]')

        secondary_ax = ax.twinx()
        secondary_ax.plot(exp_time, np.divide(y, cell_area), '-', label="current")
        secondary_ax.plot(exp_time, specific_sim_c1, '.', markersize=2, label="model")
        secondary_ax.set_ylabel(r'Current [A/$m^2$]')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = secondary_ax.get_legend_handles_labels()
        secondary_ax.legend(lines + lines2, labels + labels2, loc='upper right')

    # plt.savefig(f"{file_name} circuit fitting.png")
    plt.show()
    plt.close()
    return fit_results

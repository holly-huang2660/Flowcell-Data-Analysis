from os import listdir
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory
from textwrap import wrap
import numpy as np
import pandas as pd
# from circuit_models import voltage_source, circuit_RC, circuit_RRC, circuit_RRCRC, plot_simulated_response
from circuit_models import CircuitModel
from flowcell_data_fit import *
import matplotlib.pyplot as plt


# TODO: Make a UI to select simulation or data fitting


# # =================== Simulated Circuit =====================================
def model_circuit_plot():
    # Parameters
    wave = ['square', 'sine', 'sine']
    amplitude = [0.5, 0.8, 1]
    pulse_length = 60
    R1, R2, Cap = 150, 1000, 0.1

    sim_time = np.arange(0, pulse_length * 9, 1)  # set up time sequence
    sim_model = CircuitModel(time=sim_time)  # Initiate the class

    # # Create an arbitrary voltage input with scipy signal library
    for i in range(len(wave)):
        sim_model.voltage_source(waveform=wave[i], appv=amplitude[i], duration=pulse_length, dc_bias=0,
                                 phase_shift=np.pi)
        sim_model.circuit_RRC(xdata=sim_model.xdata, R1=R1, R2=R2, C=Cap)
        plot_title = f'R(RC) Circuit (R1={R1}Ω, R2={R2}Ω, C={Cap}F)'

        # Plot results
        sim_model.plot_simulated_response(title=plot_title, auto_scale=False, cur_max=0.01, v_max=1)

        status = []
        for v in sim_model.appv:
            if v > 0:
                status.append('+v')
            elif v < 0:
                status.append('-v')
            else:
                status.append('0v')

        # calculate average current
        q_data = pd.DataFrame(
            {'time': sim_time, 'appv': sim_model.appv, 'current': sim_model.current, 'status': status})
        q_data['power'] = q_data['appv'] * q_data['current']

        q_data = q_data.loc[(q_data['time'] >= pulse_length * 3)]
        cur_df = q_data.groupby(["status"], as_index=False)["current"].mean()
        print(cur_df)
        cyc = q_data['time'].count() / (pulse_length * 2)
        cycle_pwr = np.trapz(x=q_data['time'], y=q_data['power']) / cyc
        print(f'number of cycles={cyc}, cycle power = {round(cycle_pwr, 3)} W')


# # =================== Data exploration: flow vs current relationships ===============================
def flow_current_plot():
    data_loc = r"C:\Users\hhuang\PycharmProjects\Osmotex\flowcell-data-analysis\data comparison"
    file_list = askopenfilenames(initialdir=data_loc)
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")

    if len(file_list) > 0:
        for file in file_list:
            exp_name, xdata, ydata = get_experiment_data(filepath=file, exp_deltah=0, exp_signal=11)
            ax.plot('cur1', 'flow1', '.', data=ydata, label=exp_name[-2], alpha=0.2)
            ax.plot('cur2', 'flow2', '.', data=ydata, label=exp_name[-1], alpha=0.2)

        title_text = "\n".join(
            wrap(exp_name.replace("PO", "").replace("WY", "").replace("RK", "").replace("GB", ""), 40))
        ax.set_title(title_text)
        ax.set_ylabel("Flow [L/h/m^2]")
        ax.set_xlabel("Current [A/m^2]")

        # force symmetrical axis
        xmax = max([abs(n) for n in ax.get_xlim()])
        ymax = max([abs(n) for n in ax.get_ylim()])
        ax.set_xlim(xmax * -1, xmax)
        ax.set_ylim(ymax * -1, ymax)
        plt.grid()
        plt.legend()
        plt.show()

# =================== Fitting experiment data ===============================
def exp_data_fitting():
    file = askopenfilename()
    exp_name, xdata, ydata = get_experiment_data(filepath=file, exp_deltah=0, exp_signal=11)
    fit_exp_flow_data(exp_name, xdata, ydata)

    # # Loop through all files in a folder
    # folder_path = askdirectory()
    # # ignore any files with extensions
    # file_list = [f for f in listdir(f"{folder_path}") if "." not in f]
    # cir_list = []
    #
    # for file in file_list:
    #     file_path = f"{folder_path}/{file}"
    #     exp_name, xdata, ydata = get_experiment_data(filepath=file_path)
    #     cir_result = pd.DataFrame(fit_exp_current_data(exp_name, xdata, ydata))
    #
    #     # remove flow cell designation from file name
    #     exp_name = file.replace("_WY", "").replace("_PO", "")
    #     cir_result.insert(0, "file", exp_name)
    #
    #     cir_list.append(cir_result)
    #
    # result_df = pd.concat(cir_list, ignore_index=True)
    # result_df.to_csv("RRC_diode_fitting_results.csv")
    #
    # # ============== Save fitted model parameters to csv file ===================
    # df = pd.DataFrame(results)
    # print(df)
    #
    # save_file = input("Save results? Type y or n:\n")
    # if save_file == 'y':
    #     df.to_csv('circuit fitting results.csv', mode='a', header=False)


exp_data_fitting()
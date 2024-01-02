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


# =================== Simulated Circuit =====================================
sim_time = np.arange(0, 360, 0.01)  # set up time sequence
# Initiate the class
sim_model = CircuitModel(time=sim_time)

# # Create an arbitrary voltage input with scipy signal library
sim_model.voltage_source(waveform='square', appv=0.5, duration=60, dc_bias=0, phase_shift=0)
# # Alternatively, use Fourier approximation for square and sawtooth waveform
# sim_model.voltage_source_fourier_approx(waveform="square", appv=0.5, duration=60, dc_bias=0, fourier_n=2)

# Select circuit model to use
# Adding in a simple diode in series with the circuit
# if Vapp < threshold_voltage, no current flows through it.
# If Vapp > threshold_voltage, the diode acts as a wire (no resistance)
# sim_model.circuit_RC_diode(xdata=sim_model.xdata, R1=10, R2=10, C=1, threshold_voltage=0)
# plot_title = 'Square Wave RC Diode Circuit (R1=10Ω, R2=10Ω, C=1F)'

sim_model.circuit_RRC_diode(xdata=sim_model.xdata, R1=10, R2=100, R3=20, C=1, threshold_voltage=0)
plot_title = 'Square Wave R(RC) Diode Circuit (R1=10Ω, R2=100Ω, R3=20Ω, C=1F)'

# Plot results
sim_model.plot_simulated_response(title=plot_title)


# # Calculate columbic efficiency for one cycle
# q_data = pd.DataFrame({'time': sim_time, 'appv': sim_model.appv, 'current': sim_model.current})
# q_data = q_data.loc[(q_data['time'] >= 120) & (q_data['time'] <= 240)]
# plt.plot('time', 'current', data=q_data)
# plt.plot('time', 'appv', data=q_data)
#
# pos_current = np.where(q_data['current'] > 0, q_data['current'], 0)
# neg_current = np.where(q_data['current'] < 0, q_data['current'], 0)
# q_in = np.trapz(y=pos_current, x=q_data['time'])
# q_out = np.trapz(y=neg_current, x=q_data['time'])
# print(f"q_in = {round(q_in, 3)}, q_out = {round(q_out, 3)}, coulumbic efficiency = {round(abs(q_out / q_in), 3)}")
# plt.show()


# # =================== Data exploration: flow vs current relationships ===============================
# # data_loc = (r"C:\Users\hhuang\Myant Research Center of Canada Inc\MRCC-MyantCH - "
# #             r"Documents\Materials\EXPERIMENTAL\13mmFlowCellTest_Data")
# data_loc = r"C:\Users\hhuang\OneDrive - Myant Research Center of Canada Inc\PycharmProjects\Osmotex-Flowcell-Data-Analysis\raw data"
# file_list = askopenfilenames(initialdir=data_loc)
# fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
#
# if len(file_list) > 0:
#     for file in file_list:
#         exp_name, xdata, ydata = get_experiment_data(filepath=file)
#         ax.plot('cur1', 'flow1', '.', data=ydata, label=exp_name[-2], alpha=0.2)
#         ax.plot('cur2', 'flow2', '.', data=ydata, label=exp_name[-1], alpha=0.2)
#
#     title_text = "\n".join(wrap(exp_name.replace("_PO", "").replace("_WY", ""), 40))
#     ax.set_title(title_text)
#     ax.set_ylabel("Flow [L/h/m^2]")
#     ax.set_xlabel("Current [A/m^2]")
#
#     # force symmetrical axis
#     xmax = max([abs(n) for n in ax.get_xlim()])
#     ymax = max([abs(n) for n in ax.get_ylim()])
#     ax.set_xlim(xmax*-1, xmax)
#     ax.set_ylim(ymax*-1, ymax)
#     plt.grid()
#     plt.legend()
#     plt.savefig(f"./figures/coated sample/{exp_name} flow vs current.png")
#     #plt.show()


# =================== Fitting experiment data ===============================
# file = askopenfilename()
# exp_name, xdata, ydata = get_experiment_data(filepath=file)
# results = fit_exp_current_data(exp_name, xdata, ydata)

# Loop through all files in a folder
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



# # ============== Save fitted model parameters to csv file ===================
# df = pd.DataFrame(results)
# print(df)
#
# save_file = input("Save results? Type y or n:\n")
# if save_file == 'y':
#     df.to_csv('circuit fitting results.csv', mode='a', header=False)

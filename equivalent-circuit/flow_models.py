from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import j1, j0
import matplotlib.pyplot as plt
from circuit_models import CircuitModel
from flowcalc import FlowCalculator

A = np.pi / 4 * (13 / 1000) ** 2  # cross-sectional area (m^2)


class FlowModel:
    def __init__(self, time=np.arange(0, 420, 0.01), mode='simulate'):
        if mode == 'simulate':
            # if no time series was provided, default is 0 to 600 s with 10 ms step
            self.time = time
            circuit_model = CircuitModel(time=time)
            self.xdata = circuit_model.voltage_source(waveform='square', appv=-0.5)
            self.voltage_source = self.xdata['appv']
            self.current = circuit_model.circuit_RRC_diode(xdata=circuit_model.xdata, R1=10, R2=100, R3=20)
            self.current = np.divide(self.current, A)
            self.simulate_flow_plot()

        elif mode == 'fitting':
            file_name, xdata, ydata = self.get_experiment_data()
            self.time = xdata['time']
            self.voltage_source = xdata['appv']
            self.current = ydata['cur1']
            self.voltage = ydata['v1']
            self.exp_Q = ydata['flow1']
            self.fitted_flow_plot()

    def get_experiment_data(self):
        raw_file = askopenfilename()
        if ".comments" in raw_file:
            raw_file.replace(".comments", "")
        file_name = raw_file.split('/')[-1]
        print(file_name)
        flow_calculator = FlowCalculator(file_path=raw_file)
        raw_data = flow_calculator.raw_data

        # filter for signal and deltah
        # exp_df = raw_data.loc[((raw_data['signal'] != 0) & (raw_data['deltah'] == deltah) & (raw_data['cyc'] > 1))]
        exp_df = raw_data.loc[((raw_data['signal'] == 1) & (raw_data['deltah'] == 0))]
        exp_df = exp_df.copy()

        # construct relative time (seconds) column
        exp_df["rel time"] = exp_df["time"] - exp_df["time"].iloc[0]
        exp_df["rel time"] = exp_df["rel time"].dt.total_seconds()

        exp_voltage = exp_df['appv'].to_numpy()
        exp_time = exp_df['rel time'].to_numpy()
        exp_cur1 = exp_df['cur1'].to_numpy()
        exp_cur2 = exp_df['cur2'].to_numpy()
        exp_flow1 = exp_df['flow1'].to_numpy()
        exp_flow2 = exp_df['flow2'].to_numpy()

        cell_v1 = exp_df['vcur1'].to_numpy()
        cell_v2 = exp_df['vcur2'].to_numpy()

        xdata = pd.DataFrame({'time': exp_time, 'appv': exp_voltage})
        ydata = pd.DataFrame({'v1': cell_v1, 'v2': cell_v2,
                              'cur1': exp_cur1, 'cur2': exp_cur2,
                              'flow1': exp_flow1, "flow2": exp_flow2})
        return file_name, xdata, ydata

    def parallel_tubes_model(self, xdata, a=100e-9, psi=0.5, tau=1):
        # # ===== S. Yao, J.G. Santiago / Journal of Colloid and Interface Science 268 (2003) 133–142 ======

        # Fitting parameters
        a = a  # pore radius (m)
        psi = psi  # porosity
        tau = tau  # tortuosity

        # Membrane characteristics
        L = 100 * 10 ** -6  # membrane thickness (m)

        # Voltage input
        # effective applied axial voltage, ideal case assumption = Vapp
        current = np.multiply(xdata['current'], A)
        Veff = current  # # applied voltage (V), V=IR

        # External pressure
        # calculate pressure as the hydraulic head P = rho * g * H
        # P = 0  # pressure (Pa)
        deltah = 0  # hydraulic head (m)
        P = 1000 * 9.81 * deltah  # hydraulic pressure (Pa)
        # print(f"At height delta = {deltah} m, P = {P} Pa")

        # Solution characteristics
        mu = 0.001  # solution viscosity (Pa s)
        eta = 80 * (8.854 * 10 ** -12)  # eta_r * eta_0  fluid permittivity (C V^-1 m^-1)
        zeta = -0.03  # zeta potential (V)
        n = 30 * 10 ** -3  # Ionic strength / bulk concentration (M)
        T = 298  # Temperature of the liquid (K)

        # Constants
        N_A = 6.022 * 10 ** 23  # Avogadro constant (mol^-1)
        kb = 1.381 * 10 ** -23  # Boltzmann constant (J/K)
        e = -1.602 * 10 ** -19  # elementary charge (Coulombs)

        # Calculate Debye length (m) for 1:1 electrolyte concentration
        debye_length = ((eta * kb * T) / (2000 * N_A * e ** 2 * n)) ** 0.5
        # print(f"For {n} M solution, Debye length = {round(debye_length * 10 ** 9, 3)} nm")

        # Calculate correction factor for the finite EDL effect of flow rate
        # Debye-Huckel approximation from solving the Poisson-Boltzmann equation for electrical potential
        a_star = a / debye_length
        f = 1 - (2 * j1(a_star)) / (a_star * j0(a_star))

        # EO pumping in porous membrane modeled by treating the membrane (Q = flow rate, m^3 / s)
        # as a parellel array of cylindrival pores of uniform pore size
        # Assuming EO pump against the direction of pressure flow
        Q = (psi / tau) * ((P * A * a ** 2) / (8 * mu * L) - (eta * zeta * A * Veff) / (mu * L) * f)

        # print(f"Specific flow rate = {round(Q * unit_conversion / A, 3)} L/h/m^2")
        unit_conversion = 3.6 * 10 ** 6  # convert unit from m^3/s to L/hr
        Q = Q * unit_conversion / A
        return Q

    def fitted_flow_plot(self):
        xdata = pd.DataFrame({'time': self.time, 'current': self.current})

        popt1, pcov1 = curve_fit(f=self.parallel_tubes_model,
                                 xdata=xdata,
                                 ydata=self.exp_Q,
                                 p0=[1e-8, 0.5, 1],
                                 bounds=([0, 0, 1], [1, 1, 10])
                                 )
        fitted_a, fitted_psi, fitted_tau = popt1[0], popt1[1], popt1[2]
        print(f'radius={round(fitted_a*1e8,3)}e-8, '
              f'porosity={round(fitted_psi,3)}, '
              f'tortuosity={round(fitted_tau,3)}')
        Q = self.parallel_tubes_model(xdata, a=fitted_a, psi=fitted_psi, tau=fitted_tau)

        fig, ax1 = plt.subplots(layout="constrained")
        ax1.plot(self.time, self.voltage_source)
        ax1.set_ylabel("Voltage [V]")
        ax2 = ax1.twinx()

        ax2.plot(self.time, Q, 'r.', markersize=2, label='predicted')
        ax2.plot(self.time, self.exp_Q, 'b.', markersize=2, label='actual')
        ax2.set_ylabel("Flow [L/h/m^2]")
        ax1.grid(visible=True)

        # force symmetrical y-axis
        for ax in [ax1, ax2]:
            ymax = max([abs(n) for n in ax.get_ylim()])
            ax.set_ylim(ymax * -1, ymax)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.show()
        plt.close()

    def simulate_flow_plot(self):
        xdata = pd.DataFrame({'time': self.time, 'current': self.current})
        Q = self.parallel_tubes_model(xdata=xdata)

        fig, ax1 = plt.subplots(layout="constrained")
        ax1.plot(self.time, self.voltage_source)
        ax1.set_ylabel("Voltage [V]")
        ax2 = ax1.twinx()

        ax2.plot(self.time, Q, 'r.', markersize=2, label='predicted')
        ax2.set_ylabel("Flow [L/h/m^2]")
        ax1.grid(visible=True)

        # force symmetrical y-axis
        for ax in [ax1, ax2]:
            ymax = max([abs(n) for n in ax.get_ylim()])
            ax.set_ylim(ymax * -1, ymax)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.show()
        plt.close()


example_flow = FlowModel(mode='simulate')

from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


class CircuitModel:
    def __init__(self, time):

        self.time = time
        # if no time series was provided, default is 0 to 240 s with 10 ms step
        if self.time is None:
            self.time = np.arange(0, 240, 0.01)

        self.appv = []
        self.current = []
        self.xdata = []
        self.Vc = []  # capacitor voltage

    def circuit_RC(self, xdata, resistance=10.0, capacitance=1.0):
        # xdata : pandas dataframe with time (in seconds) and appv column
        time = xdata['time']
        voltage_waveform = xdata['appv']
        dt = time[1] - time[0]

        # Set up circuit elements
        R = resistance  # resistor in ohms (default = 10 ohms)
        C = capacitance  # capacitor in F (default = 1 F)
        Q = 0  # initial charge of the capacitor assumed to be zero

        # list for plotting
        RC_current = []
        Vc = []

        # loop through time period
        for index in range(len(time)):
            Emf = voltage_waveform[index]
            dQ = dt / R * (Emf - Q / C)
            Q += dQ
            RC_current.append(dQ / dt)
            self.Vc.append(Q / C)

        self.current = RC_current
        return RC_current

    def circuit_RRC(self, xdata, R1=10, R2=20, C=1):
        # xdata : pandas dataframe with time (in seconds) and appv column
        time = xdata['time']
        voltage_waveform = xdata['appv']
        dt = time[1] - time[0]

        # Set up initial conditions
        R1 = R1
        R2 = R2
        C = C
        Q = 0
        V_PZC = 0

        # list for plotting
        RRC_current = []

        for index in range(len(time)):
            Emf = voltage_waveform[index]

            # Matrix operation I = R^-1 V
            R_matrix = np.array([[1, -1, -1],
                                 [R1, R2, 0],
                                 [0, R2, 0]])
            V_matrix = np.array([0, Emf, Q / C])

            R_inv = np.linalg.inv(R_matrix)
            I_matrix = np.matmul(R_inv, V_matrix)
            I1, I2, I3 = I_matrix[0], I_matrix[1], I_matrix[2]
            dQ = I3 * dt
            Q += dQ

            RRC_current.append(I1)
        self.current = RRC_current
        return RRC_current

    def circuit_RRCRC(self, xdata, R1=10, R2=20, R3=50, C1=1, C2=2):
        # xdata : pandas dataframe with time (in seconds) and appv column
        time = xdata['time']
        voltage_waveform = xdata['appv']
        dt = time[1] - time[0]

        # Set up initial conditions
        R1 = R1
        R2 = R2
        R3 = R3
        C1 = C1
        C2 = C2
        Q1 = 0
        Q2 = 0

        # list for plotting
        randle_current = []

        for index in range(len(time)):
            Emf = voltage_waveform[index]

            # Matrix operation I = R^-1 V
            R_matrix = np.array([[1, -1, 0, -1, 0],
                                 [1, 0, -1, 0, -1],
                                 [R1, R2, R3, 0, 0],
                                 [0, R2, 0, 0, 0],
                                 [0, 0, R3, 0, 0]
                                 ])
            V_matrix = np.array([0, 0, Emf, Q1 / C1, Q2 / C2])

            R_inv = np.linalg.inv(R_matrix)
            I_matrix = np.matmul(R_inv, V_matrix)
            I1, I2, I3, I4, I5 = I_matrix[0], I_matrix[1], I_matrix[2], I_matrix[3], I_matrix[4]

            dQ1 = I4 * dt
            Q1 += dQ1
            dQ2 = I5 * dt
            Q2 += dQ2

            randle_current.append(I1)
        self.current = randle_current
        return randle_current

    def circuit_RC_diode(self, xdata, R1=10.0, R2=20.0, C=1.0, threshold_voltage=0):
        # xdata : pandas dataframe with time (in seconds) and appv column
        time = xdata['time']
        voltage_waveform = xdata['appv']
        dt = time[1] - time[0]

        # Set up circuit elements
        R1 = R1  # resistor in ohms (default = 10 ohms)
        R2 = R2  # resistor in ohms (default = 20 ohms)
        C = C  # capacitor in F (default = 1 F)
        Q = 0  # initial charge of the capacitor assumed to be zero
        Vk = threshold_voltage  # threshold voltage of the diode element

        # list for plotting
        RC_diode_current = np.zeros(len(time))

        # loop through time period
        for index in range(len(time)):
            Emf = voltage_waveform[index]

            if Emf < Vk:
                R = R1
            else:
                R = (R1 + R2)

            dQ = dt / R * (Emf - Q / C)
            Q += dQ
            RC_diode_current[index] = dQ / dt

        self.current = RC_diode_current
        return RC_diode_current

    def circuit_RRC_diode(self, xdata, R1=10, R2=20, R3=5, C=1, threshold_voltage=0):
        # xdata : pandas dataframe with time (in seconds) and appv column
        time = xdata['time']
        voltage_waveform = xdata['appv']
        dt = time[1] - time[0]

        # Set up initial conditions
        R1 = R1
        R2 = R2
        R3 = R3
        C = C
        Q = 0
        Vk = threshold_voltage
        V_PZC = 0

        # list for plotting
        RRC_diode_current = []
        V1 = []

        for index in range(len(time)):
            Emf = voltage_waveform[index]

            if Emf < Vk:
                R = R1
            else:
                R = (R1 + R3)

            # Matrix operation I = R^-1 V
            R_matrix = np.array([[1, -1, -1],
                                 [R, R2, 0],
                                 [0, R2, 0]])
            V_matrix = np.array([0, Emf, Q / C])

            R_inv = np.linalg.inv(R_matrix)
            I_matrix = np.matmul(R_inv, V_matrix)
            I1, I2, I3 = I_matrix[0], I_matrix[1], I_matrix[2]
            dQ = I3 * dt
            Q += dQ

            RRC_diode_current.append(I1)

        self.current = RRC_diode_current
        return RRC_diode_current

    def voltage_source(self, waveform: str, appv=0.5, duration=60, dc_bias=0, phase_shift=0):
        # convert duration to frequency
        frequency = np.pi / duration

        # Update class attribute (voltage and xdata)
        if "sine" in waveform:
            self.appv = np.sin(frequency * self.time + phase_shift) * appv + dc_bias
        elif "sawtooth" in waveform:
            self.appv = scipy.signal.sawtooth(frequency * self.time + phase_shift, 0.5) * appv + dc_bias
        else:
            self.appv = scipy.signal.square(frequency * self.time + phase_shift) * appv + dc_bias

        self.xdata = pd.DataFrame({'time': self.time, 'appv': self.appv})

        # Set first phase of voltage pulse to zero to match experiment conditions
        self.xdata.loc[(self.xdata['time'] < duration), 'appv'] = 0
        return self.xdata

    def voltage_source_fourier_approx(self, waveform: str, appv=0.5, duration=60, dc_bias=0, fourier_n=2):
        # convert duration to frequency
        frequency = np.pi / duration
        # Number of modes for fourier series
        n = fourier_n

        # Update class attribute (voltage and xdata)
        if "sine" in waveform:
            # sinusoidal
            self.appv = dc_bias + appv * np.sin(frequency * self.time)
        elif "sawtooth" in waveform:
            # Triangle wave Fourier series
            f_series = 0
            for n in np.arange(1, n + 1, 1):
                f_series += (-1) ** (n - 1) * np.sin((2 * n - 1) * frequency * self.time) / (2 * n - 1) ** 2
            self.appv = dc_bias + 8 * appv / np.pi ** 2 * f_series
        else:
            # Square wave Fourier series
            f_series = 0
            for n in np.arange(1, n + 1, 1):
                f_series += np.sin((2 * n - 1) * frequency * self.time) / (2 * n - 1)
            self.appv = dc_bias + 4 * appv / np.pi * f_series

        self.xdata = pd.DataFrame({'time': self.time, 'appv': self.appv})

    def plot_simulated_response(self, title="Equivalent circuit simulation", auto_scale=True, cur_max=0.01, v_max=1):
        # Plotting resultant current and applied voltage
        fig, ax1 = plt.subplots(figsize=(6, 4), layout="constrained")
        ax1.grid()
        ax1.plot('time', 'appv', 'r--', data=self.xdata, label='Voltage')
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('Voltage [V]')

        # force symmetrical axis
        ymax = max([abs(n) for n in ax1.get_ylim()])
        ax1.set_ylim(ymax * -1, ymax)
        ax1.set_xlim(0, self.time[-1])

        title_text = "\n".join(wrap(title, 40))

        ax1.set_title(title_text)

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.current, 'b--', label="Current")
        ax2.set_ylabel(r'Current [A]')

        # force symmetrical axis
        if auto_scale:
            ymax1 = max([abs(n) for n in ax1.get_ylim()])
            ymax2 = max([abs(n) for n in ax2.get_ylim()])
        else:
            ymax1 = v_max
            ymax2 = cur_max

        ax1.set_ylim(ymax1 * -1, ymax1)
        ax2.set_ylim(ymax2 * -1, ymax2)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.show()


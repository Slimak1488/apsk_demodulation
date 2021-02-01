import numpy as np
from matplotlib import pyplot as plt
from sigproc import Signal
from scipy.signal import butter, lfilter, freqs, firwin, fftconvolve
from scipy.fftpack import fft, ifft, fftshift


#################################
class Apsk:

    #################################
    def __init__(self, 
            modulation = {'0':(0,0), '1':(1,0)},
            baud_rate = 10,
            bits_per_baud = 1,
            carrier_freq = 100):
        '''
        Create a modulator using OOK by default
        '''
        self.modulation    = modulation
        self.baud_rate     = baud_rate
        self.bits_per_baud = bits_per_baud
        self.carrier_freq  = carrier_freq

        self.constellation = [(a*np.cos(p/180.0*np.pi), a*np.sin(p/180.0*np.pi), t) for t, (a, p) in modulation.items()]

    #################################
    def modulate_signal(self, data):
        '''
        Generate signal corresponding to the current modulation scheme to
        represent given binary string, data.
        '''
        def create_func(data):
            slot_data = []
            for i in range(0, len(data), self.bits_per_baud):
                slot_data.append(self.modulation[data[i:i+self.bits_per_baud]])

            def timefunc(t):
                slot = int(t*self.baud_rate)
                start = float(slot)/self.baud_rate
                offset = t #- start
                amplitude, phase = slot_data[slot]
                freq = 2 * np.pi * self.carrier_freq * offset
                freq = freq if freq < 2 * np.pi else freq - 2 * np.pi
                return amplitude*np.sin(freq + phase/180.0 * np.pi)

            return timefunc

        func = create_func(data)
        duration = float(len(data))/(self.baud_rate*self.bits_per_baud)
        s = Signal(duration=duration, carrier_freq=self.carrier_freq, func=func)
        return s

    #################################
    def lowpass_filter(self, data, order=5, filter='zero'):

        def butter_lowpass(order, cutoff, fs):
            nyq_fs = 0.5*fs
            b, a = butter(order, 0.5, btype='low')
            return b, a

        if filter == 'butter':
            cutoff = float(self.carrier_freq)
            b, a = butter_lowpass(order, cutoff, 1000)
            y = lfilter(b, a, data)
            return y

        if filter == 'zero':
            f_data = fft(data)
            f_data[100:] = 0
            data = ifft(f_data)
            return data


    #################################
    def identify_symbol(self, symbol):
        min_dist = 0

    #################################
    def demodulate_signal(self, signal):
        sig = signal.signal
        n = len(sig)
        I = np.zeros(n, dtype=float)
        Q = np.zeros(n, dtype=float)
        for i in range(n):
            offset = float(i)/signal.sampling_rate
            freq = 2 * np.pi * self.carrier_freq * offset
            freq = freq if freq < 2 * np.pi else freq - 2 * np.pi
            I[i] = 2*sig[i]*np.cos(freq)
            Q[i] = 2*sig[i]*np.sin(freq)
        n_taps = 640 * 2
        h = firwin(n_taps // 8, float(self.carrier_freq / 160), nyq=float(self.carrier_freq / 2))
        # I, Q = self.lowpass_filter(I), self.lowpass_filter(Q)
        I, Q = lfilter(h, 1.0, I), lfilter(h, 1.0, Q)
        signal.set_IQ_components(I, Q)
        step = int(n/(signal.duration * self.baud_rate))
        c_data = np.arange(step//2, n, step)
        i_value, q_value = I[c_data], Q[c_data]
        return i_value, q_value

    #################################
    def plot_constellation(self, r, s=None, q=None):
        sx, sy, t = zip(*self.constellation)
        plt.clf()
        plt.scatter(sx, sy, s=30)
        plt.scatter(s, q, s=10, c='deeppink')
        plt.axes().set_aspect('equal')
        for x, y, t in self.constellation:
            plt.annotate(t, (x-.04, y-.04), ha='right', va='top')
        plt.axis([-r, r, -r, r])
        plt.axhline(0, color='red')
        plt.axvline(0, color='red')
        plt.grid(True)
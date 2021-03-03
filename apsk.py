import numpy as np
from matplotlib import pyplot as plt
from sigproc import Signal
from scipy.signal import butter, lfilter, freqs, firwin, fftconvolve
from scipy.fftpack import fft, ifft, fftshift
from commpy.filters import rcosfilter

#################################
class Apsk:

    #################################
    def __init__(self, 
            modulation = {'0':(0,0), '1':(1,0)},
            baud_rate = 10,
            bits_per_baud = 1,
            carrier_freq = 100):

        self.modulation    = modulation
        self.baud_rate     = baud_rate
        self.bits_per_baud = bits_per_baud
        self.carrier_freq  = carrier_freq

        self.constellation = [(a*np.cos(p/180.0*np.pi), a*np.sin(p/180.0*np.pi), t) for t, (a, p) in modulation.items()]
        self.cm = []
        self.sI = []
        self.sQ = []

    #################################
    def modulate_signal(self, data, savefile=None, lvl_noise=None, shift_dopler=0):
        '''
        Generate signal corresponding to the current modulation scheme to
        represent given binary string, data.
        '''
        def create_func(data):
            slot_data = []
            shift = [0, 90, 180, 270]
            shift_phase = shift[np.random.randint(0, 4)] / 180.0 * np.pi
            # shift_phase = 0 / 180.0 * np.pi
            print("Начальная фаза:", shift_phase/np.pi * 180)
            for i in range(0, len(data), self.bits_per_baud):
                slot_data.append(self.modulation[data[i:i+self.bits_per_baud]])

            def timefunc(t):
                slot = int(t*self.baud_rate)
                amplitude, phase = slot_data[slot]
                freq = 2 * np.pi * (self.carrier_freq + shift_dopler) * t + shift_phase
                freq = freq if freq < 2 * np.pi else freq - 2 * np.pi
                return amplitude*np.sin(freq + phase/180.0 * np.pi)

            return timefunc

        func = create_func(data)
        duration = float(len(data))/(self.baud_rate*self.bits_per_baud)
        s = Signal(duration=duration, carrier_freq=self.carrier_freq, func=func)

        if lvl_noise is not None:
            s.addNoise(lvl_noise)

        if savefile:
            s.write_wav(savefile)

        return s

    #################################
    def demodulate_signal(self, readfile, preambul):
        m_signal = Signal(carrier_freq=self.carrier_freq)
        m_signal.read_wav(readfile)

        def timefunc(t, sig):
            freq = 2 * np.pi * self.carrier_freq * t
            freq = freq if freq < 2 * np.pi else freq - 2 * np.pi
            I = 2*sig*np.cos(freq)
            Q = 2*sig*np.sin(freq)
            return I, Q

        n = m_signal.get_len_signal()
        I, Q = m_signal.getShiftFreqSignal(timefunc, n)

        def lowpass_filter(I, Q, numtaps, cutoff, N, Ts, Fs, alpha=0.35):

            h = firwin(numtaps, cutoff, nyq=float(self.carrier_freq / 2))
            I, Q = lfilter(h, 1.0, I), lfilter(h, 1.0, Q)

            sPSF = rcosfilter(N, alpha, Ts, Fs)[1]
            I, Q = np.convolve(sPSF, I, mode='same'), np.convolve(sPSF, Q, mode='same')

            return I, Q

        n = m_signal.get_len_signal()
        duration = m_signal.get_duration()

        cutoff = float(self.carrier_freq / 24)
        N = n // self.baud_rate
        simbol_period = 1 / self.baud_rate
        I, Q = lowpass_filter(I, Q, 255, cutoff, N, simbol_period, 0.0000001)

        m_signal.set_IQ_components(I, Q)

        step = int(N/duration)
        c_data = np.arange(step//2, n, step)

        bits = self.component_to_bits(I[c_data], Q[c_data], preambul)
        print(bits[10:])

        return m_signal

    #################################
    def component_to_bits(self, iSamples, qSamples, preambul):
        coef_dempf = 4
        alpha = 0
        bits = []
        shift_phase = 0
        count_bit = 0

        def getChangedPhaseModulation(phi0):
            constellation = self.modulation.copy()
            for symbol in constellation:
                phi = constellation[symbol][1] + phi0
                if phi > 360:
                    phi -= 360
                constellation[symbol] = (constellation[symbol][0], phi)
            return constellation

        def nearest_value(values: list, one: (float, int)) -> (float, int):
            return min(values, key=lambda n: (abs((one[0] - n[0])**2 + (one[1] - n[1])**2), n))

        def get_key(d, val):
            for k, v in d.items():
                if v == val:
                    return k

        prev_constellation = self.modulation.copy()

        for (i, q) in zip(iSamples, qSamples):
            if i > 0:
                curr_phase = (90 - np.arctan(q/i)/np.pi * 180 if np.arctan(q/i)/np.pi * 180 > 0 else 90 - np.arctan(q/i) / np.pi * 180)
            else:
                curr_phase = (270 - np.arctan(q / i) / np.pi * 180 if np.arctan(q / i) / np.pi * 180 > 0 else 270 - np.arctan(q / i) / np.pi * 180)

            #####Восстановление начальной фазы#######
            if count_bit < len(preambul):
                key = preambul[count_bit:count_bit + self.bits_per_baud]
                dif_phase = curr_phase - prev_constellation[key][1]
                if dif_phase < 0:
                    dif_phase = 360 - np.abs(dif_phase)
                shift_phase += np.abs(dif_phase) * self.bits_per_baud / len(preambul)
                count_bit += self.bits_per_baud
            else:
                print("Сдвиг фазы", shift_phase)
                if curr_phase - shift_phase < 0:
                    ph0 = 360 - shift_phase
                    curr_phase = curr_phase + ph0
                else:
                    curr_phase -= shift_phase

                self.setIQSamles(i, q)

            amplitude = np.sqrt(i**2 + q**2)
            curr_constellation = getChangedPhaseModulation(alpha)
            value = nearest_value(curr_constellation.values(), (amplitude, curr_phase))
            bits.append(get_key(curr_constellation, value))

            prev_constellation = curr_constellation.copy()

            ###Адаптивный алгоритм декодирования сигналов на основе фильтра автоподстройки фазы###
            d_phase = self.modulation[bits[-1]][1] + alpha

            if alpha > 0:
                d_phase = d_phase if d_phase < 360 else 360 - d_phase
            elif alpha < 0:
                d_phase = d_phase if d_phase > 0 else 360 + d_phase

            if curr_phase >= d_phase:
                alpha += coef_dempf
            elif curr_phase < d_phase:
                alpha -= coef_dempf

            if alpha > 360:
                alpha -= 360
            if alpha < -360:
                alpha += 360

            c = [(a * np.cos(p / 180.0 * np.pi), a * np.sin(p / 180.0 * np.pi), t) for t, (a, p) in curr_constellation.items()]
            self.cm.append(c)

        return bits

    #################################
    def setIQSamles(self, i, q):
        self.sI.append(i)
        self.sQ.append(q)

    #################################
    def plot_constellation(self, r):

        sx, sy, t = zip(*self.constellation)
        plt.clf()
        plt.scatter(sx, sy, s=30)
        plt.scatter(self.sQ, self.sI, s=10, c='deeppink')
        plt.axes().set_aspect('equal')
        for con in self.cm:
            for x, y, t in con:
                plt.scatter(x, y, s=5, c='green')
        for x, y, t in self.constellation:
            plt.annotate(t, (x - .04, y - .04), ha = 'right', va = 'top')
        plt.axis([-r, r, -r, r])
        plt.axhline(0, color='red')
        plt.axvline(0, color='red')
        plt.grid(True)
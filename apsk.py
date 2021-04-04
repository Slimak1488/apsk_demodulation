import numpy as np
from numpy import sqrt, pi
from matplotlib import pyplot as plt
from sigproc import Signal
from scipy.signal import butter, lfilter, freqs, firwin, fftconvolve, medfilt
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

        self.constellation = self.set_constellation(modulation)
        self.cm = []
        self.I = []
        self.Q = []
        self.sI = []
        self.sQ = []
        self.dif = 0
        self.shift_phase_r1 = 0
        self.shift_phase_r2 = 0
        self.out_symbols = ''

    #################################
    def set_constellation(self, modulation):
        return [(a*np.cos(p), a*np.sin(p), t) for t, (a, p) in modulation.items()]

    #################################
    def modulate_signal(self, data, savefile=None, lvl_noise=None, phase0=None, shift_dopler=0):
        '''
        Generate signal corresponding to the current modulation scheme to
        represent given binary string, data.
        '''
        def create_func(data):
            slot_data = []
            if phase0 is not None:
                print(phase0)
                shift_phase = phase0 / 180.0 * pi
            else:
                shift_phase = np.random.uniform(pi / 12, 2 * pi)
                shift_phase = 201/180*pi
                # print("Началная фаза:", shift_phase/pi * 180)
            for i in range(0, len(data), self.bits_per_baud):
                slot_data.append(self.modulation[data[i:i+self.bits_per_baud]])

            def timefunc(t):
                slot = int(t*self.baud_rate)
                amplitude, phase = slot_data[slot]
                freq = 2 * pi * (self.carrier_freq + shift_dopler) * t + shift_phase
                freq = freq if freq < 2 * pi else freq - 2 * pi
                return amplitude*np.sin(freq + phase)

            return timefunc

        func = create_func(data)
        duration = float(len(data))/(self.baud_rate*self.bits_per_baud)
        s = Signal(duration=duration, carrier_freq=self.carrier_freq, func=func)

        lo, hi = 46000, 50000
        sr = 500000
        b, a = butter(N=6, Wn=[2 * lo / sr, 2 * hi / sr], btype='band')
        s.signal = lfilter(b, a, s.signal)

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
            freq = 2 * pi * self.carrier_freq * t
            freq = freq if freq < 2 * pi else freq - 2 * pi
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

        cutoff = float(self.carrier_freq / 240)
        N = int(n // (self.baud_rate*duration))#!!!
        symbol_period = 1 / self.baud_rate
        I, Q = lowpass_filter(I, Q, 255, cutoff, N, symbol_period, 1/100000.)

        m_signal.set_IQ_components(I, Q)

        self.component_to_bits(I[::50], Q[::50])
        bits = self.component_to_bits(I[::50], Q[::50], self.setIQSamles)
        # self.component_to_bits(I, Q, self.setIQ)
        for s in bits[1:]:
            self.out_symbols += str(s)
        phase = self.findPhaseInversion(preambul)
        print("phase", phase)
        if phase == -1:
            phase = 0
        self.shift_phase_r1 -= phase
        self.shift_phase_r2 -= phase
        self.sI.clear()
        self.sQ.clear()
        bits = self.component_to_bits(I[::50], Q[::50], self.setIQSamles)
        self.out_symbols = ''
        for s in bits[1:]:
            self.out_symbols += str(s)

        return m_signal

    #################################
    def component_to_bits(self, iSamples, qSamples, printFunc=None):
        coef_dempf = 3.6
        alpha = []
        beta = []
        bits = []

        def getChangedPhaseModulation(phi0):
            constellation = self.modulation.copy()
            for symbol in constellation:
                phi = constellation[symbol][1] + phi0
                if phi > 2*pi:
                    phi -= 2*pi
                constellation[symbol] = (constellation[symbol][0], phi)
            return constellation

        # prev_constellation = self.modulation.copy()

        for (i, q) in zip(iSamples, qSamples):
            if i > 0:
                phase = (pi/2 - np.arctan(q/i) if np.arctan(q/i) > 0 else pi/2 - np.arctan(q/i))
            else:
                phase = (3*pi/2 - np.arctan(q / i) if np.arctan(q / i) > 0 else 3*pi/2 - np.arctan(q / i))

            amplitude = np.sqrt(i**2 + q**2)

            if amplitude > 2*sqrt(7):
                phase += self.shift_phase_r1
            else:
                phase += self.shift_phase_r2

            if phase < 0:
                phase += 2*pi

            if phase > 2*pi:
                phase -= 2*pi

            if printFunc:
                printFunc(amplitude, phase)

            curr_constellation = getChangedPhaseModulation(0)
            value = self.nearest_value(curr_constellation.values(), (amplitude, phase))
            bits.append(self.getKey(curr_constellation, value))

            if amplitude > 2*sqrt(7):
                alpha.append(curr_constellation[bits[-1]][1] - phase)
            else:
                beta.append(curr_constellation[bits[-1]][1] - phase)

            # prev_constellation = curr_constellation.copy()

        if printFunc is None:
            self.shift_phase_r1 = np.median(alpha)
            self.shift_phase_r2 = np.median(beta)

            pos_alpha = [i for i in alpha if i > 0]
            neg_alpha = [i for i in alpha if i < 0]

            if np.median(pos_alpha) >= pi / 13.6 and np.median(neg_alpha) <= -pi / 13.6:
                if -sum(neg_alpha)/len(neg_alpha) >= sum(pos_alpha)/len(pos_alpha):
                    self.shift_phase_r1 = np.median(neg_alpha)
                else:
                    self.shift_phase_r1 = np.median(pos_alpha)

            pos_beta = [i for i in beta if i > 0]
            neg_beta = [i for i in beta if i < 0]

            if np.median(pos_beta) >= pi / 4.6 and np.median(neg_beta) <= -pi / 4.6:
                if -sum(neg_beta)/len(neg_beta) >= sum(pos_beta)/len(pos_beta):
                    self.shift_phase_r2 = np.median(neg_beta)
                else:
                    self.shift_phase_r2 = np.median(pos_beta)

            if np.abs(self.shift_phase_r1 - self.shift_phase_r2) > 0.91:
                if self.shift_phase_r1 < 0:
                    self.shift_phase_r2 -= pi / 2
                else:
                    self.shift_phase_r2 += pi / 2

            if self.shift_phase_r2 < 0 and np.abs(self.shift_phase_r1 - self.shift_phase_r2) > pi/80:
                self.shift_phase_r1 -= pi / 6
            if self.shift_phase_r2 > 0 and np.abs(self.shift_phase_r1 - self.shift_phase_r2) > pi/80:
                self.shift_phase_r1 += pi / 6

            print("r1: ", self.shift_phase_r1)
            print("r2: ", self.shift_phase_r2)

            self.dif = np.abs(self.shift_phase_r1 - self.shift_phase_r2)
            print(self.dif)

        listFreqBits = []
        for i in range(len(bits)//10):
            lfreq = bits[i*10:i*10+10]
            listFreqBits.append(max(set(lfreq), key=lfreq.count))

        return listFreqBits

    #################################
    def setIQSamles(self, a, ph):
        self.sI.append(a*np.sin(ph))
        self.sQ.append(a*np.cos(ph))

    def setIQ(self, a, ph):
        self.I.append(a*np.sin(ph))
        self.Q.append(a*np.cos(ph))

    #################################
    def findPhaseInversion(self, preamble):
        for phase in [0, pi/2, pi, 3*pi/2]:
            rot_preamble = ''
            for i in range(0, len(preamble), self.bits_per_baud):
                value = self.modulation[preamble[i:i+self.bits_per_baud]]
                ph0 = value[1] + phase
                if ph0 > 2*pi:
                    ph0 -= 2*pi
                value = value[0], ph0
                value = self.nearest_value(self.modulation.values(), (value[0], value[1]))
                rot_preamble += self.getKey(self.modulation, value)
            if self.out_symbols.find(rot_preamble) % self.bits_per_baud == 0:
                return phase
        return -1

    def nearest_value(self, values: list, one: (float, int)) -> (float, int):
        return min(values, key=lambda n: (abs((one[0] - n[0]) ** 2 + (one[1] - n[1]) ** 2), n))

    def getKey(self, d, val):
        for k, v in d.items():
            if v == val:
                return k

    def getDemodulateSymbols(self):
        return self.out_symbols

    def getShiftPhaseR1(self):
        return self.shift_phase_r1/pi*180

    def getShiftPhaseR2(self):
        return self.shift_phase_r2/pi*180

    def plot_constellation(self, r):
        sx, sy, t = zip(*self.constellation)
        plt.clf()
        plt.scatter(sx, sy, s=20)
        plt.scatter(self.sQ, self.sI, s=15, c='deeppink')
        plt.axes().set_aspect('equal')
        # for con in self.cm:
        #     for x, y, t in con:
        #         plt.scatter(x, y, s=5, c='green')
        for x, y, t in self.constellation:
            plt.annotate(t, (x - .04, y - .04), ha='right', va='top')
        plt.plot(self.Q, self.I, linestyle='solid', linewidth=0.3, color='green')
        plt.axis([-r, r, -r, r])
        plt.axhline(0, color='red')
        plt.axvline(0, color='red')
        plt.grid(True)

        #####Восстановление начальной фазы#######
        # if count_bit < len(preambul):
        #     key = preambul[count_bit:count_bit + self.bits_per_baud]
        #     dif_phase = phase - prev_constellation[key][1]
        #     if dif_phase < 0:
        #         dif_phase = 2*pi - np.abs(dif_phase)
        #     shift_phase += np.abs(dif_phase) * self.bits_per_baud / len(preambul)
        #     count_bit += self.bits_per_baud
        # else:
        #     print("Сдвиг фазы", shift_phase)
        #     if phase - shift_phase < 0:
        #         ph0 = 2*pi - shift_phase
        #         phase = phase + ph0
        #     else:
        #         phase -= shift_phase

        # self.setIQSamles(amplitude, phase)

        ###Адаптивный алгоритм декодирования сигналов на основе фильтра автоподстройки фазы###
        # d_phase = self.modulation[bits[-1]][1] + alpha

        # c = self.set_constellation(curr_constellation)
        # self.cm.append(c)
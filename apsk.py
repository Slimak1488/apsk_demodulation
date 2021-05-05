import numpy as np
from numpy import sqrt, pi
from matplotlib import pyplot as plt
from sigproc import Signal
from scipy.signal import butter, lfilter, freqs, firwin, fftconvolve, medfilt
from scipy.fftpack import fft, ifft, fftshift
from commpy.filters import rcosfilter


#################################
class APSK:

    #################################
    def __init__(self,
                 modulation={'0': (0, 0), '1': (1, 0)},
                 baud_rate=10,
                 bits_per_baud=1,
                 carrier_freq=100):

        self.modulation = modulation
        self.baud_rate = baud_rate
        self.bits_per_baud = bits_per_baud
        self.carrier_freq = carrier_freq

        self.constellation = self.set_constellation(modulation)

    #################################
    def set_constellation(self, modulation):
        return [(a * np.cos(p), a * np.sin(p), t) for t, (a, p) in modulation.items()]


#################################
class APSKModulator(APSK):

    #################################
    def __init__(self,
                 modulation,
                 baud_rate,
                 bits_per_baud,
                 carrier_freq,
                 sampling_rate=500000):

        self.sampling_rate = sampling_rate

        super(APSKModulator, self).__init__(modulation, baud_rate, bits_per_baud, carrier_freq)

    #################################
    def modulate_signal(self, data, savefile=None, lvl_noise=None, phase0=None, shift_dopler=0):

        def create_func(data):
            slot_data = []
            if phase0 is not None:
                print(phase0)
                shift_phase = phase0 / 180.0 * pi
            else:
                # shift_phase = np.random.uniform(pi / 12, 2 * pi)
                shift_phase = 59 / 180.0 * pi
            for i in range(0, len(data), self.bits_per_baud):
                slot_data.append(self.modulation[data[i:i + self.bits_per_baud]])

            def timefunc(t):
                slot = int(t * self.baud_rate)
                amplitude, phase = slot_data[slot]
                freq = 2 * pi * (self.carrier_freq + shift_dopler) * t + shift_phase
                freq = freq if freq < 2 * pi else freq - 2 * pi
                return amplitude * np.sin(freq + phase)

            return timefunc

        func = create_func(data)
        duration = float(len(data)) / (self.baud_rate * self.bits_per_baud)
        s = Signal(duration, self.sampling_rate, self.carrier_freq, func)

        lo, hi = self.carrier_freq - 2 * self.baud_rate, self.carrier_freq + 2 * self.baud_rate
        b, a = butter(N=2, Wn=[2 * lo / self.sampling_rate, 2 * hi / self.sampling_rate], btype='band')
        s.signal = lfilter(b, a, s.signal)

        if lvl_noise is not None:
            s.addNoise(lvl_noise)

        if savefile:
            s.write_wav(savefile)

        return s


#################################
class APSKDemodulator(APSK):

    #################################
    def __init__(self,
                 modulation,
                 baud_rate,
                 bits_per_baud,
                 carrier_freq,
                 preamble,
                 decision_boundary):

        self.I = []
        self.Q = []
        self.quad_counts = 10
        self.decimation = 0
        self.preamble = preamble
        self.shift_phase_r1 = 0
        self.shift_phase_r2 = 0
        self.coef_demp = 0
        self.out_symbols = ''
        self.decision_boundary = decision_boundary

        super(APSKDemodulator, self).__init__(modulation, baud_rate, bits_per_baud, carrier_freq)

    #################################
    def demodulate_signal(self, readfile):
        s = Signal(carrier_freq=self.carrier_freq)
        s.read_wav(readfile)

        def timefunc(t, sig):
            freq = 2 * pi * self.carrier_freq * t
            freq = freq if freq < 2 * pi else freq - 2 * pi
            I = 2 * sig * np.cos(freq)
            Q = 2 * sig * np.sin(freq)
            return I, Q

        n = s.get_len_signal()
        I, Q = s.getShiftFreqSignal(timefunc, n)

        def lowpass_filter(i, q, numtaps, cutoff, N, Ts, Fs, alpha=0.35):

            h = firwin(numtaps, cutoff, nyq=float(self.carrier_freq / 2))
            i, q = lfilter(h, 1.0, i), lfilter(h, 1.0, q)

            sPSF = rcosfilter(N, alpha, Ts, Fs)[1]
            i, q = np.convolve(sPSF, i, mode='same'), np.convolve(sPSF, q, mode='same')

            return i, q

        duration = s.get_duration()

        cutoff = float(self.carrier_freq / 240)
        samples_per_symbol = int(n / (self.baud_rate * duration)) #количество отсчётов на один символ
        symbol_period = 1 / self.baud_rate
        I, Q = lowpass_filter(I, Q, 255, cutoff, samples_per_symbol, symbol_period, 1 / 100000.)

        s.set_IQ_components(I, Q)

        self.decimation = samples_per_symbol // self.quad_counts
        sI, sQ = I[::self.decimation], Q[::self.decimation]

        gain = self.automatic_gain_control(sI, sQ)
        # gain = 1
        self.coef_demp = self.calculate_coef_dempf(sI[:5*self.quad_counts], sQ[:5*self.quad_counts])
        # self.coef_demp = 0.061
        self.coef_demp = 0.055
        # self.component_to_bits(sI, sQ, gain=gain, Phaser=False)

        bits = self.component_to_bits(sI, sQ, gain)
        for i in bits:
            self.out_symbols += str(i)
        phase = self.findPhaseInversion()
        print("phase", phase)
        if phase == -1:
            phase = 0
        self.shift_phase_r1 -= phase
        self.shift_phase_r2 -= phase

        bits = self.component_to_bits(sI, sQ, gain, setIQ=True)
        # self.component_to_bits(I, Q, gain, setIQ=True)
        self.out_symbols = ''
        for i in bits:
            self.out_symbols += str(i)

        return s

    #################################
    def automatic_gain_control(self, iSamples, qSamples):
        amplitude = []
        for (i, q) in zip(iSamples, qSamples):
            amplitude.append(np.sqrt(i ** 2 + q ** 2))

        print(sum(amplitude) / len(amplitude))
        m_decision_boundary = sum(amplitude) / len(amplitude)

        return self.decision_boundary / m_decision_boundary

    #################################
    def calculate_coef_dempf(self, iSamples, qSamples):
        raid_phase = 0
        cur_phase = None
        for (i, q) in zip(iSamples[::self.quad_counts], qSamples[::self.quad_counts]):
            pre_phase = cur_phase
            if i > 0:
                cur_phase = (pi / 2 - np.arctan(q / i) if np.arctan(q / i) > 0 else pi / 2 - np.arctan(q / i))
            else:
                cur_phase = (3 * pi / 2 - np.arctan(q / i) if np.arctan(q / i) > 0 else 3 * pi / 2 - np.arctan(q / i))

            if pre_phase is not None:
                raid_phase = np.abs(cur_phase - pre_phase) / self.quad_counts

        return raid_phase

    #################################
    def component_to_bits(self, iSamples, qSamples, gain=1, Damper=True, Phaser=False, setIQ=False):
        alpha = []
        beta = []
        symbols = []
        a_flag = 0
        ph_flag = 0
        delta = 0
        count = 0
        tmp1 = 0

        for (i, q) in zip(iSamples, qSamples):
            if i > 0:
                phase = (pi / 2 - np.arctan(q / i) if np.arctan(q / i) > 0 else pi / 2 - np.arctan(q / i))
            else:
                phase = (3 * pi / 2 - np.arctan(q / i) if np.arctan(q / i) > 0 else 3 * pi / 2 - np.arctan(q / i))

            amplitude = gain * np.sqrt(i ** 2 + q ** 2)

            if amplitude > self.decision_boundary:
                phase += self.shift_phase_r1
                a_flag |= 1
            elif amplitude > self.decision_boundary / 4:
                phase += self.shift_phase_r2
                a_flag |= 2
            else:
                symbols.append('')
                count += 1
                if setIQ:
                    self.setIQ(amplitude, phase)
                continue

            phase += delta

            if phase < 0:
                phase += 2 * pi

            if phase > 2 * pi:
                phase -= 2 * pi

            if Damper:
                tmp = 0
                count += 1
                reference_phase = self.nearest_value(self.modulation.values(), (amplitude, phase))[1]
                if reference_phase - phase > self.coef_demp / 2:
                    delta += self.coef_demp
                    phase += self.coef_demp
                    tmp = self.coef_demp
                    ph_flag |= 1
                elif reference_phase - phase < -self.coef_demp / 2:
                    delta -= self.coef_demp
                    phase -= self.coef_demp
                    tmp = -self.coef_demp
                    ph_flag |= 2

                if a_flag == ph_flag == 3:
                    delta += tmp1
                    a_flag, ph_flag = 0, 0
                tmp1 = -tmp
                if count % self.quad_counts == 0:
                    a_flag, ph_flag = 0, 0

            value = self.nearest_value(self.modulation.values(), (amplitude, phase))
            symbols.append(self.getKey(self.modulation, value))

            if setIQ:
                self.setIQ(amplitude, phase)

            if amplitude > self.decision_boundary:
                alpha.append(self.modulation[symbols[-1]][1] - phase)
            else:
                beta.append(self.modulation[symbols[-1]][1] - phase)

        if Phaser:
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

        listFreqBits = []
        for i in range(len(symbols) // self.quad_counts):
            lfreq = symbols[i * self.quad_counts:i * self.quad_counts + self.quad_counts]
            listFreqBits.append(max(set(lfreq[self.quad_counts // 2:]), key=lfreq[self.quad_counts // 2:].count))

        return listFreqBits

    #################################
    def setIQ(self, a, ph):
        self.I.append(a * np.sin(ph))
        self.Q.append(a * np.cos(ph))

    #################################
    def changeIQ(self, i, a, ph):
        self.I[i] = a * np.sin(ph)
        self.Q[i] = a * np.cos(ph)

    #################################
    def findPhaseInversion(self):
        for phase in [0, pi / 2, pi, 3 * pi / 2]:
            rot_preamble = ''
            for i in range(0, len(self.preamble), self.bits_per_baud):
                value = self.modulation[self.preamble[i:i + self.bits_per_baud]]
                ph0 = value[1] + phase
                if ph0 > 2 * pi:
                    ph0 -= 2 * pi
                value = value[0], ph0
                value = self.nearest_value(self.modulation.values(), (value[0], value[1]))
                rot_preamble += self.getKey(self.modulation, value)
            if self.out_symbols.find(rot_preamble) % self.bits_per_baud == 0:
                return phase
        return -1

    #################################
    def nearest_value(self, values: list, one: (float, int)) -> (float, int):
        return min(values, key=lambda n: (abs((one[0] - n[0]) ** 2 + (one[1] - n[1]) ** 2), n))

    #################################
    def getKey(self, d, val):
        for k, v in d.items():
            if v == val:
                return k

    #################################
    def getDemodulateSymbols(self):
        return self.out_symbols

    #################################
    def plot_constellation(self, r):
        sx, sy, t = zip(*self.constellation)
        plt.clf()
        plt.scatter(sx, sy, s=20)
        # plt.scatter(self.Q[::self.decimation], self.I[::self.decimation], s=15, c='deeppink')
        plt.scatter(self.Q[:], self.I[:], s=15, c='deeppink')
        plt.axes().set_aspect('equal')
        for x, y, t in self.constellation:
            plt.annotate(t, (x - .04, y - .04), ha='right', va='top')
        plt.plot(self.Q[:], self.I[:], linestyle='solid', linewidth=0.3, color='green')
        plt.axis([-r, r, -r, r])
        plt.axhline(0, color='red')
        plt.axvline(0, color='red')
        plt.grid(True)

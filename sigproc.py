
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from commpy.filters import rrcosfilter

plt.rc('font', family='Sawasdee', weight='bold')
plt.rc('axes', unicode_minus=False)

###########################################
class Signal(object):

    #######################################
    def __init__(self, duration=1.0, sampling_rate=500000, carrier_freq=0, func=None):
        '''
        Initialize a signal object with the specified duration (in seconds)
        and sampling rate (in Hz).  If func is provided, signal
        data will be initialized to values of this function for the entire
        duration.
        '''
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.carrier_freq = carrier_freq
        self.freqs = np.arange(int(duration*sampling_rate), dtype=complex)
        self.signal = np.arange(int(duration*sampling_rate), dtype=np.float64)
        self.freqs[:] = 0j
        self.I = []
        self.Q = []
        if func is not None:
            self.t=self.sample_time_function(func)

    #######################################
    def read_wav(self, wav_file):

        rate, data = wavfile.read(wav_file)
        n = data.shape[0]
        self.sampling_rate = rate
        self.duration = float(n)/rate
        normalizer = 4

        self.signal = data
        self.freqs = fft(data)

    #######################################
    def write_wav(self, wav_file):

        normalized_signal = self.signal / np.abs(self.signal).max()
        wavfile.write(
                wav_file, 
                self.sampling_rate, 
                self.signal.astype(np.dtype('float32')))

    #######################################
    def get_sampling_rate(self):

        return self.sampling_rate

    #######################################
    def get_duration(self):

        return self.duration

    def get_len_signal(self):

        return len(self.signal)

    #######################################
    def sample_time_function(self, func):
        n = self.get_len_signal()
        signal = np.zeros(n, dtype=float)
        for i in range(n):
            signal[i] = func(float(i)/self.sampling_rate)
            self.signal = signal
            self.freqs = fft(signal)

    #######################################
    def getShiftFreqSignal(self, func, n):
        shift_I_component = np.zeros(n, dtype=float)
        shift_Q_component = np.zeros(n, dtype=float)

        for i in range(n):
            shift_I_component[i], shift_Q_component[i] = func(float(i) / self.sampling_rate, self.signal[i])
        return shift_I_component, shift_Q_component

    #######################################
    def addNoise(self, noise_level_db):
        noise_avg_watts = 10 ** (noise_level_db / 10.)
        m_noise = np.random.normal(0, noise_avg_watts, len(self.signal))
        n = m_noise
        sig = self.signal
        self.signal = sig + n

    #######################################
    def set_IQ_components(self, I, Q):
        self.I = I
        self.Q = Q

    #######################################
    def get_time_domain(self):

        x_axis = np.linspace(0, self.duration, len(self.freqs))
        y_axis = self.signal
        return x_axis, y_axis

    #######################################
    def get_IQ_domain(self):
        x_axis = np.linspace(0, self.duration, len(self.freqs))
        I_axis = self.I
        Q_axis = self.Q
        return x_axis, I_axis, Q_axis

    #######################################
    def get_freq_domain(self):
        '''
        Return a tuple (X,A,P) where X is an array storing the frequency axis
        up to the Nyquist frequency (excluding negative frequency), and A and
        P are arrays storing the amplitude and phase shift (in degree) of each
        frequency
        '''
        n = len(self.freqs)
        num_freqs = int(np.ceil((n+1)/2.0))
        x_axis = np.linspace(0, self.sampling_rate, n)

        # extract only positive frequencies and scale them so that the
        # magnitude does not depend on the length of the array
        a_axis = abs(self.freqs[:num_freqs])/float(n)
        p_axis = np.arctan2(
                    self.freqs[:num_freqs].imag,
                    self.freqs[:num_freqs].real) * 180.0/np.pi

        # double amplitudes of the AC components (since we have thrown away
        # the negative frequencies)
        a_axis[1:] = a_axis[1:]*2

        return x_axis, a_axis, p_axis

    #######################################
    def copy(self):
        '''
        Clone the signal object into another identical signal object.
        '''
        s = Signal()
        s.duration = self.duration
        s.sampling_rate = self.sampling_rate
        s.freqs = np.array(self.freqs)
        return s

    #######################################
    def mix(self, signal):
        '''
        Mix the signal with another given signal.  Sampling rate and duration
        of both signals must match.
        '''
        if self.sampling_rate != signal.sampling_rate \
           or len(self.freqs) != len(signal.freqs):
            raise Exception(
                'Signal to mix must have identical sampling rate and duration')

        self.freqs += signal.freqs

    #######################################
    def __add__(self, s):
        newSignal = self.copy()
        newSignal.mix(s)
        return newSignal

    #######################################
    def plot(self, dB=False, phase=False, stem=False, IQ_component=False, frange=(0, 40000)):

        plt.subplots_adjust(hspace=.5)

        if phase:
            num_plots = 3
        else:
            num_plots = 2

        if IQ_component:
            num_plots = 5

        # plot time-domain signal
        plt.subplot(num_plots, 1, 1)
        plt.cla()
        x,y = self.get_time_domain()
        plt.grid(True)
        plt.xlabel(u'Time (s)')
        plt.ylabel('Amplitude')
        plt.plot(x,y,'g')

        # plot frequency vs. amplitude
        x,a,p = self.get_freq_domain()
        start_index = int(float(frange[0])/self.sampling_rate*len(self.freqs))
        stop_index  = int(float(frange[1])/self.sampling_rate*len(self.freqs))
        x = x[start_index:stop_index]
        a = a[start_index:stop_index]
        p = p[start_index:stop_index]
        plt.subplot(num_plots, 1, 2)
        plt.cla()
        plt.grid(True)
        plt.xlabel(u'Frequency (Hz)')

        if dB:
            a = 10.*np.log10(a + 1e-10) + 100
            plt.ylabel(u'Amplitude (dB)')
        else:
            plt.ylabel(u'Amplitude')

        if stem:
            plt.stem(x,a,'b')
        else:
            plt.plot(x,a,'b')

        # plot frequency vs. phase-shift
        if phase:
            plt.subplot(num_plots, 1, 3)
            plt.cla()
            plt.grid(True)
            plt.xlabel(u'Frequency (Hz)')
            plt.ylabel(u'Phase (degree)')
            plt.ylim(-180,180)
            if stem:
                plt.stem(x[start_index:stop_index],p[start_index:stop_index],'r')
            else:
                plt.plot(x[start_index:stop_index],p[start_index:stop_index],'r')

        if IQ_component:
            x, Iy, Qy = self.get_IQ_domain()

            plt.subplot(num_plots, 1, 4)
            plt.cla()
            plt.grid(True)
            plt.xlabel(u'Time (s)')
            plt.ylabel('I')
            plt.plot(x, Iy, 'r')

            plt.subplot(num_plots, 1, 5)
            plt.cla()
            plt.grid(True)
            plt.xlabel(u'Time (s)')
            plt.ylabel('Q')
            plt.plot(x, Qy, 'r')

        plt.show()




import sys
from apsk import Apsk
from matplotlib import pyplot as plt
import numpy as np
from numpy import sqrt, pi, random
from sigproc import Signal

from pylab import*
R1 = sqrt(7)
R2 = 3*R1

modulation = {
    '0000': (R2, pi/4),
    '0001': (R2, 7*pi/4),
    '0010': (R2, 3*pi/4),
    '0011': (R2, 5*pi/4),
    '0100': (R2, pi/12),
    '0101': (R2, 23*pi/12),
    '0110': (R2, 11*pi/12),
    '0111': (R2, 13*pi/12),
    '1000': (R2, 5*pi/12),
    '1001': (R2, 19*pi/12),
    '1010': (R2, 7*pi/12),
    '1011': (R2, 17*pi/12),
    '1100': (R1, pi/4),
    '1101': (R1, 7*pi/4),
    '1110': (R1, 3*pi/4),
    '1111': (R1, 5*pi/4),
}


def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def test1():
    # data = input("Inpute text: ")
    # symbols = text_to_bits(data)
    bits = np.random.randint(0, 2, 40)
    symbols = ''
    for bit in bits:
        symbols += str(bit)

    freq = 48000

    preambul = "011001010110110001010110010101100101011001011011101111110010011011001111100111011111000011100111"
    q1 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
    print(symbols)
    # bits = '11001100110011001100110011001100'
    symbols = "0000110000010101110100101010110000100111100111111110111110010011011001111100111011111000011100111110111110000111000011000011000011000000"
    print(symbols)
    m_signal = q1.modulate_signal(symbols, savefile='3.wav', lvl_noise=None, shift_dopler=0)
    plt.figure(1)
    # m_signal.plot(dB=False, phase=False, stem=True, frange=(0, 1000))

    q2 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
    d_signal = q2.demodulate_signal('3.wav', preambul)

    demod_symbols = q2.getDemodulateSymbols()
    if symbols == demod_symbols:
        print("ОК!!!")
    else:
        # print("Error!!!")
        print(demod_symbols)

    q2.plot_constellation(R2 + 1)
    plt.figure(3)
    d_signal.plot(stem=True, IQ_component=True, frange=(0, 100000))

def test2():
    symbols = "0000110000010101110100101010110000100111"
    freq = 48000
    y_r1 = []
    y_r2 = []
    maxshiftphase = 360
    x_axis = range(maxshiftphase)
    for phase in x_axis:
        q1 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
        q1.modulate_signal(symbols, savefile='3.wav', lvl_noise=None, phase0=phase, shift_dopler=0)
        q2 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
        q2.demodulate_signal('3.wav', '0000')
        y_r1.append(q2.getShiftPhaseR1())
        y_r2.append(q2.getShiftPhaseR2())

    plt.subplot(2, 1, 1)
    plt.cla()
    plt.plot(x_axis, y_r1)
    plt.ylabel("r1")
    plt.subplot(2, 1, 2)
    plt.cla()
    plt.plot(x_axis, y_r2)
    plt.ylabel("r2")

    plt.show()


if __name__ == '__main__':
    test1()
    # test2()
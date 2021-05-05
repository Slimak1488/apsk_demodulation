import sys
from apsk import APSKModulator, APSKDemodulator
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

    carrier_freq = 48000

    preamble = ""
    # symbols = "010000001100000100001100000101011101010011010010101011000010"#000011000001"
    # symbols = '01000000001000010010111111000100001111111011100001110111010111110011101001101110111111100101'
    # symbols = "01000000001000010010111111000100"
    q1 = APSKModulator(modulation, 1000, 4, carrier_freq)
    preamble = "0000111100101111110100111100000111100000"
    symbols = "0000010001010001" #1
    # symbols = "00101010100000000100010100011010100100011011001111001111010101110110001000101010"
    symbols = "0000110000010101110101001101001010101100001001111001111111101111100100110110011111001110111110000111001111101111100001110000110000110000010011000000"
    i = np.random.randint(4, (len(symbols)-4)//4)
    symbols = symbols[:4*i] + preamble + symbols[4*i:]
    m_signal = q1.modulate_signal(symbols, savefile='3.wav', lvl_noise=None, shift_dopler=20)
    # plt.figure(1)
    # m_signal.plot(dB=False, phase=False, stem=True, frange=(0, 1000))

    q2 = APSKDemodulator(modulation, 1000, 4, carrier_freq, preamble, 2*R1)
    d_signal = q2.demodulate_signal('3.wav')

    demod_symbols = q2.getDemodulateSymbols()
    if symbols == demod_symbols:
        print("ОК!!!")
    else:
        print(symbols)
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
        q1 = APSKModulator(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
        q1.modulate_signal(symbols, savefile='3.wav', lvl_noise=None, phase0=phase, shift_dopler=0)
        q2 = APSKDemodulator(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
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

def test3():
    preambul = "00001111001011010011110000011110"
    y_axis = []
    freq = 48000
    maxshiftPhase = 360
    x_axis = range(maxshiftPhase)
    for phase in x_axis:
        bits = np.random.randint(0, 2, 200)
        symbols = ''
        for bit in bits:
            symbols += str(bit)
        i = np.random.randint(4, (len(symbols) - 4) // 4)
        symbols = symbols[:4 * i] + preambul + symbols[4 * i:]
        q1 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
        q1.modulate_signal(symbols, savefile='3.wav', lvl_noise=None, phase0=phase, shift_dopler=0)
        q2 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
        q2.demodulate_signal('3.wav', preambul)
        demod_symbols = q2.getDemodulateSymbols()
        # y_axis.append(q2.dif)
        if symbols[:-4] == demod_symbols:
            y_axis.append(1)
        else:
            print("Входная последовательность:", symbols)
            print("Выходная последовательность:", demod_symbols)
            y_axis.append(0)

    plt.plot(x_axis, y_axis)
    plt.show()

if __name__ == '__main__':
    test1()
    # test2()
    # test3()
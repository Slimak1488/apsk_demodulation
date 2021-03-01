import sys
from apsk import Apsk
from matplotlib import pyplot as plt
from numpy import sqrt
from sigproc import Signal

from pylab import*
R1 = sqrt(7)
R2 = 3*R1

modulation = {
    '0000' : (R2, 45.0000),
    '0001' : (R2, 315.0000),
    '0010' : (R2, 135.0000),
    '0011' : (R2, 225.0000),
    '0100' : (R2, 15.0000),
    '0101' : (R2, 345.0000),
    '0110' : (R2, 165.0000),
    '0111' : (R2, 195.0000),
    '1000' : (R2, 75.0000),
    '1001' : (R2, 285.0000),
    '1010' : (R2, 105.0000),
    '1011' : (R2, 255.0000),
    '1100' : (R1, 45.0000),
    '1101' : (R1, 315.0000),
    '1110' : (R1, 135.0000),
    '1111' : (R1, 225.0000),
}


def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


if __name__ == '__main__':
    data1 = input("Inpute text: ")
    bits = text_to_bits(data1)
    freq = 48000
    # preambul = "1010100110101001101010011010100110101001"
    # preambul =   "1000101110001011100010111000101110001011"
    # preambul = "0100011101000111010001110100011101000111"
    preambul = "0110010101100101011001010110010101100101"
    # preambul = "0110010001100100011001000110010001100100"
    q1 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
    print(bits)
    bits = preambul + bits
    m_signal = q1.modulate_signal(bits, savefile='3.wav', lvl_noise=None, shift_dopler=0)
    # plt.figure(1)
    # m_signal.plot(dB=False, phase=False, stem=True, frange=(0, 100000))

    q2 = Apsk(baud_rate=1000, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
    d_signal = q2.demodulate_signal('3.wav', preambul)
    q2.plot_constellation(R2+1)
    plt.figure(3)
    d_signal.plot(stem=True, IQ_component=True, frange=(0, 100000))


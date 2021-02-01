import sys
from apsk import Apsk
from matplotlib import pyplot as plt
from numpy import sqrt
from sigproc import Signal

R1 = sqrt(7)
R2 = 3*R1

modulation = {
    '0000' : (R2, 45.0000),
    '0001' : (R2, 315.0000),
    '0010' : (R2,  135.0000),
    '0011' : (R2,  225.0000),
    '0100' : (R2, 15.0000),
    '0101' : (R2, 345.0000),
    '0110' : (R2, 165.0000),
    '0111' : (R2, 195.0000),
    '1000' : (R2, 75.0000),
    '1001' : (R2, 285.0000),
    '1010' : (R2,  105.0000),
    '1011' : (R2,  255.0000),
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
    bits1 = text_to_bits(data1)
    # bits1 = "001100000011111100000011000000110000001100000111"
    freq = 48000

    q1 = Apsk(baud_rate=10, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
    print(bits1)
    s = q1.modulate_signal(bits1)
    plt.figure(1)
    # q1.plot_constellation(R2+1)
    # plt.figure(2)
    s.write_wav('3.wav')
    s.plot(dB=False, phase=False, stem=True, frange=(0, 40000))

    q2 = Apsk(baud_rate=10, bits_per_baud=4, carrier_freq=freq, modulation=modulation)
    ss = Signal(carrier_freq=freq)
    ss.read_wav('3.wav')
    # plt.plot(ss.signal)
    # plt.show()
    I, Q = q2.demodulate_signal(ss)
    q2.plot_constellation(R2+1, Q, I)
    plt.figure(3)
    ss.plot(stem=False, IQ_component=True)

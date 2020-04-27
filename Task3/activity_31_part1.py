import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def plot_function(v, title="No Title"):
    plt.figure(figsize=(10,5))
    plt.plot(v)
    plt.title(title)
    plt.show()

def generate_sin_wave(freq_hz, n_pnts=1000):
    return np.sin(np.linspace(0, 1, (n_pnts)) * 2 * np.pi * freq_hz)

sum_of_3_sin = generate_sin_wave(5) + generate_sin_wave(50) + generate_sin_wave(100)

plot_function(sum_of_3_sin, "3 sin waves with freq: 5, 50, 100")

def filter_low_and_high(s, order=2):
    # hamming = signal.hamming(1000)
    # s = np.multiply(hamming, s)
    
    cutoff5 = 5 / (0.5 * 1000)
    cutoff100 = 100 / (0.5 * 1000)
    b1, a1 = signal.butter(order, cutoff5, "lowpass")
    b2, a2 = signal.butter(order, cutoff100, "highpass")
    signal.filtfilt(b1, a1, s)
    return signal.filtfilt(b2, a2, s)
    # @TODO: implement this.


filtered_50_hz_sin = filter_low_and_high(sum_of_3_sin)  
plot_function(filtered_50_hz_sin, "Desired Output")

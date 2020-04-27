import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import pyaudio
from numpy.linalg import norm
from numpy.fft import rfft
from scipy import signal

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 4096

controller = pyaudio.PyAudio()

# TODO: Step (1) open stream
audio_stream = controller.open(RATE, CHANNELS, FORMAT, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

fig = plt.figure(figsize=(5,5))
ax = plt.axes(xlim=(0, 100), ylim=(0, 100))

base = plt.Circle((50, 50), 0.2, fc='b')

def init():
    ax.add_patch(base)
    return base

def loop(i):
    stream_data = audio_stream.read(FRAMES_PER_BUFFER) # <- @TODO replace this line with code to read a sample from the microphone.
    data = np.fromstring(stream_data, dtype=np.int16).astype(np.float32)
    if sum(data) > 0: 
        hamming = np.hamming(4096)
        data = np.multiply(hamming, data)
    
        # fft
        fft_data = rfft(data, 4096)
        # base freq range calc w/ i * 44100 / 4096
        s = 0
        for i in range(5, 24):
            s += fft_data[i]
        s = abs(s)**2
        s = s / (24-5)
        s = s / 44100
        print(s)
        calc_base = s / 22050
        base.set_radius(calc_base)

    return base


anim = animation.FuncAnimation(fig, loop, 
        init_func=init, 
        frames=10, 
        interval=10,
        blit=False)

plt.show()


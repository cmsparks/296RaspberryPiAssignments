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
BINS = 6
FREQ_RANGE = [5, 24, 46, 186, 372, 558, 2000]

controller = pyaudio.PyAudio()

# TODO: Step (1) open stream
audio_stream = controller.open(RATE, CHANNELS, FORMAT, input=True, frames_per_buffer=FRAMES_PER_BUFFER)


fig, ax = plt.subplots()

ax.set_xlim(0, 10)
ax.set_ylim(0, 100)



# fig = plt.figure(figsize=(5,5))
# ax = plt.axes(xlim=(0, 2), ylim=(0, 2))


def init():
    return

def loop(i):
    stream_data = audio_stream.read(FRAMES_PER_BUFFER) # <- @TODO replace this line with code to read a sample from the microphone.
    data = np.fromstring(stream_data, dtype=np.int16).astype(np.float32)
    if sum(data) > 0: 
        # fft and cleaning data
        hamming = np.hamming(4096)
        data = np.multiply(hamming, data)
    
        fft_data = rfft(data, 4096)
        for d in fft_data:
            if d < 0: 
                d = 0
        fft_data = fft_data.real
        hist = []

        for r in range(BINS):
            low = round(FREQ_RANGE[r])
            high = round(FREQ_RANGE[r+1])
            v = 0
            for i in range(low, high):
                v += fft_data[i]
            if (v < 0):
                v = 0
            hist.append(v)

        m = 0
        # base freq range calc w/ i * 44100 / 4096
        for h in hist:
            h = abs(h)**2
            h = h / (24-5)
            h = h / 44100
            if h > m:
                m = h
            elif h < 0:
                h = 0

        print(hist)
        print(range(BINS))
        plt.cla()
        plt.bar([0,1,2,3,4,5], hist, width=1, align="edge")
        ax.set_xlim(0, BINS)
        ax.set_ylim(0, 44100 * 2)


anim = animation.FuncAnimation(fig, loop, 
        init_func=init, 
        frames=10, 
        interval=10,
        blit=False)

plt.show()


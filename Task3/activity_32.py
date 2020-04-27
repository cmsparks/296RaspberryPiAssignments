import pyaudio
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft
from numpy.fft import rfft
from scipy import signal
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 4096

def plot_function(v, title="No Title"):
    plt.figure(figsize=(10,5))
    plt.plot(v)
    plt.title(title)
    plt.show()

controller = pyaudio.PyAudio()

# TODO: Step (1) open stream
audio_stream = controller.open(RATE, CHANNELS, FORMAT, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

while True:
  try:
    # TODO: Step (2) read from stream
    stream_data = audio_stream.read(FRAMES_PER_BUFFER) # <-- TODO: Change this!
    data = np.fromstring(stream_data, dtype=np.int16).astype(np.float32)
    # Hamming filter
    hamming = np.hamming(4096)
    data = np.multiply(hamming, data)
    
    # fft
    fft_data = fft(data, 4096)
    plot_function(data)
    plot_function(fft_data)

    print(norm(data))
  except KeyboardInterrupt:
    # TODO: Step (3) close stream
    controller.close(audio_stream)
    controller.terminate()
    break

print('\nShutting down')


import pyaudio
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft
from scipy import signal
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 4096


#log_scale = generate_log_10_values(0, log(len(signal)))
log_scale = []
i = 0
while i < np.log10(100): # replace 100 with length of signal
    log_scale.append(10**i)
    i = i + 1

#indices = [geometric_mean(log_scale[i], log_scale[i - 1]) for i from 1 to len(log_scale) - 1]
indices = []
for i in range(1, len(log_scale)):
    temp  = log_scale[i] * log_scale[i-1]
    temp ** 0.5
    indices.append(temp)

#split_on_indices(signal, indices)

bins = [][]
bins[0] = indices

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

    if data < bins[0][0]:
        bins[0].append(data)
    elif data > bins[len(bins)][0]:
        bins[len(bins)].append(data)
    else:
        i = 1
        while i < len(bins):
            if data > bins[i - 1][0] and data < bins[i][0]:
                bins[i].append(data)


    print(norm(data))
  except KeyboardInterrupt:
    break

print('\nShutting down')

# TODO: Step (3) close stream
controller.close(audio_stream)
controller.terminate()


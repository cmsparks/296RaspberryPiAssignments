import pyaudio
import numpy as np
from numpy.linalg import norm
from scipy import signal

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 4096

controller = pyaudio.PyAudio()

# TODO: Step (1) open stream
audio_stream = controller.open(RATE, CHANNELS, FORMAT, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

while True:
  try:
    # TODO: Step (2) read from stream
    stream_data = audio_stream.read(FRAMES_PER_BUFFER) # <-- TODO: Change this!
    data = np.fromstring(stream_data, dtype=np.int16).astype(np.float32)
    # Hamming filter
    hamming = signal.hamming(4096)
    data = np.multiply(hamming, data)
    print(norm(data))
  except KeyboardInterrupt:
    break

print('\nShutting down')

# TODO: Step (3) close stream
controller.close(audio_stream)
controller.terminate()

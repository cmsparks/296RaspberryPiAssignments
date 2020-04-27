import pyaudio
import numpy as np
from numpy.linalg import norm

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 4096

controller = pyaudio.PyAudio()

# TODO: Step (1) open stream
audio_stream = controller.open()

while True:
   try:
            # TODO: Step (2) read from stream
            stream_data = audio_stream.read() # <-- TODO: Change this!
       data = np.fromstring(stream_data, dtype=np.int16).astype(np.float32)
       print(norm(data))
   except KeyboardInterrupt:
       break

print('\nShutting down')

# TODO: Step (3) close stream
controller.close(audio_stream)
controller.terminate()

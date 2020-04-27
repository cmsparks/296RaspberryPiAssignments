

## Notes:

All the task's here are geared towards the final goal of making a sound visualizer.

Changes will roll out every week. Be sure to check this webpage often for updates!

```
Changelog:
1. Corrected Mel filterbank code in Task 3.3.2. Piazza @103
2. Corrected description for Task 3.1. Piazza @107 
```
## Task 1 : Setup

The main idea behind dedicating an entire section to setup is to get you comfortable with interacting with your Raspberry Pi. For many of you, this would be the first time you'd have to interact with a _headless_ computer.  A couple of things to keep in mind while doing this task:

* What makes a Raspberry Pi _different_ from an Arduino?
* What makes a Raspberry Pi _similar_ to an Arduino?
* How can a Raspberry Pi be used in embedded system applications?

### Task 1.1: Downloading and installing Raspbian

#### Online References:
* [Reference Tutorial](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)
* [What is a headless computer?](https://en.wikipedia.org/wiki/Headless_computer)

1. __Download Raspbian:__ I'd recommend downloading and installing the full raspbian desktop environment (as opposed to the Lite installation) so that you can connect a display in case the network fails.  

2. __Flash it onto an SD Card__: The easiest way to do this is to use the [Raspberry Pi Imager](https://www.raspberrypi.org/blog/raspberry-pi-imager-imaging-utility/).

3. __Boot up__: Follow along the steps. Make sure to enable SSH, VNC and Remote GPIO.

4. __Wi-fi Setup__: If your Pi has wi-fi, you'll need to follow some additional steps.

   1. __Set Wi-fi country__: This can be done by running 

      ```bash
      $ sudo raspi-config
      ```

      and navigating into `Localization Options` > `Change Wi-fi country`. Reboot your Pi for changes to take effect.

   2. __Register your Pi on IllinoiNet_Guest__: To whitelist your Pi, go to the [IllinoisNet_Guest management portal](https://go.illinois.edu/illinoisnetguest) and click on `Add Device` (This may take ~ 10-15 minutes). You might need to find your mac address(es). This can be done using:

      ```bash
      $ ifconfig -a
      ```
   3.  __How to connect to IllinoisNet?__: IllinoisNet is harder to connect to due to the security standard it uses, which is, WPA Enterprise. 
      -  Install network manager packages
         - You need network access to be able to download packages, please connect to ethernet/wifi first before moving on.
         ```bash
         $ sudo apt install network-manager network-manager-gnome
         ```

      - Disable dhcpcd
         ```bash
         $ sudo systemctl disable dhcpcd
         $ sudo systemctl stop dhcpcd
         ```

      - Enable NetworkManager 
        
         edit ` /etc/NetworkManager/NetworkManager.conf`
         ```conf
         managed=true
         ```

      - Add wifi profile for IllinoisNet
         - Create a wifi connection
            ```bash
            $ nm-connection-editor
            ```
            Click the + button and fill details

      - Activate IllinoisNet connection
         ```bash
         $ nmtui
         ```


### Task 1.2: Connecting to Pi over SSH

#### Online References: 

* [Reference Tutorial](https://www.raspberrypi.org/documentation/remote-access/ssh/)
* [What is SSH?](https://en.wikipedia.org/wiki/Secure_Shell)
* [SSH tips and practical examples](https://hackertarget.com/ssh-examples-tunnels/)

1. __Enable SSH__: You can do this by going into `raps-config` and navigating to `Interfacing Options` > `SSH`.

2. __Connect your laptop to IllinoisNet_Guest__: You need to be on the same network as the Pi to login  via SSH.

3. __Find your Pi's IP address__: This can be done using `ifconfig` and noting down the `inet` field under `wlan0`. 

4. __Connect to your Pi!__: From your laptop, type: 

   ```bash
   $ ssh pi@<IP ADDRESS UNDER IFCONFIG>
   ```

### Task 1.3 (Extra): Running VSCode on the Raspberry Pi

#### Online References: 

* https://code.visualstudio.com/docs/remote/remote-overview

### Task 1.4 (Extra): VNC

#### Online References:

* [Reference Tutorial](https://www.raspberrypi.org/documentation/remote-access/vnc/)

### Task 1.5: CS225 on your Raspberry Pi!

Git comes pre-installed with Raspbian! Clone your CS225 Repo in any folder and try compiling some code!

### Task 1.6 (Extra): Add a CRON job to pull your CS225 Repo every x minutes! 

#### Online References:

* [A basic tutorial](https://opensource.com/article/17/11/how-use-cron-linux)
* [An online crontab editor](https://crontab.guru/)
* [An old HackerNews thread](https://news.ycombinator.com/item?id=18176031)
* [A blog post detailing best practices](https://sanctum.geek.nz/arabesque/cron-best-practices/)

## Task 2: Setting up the Audio Input

The main goal of this section is to be able to read audio from a device over bluetooth. There are 3 steps in this section:

1. Setting up the bluetooth connection
2. Routing audio from the bluetooth device to an audio sink.
3. "Listening" to the audio sink for our audio stream.

Note: We will be using python for most of our computations. The main reason for this (sudden) shift to python is because it lets us use existing libraries to abstract away the specifics for streaming audio inputs.

### Task 2.1: Installing dependencies and setting up bluetooth.

1. __Dependencies__: We first need to install a bunch of dependences.

   ```bash
   $ sudo rpi-update
   $ sudo apt update
   $ sudo apt install bluez pulseaudio-module-bluetooth python-gobject python-gobject-2 bluez-tools udev portaudio19-dev python-pyaudio python3-pyaudio
   # python3 dependencies
   $ pip3 install -U numpy scipy setuptools
   $ pip3 install pyaudio
   ```

   You may encounter errors specific to your raspberry pi model. In that case, google the specific error message (don't forget to add your pi model to the search query!). There is a lot of online discussion to assist with most of these errors. 

2. __Connect to bluetooth__: The straightforward way to go about this is to plug your raspberry pi into a display (or use VNC!) and pair the pi with your phone using the GUI. However, we can use the terminal to do the same thing!

   ```bash
   $ bluetoothctl
   [bluetooth] list
   # An output should appear representing your bluetooth dongle or the bluetooth module on the Pi 3
   [bluetooth] agent on
   [bluetooth] default-agent
   [bluetooth] discoverable on
   [bluetooth] scan on
   # The MAC address of your device that you want to pair might be listed. If so, note down the MAC address that is associated with the name of the device you want to pair
   [bluetooth] pair XX:XX:XX:XX:XX:XX
   [bluetooth] trust XX:XX:XX:XX:XX:XX
   ```

### Task 2.2: Setting up the audio playback

__Online References__:

* [Setup Pi as a bluetooth speaker](https://raspberrypi.stackexchange.com/questions/47708/setup-raspberry-pi-3-as-bluetooth-speaker)

1. __Add user to group__: The first step is to add your current user to the PulseAudio group. You can get your current user using `$ whoami`

   ```bash
   $ sudo usermod -a -G lp pi
   ```

2. __Add audio configuration file__: Create `audio.conf` at the location `/etc/bluetooth/` and paste this into the file. (You'll need sudo access to make changes at this location):

   ```bash
   [General]:
   Enable=Source,Sink,Media,Socket
   ```

3. __Start pulseaudio__: This can be done using this command:

   ```bash
   $ pulseaudio -D
   ```

4. __Check playback__: Connect a speaker (or a pair of headphones) to the audio jack and try playing something on your phone (or use the HDMI port if your monitor has speakers). If you can't hear anything, you might need to force the audio through the audio jack. This can be done by:

   ```bash
   $ sudo raspi-config
   # Advanced Options -> Audio -> Force 3.5mm  
   ```

### Task 2.3: Reading in audio data

__Online References__:

1. [Pulseaudio under the hood](https://gavv.github.io/articles/pulseaudio-under-the-hood/)
2. [PyAudio documentation](https://people.csail.mit.edu/hubert/pyaudio/docs/)

1. __Testing input and output stream__: Before we proceed, lets make sure our input and output streams work. 

   ```bash
   $ pactl info
   ```

   gives us basic information about our pulseaudio server instance including the `Default Sink` and the `Default Source`. To see a list of potential sources/sinks, we can use the command `$ pactl list sinks short` or `$ pactl list sources short`. If we want to add bluetooth streams, we need to execute `$ pacmd load-module module-bluetooth-discover` first (If you're getting a `pa_context_connect() failed ` error, reboot your pi.)

   

   Now, we can use `parecord` to record the source stream. If `paplay` gives us the input back, we are good to go.

   ```bash
   # make sure your bluetooth device is playing something
   $ parecord -v /tmp/test.wav
   $ paplay -v /tmp/test.wav
   # play back is the input stream = good to go
   ```

2. __Reading in from the source stream__:

   Reading from a sound stream in PyAudio consists of 3 steps:

   1. Open a new stream in the controller
   2. Read in data from the stream 
   3. Close the stream when done.

   Your goal for this activity is trying to figure out how to read a sound input using PyAudio. You should be able to complete this activity using only the functions given below (but its okay to use other functions as well!).

   Documentation:

   * [PyAudio()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio)
   * [PyAudio.open()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio.open)
   * [Stream.read()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.Stream.read)
   * [PyAudio.close()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio.close)

   The code provided below should help you get started.

   ```python
   import pyaudio
   import numpy as np
   from numpy.linalg import norm
   
   FORMAT = pyaudio.paInt16
   CHANNELS = 1
   RATE = 44100
   FRAMES_PER_BUFFER = 4096
   
   controller = pyaudio.PyAudio()
   
   # TODO: Step (1) open stream
   
   while True:
       try:
         	# TODO: Step (2) read from stream
         	stream_data = None # <-- TODO: Change this!
           data = np.fromstring(stream_data, dtype=np.int16).astype(np.float32)
           print(norm(data))
       except KeyboardInterrupt:
           break
   
   print('\nShutting down')
   
   # TODO: Step (3) close stream
   
   controller.terminate()
   ```

   To run this code, save it as `activity_23.py` and run: 

   ```bash
   $ python3 activity_23.py
   ```

### Task 2.4: Outputting audio data

Your goal for this activity is trying to figure out how to output sound using PyAudio. You should be able to complete this activity using only the function given below (but its okay to have something else!).

Documentation:

1. [PyAudio()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio)
2. [PyAudio.open()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.PyAudio.open)
3. [Stream.write()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.Stream.write)
4. [Stream.stop_stream()](https://people.csail.mit.edu/hubert/pyaudio/docs/#pyaudio.Stream.stop_stream)

This code should help you get started.

```python
import pyaudio
import numpy as np

rate=44100
def get_bit_stream(rate=44100, lambdah=1879.69, duration=3):
    '''
    Function that returns a bit stream with 5 tones at diff. amplitudes.
    '''
    n_frames = int(rate * duration)
    bit_stream = str()
    for x in range(n_frames):
        val = (1 + int((x * 2 * 5) / n_frames)) * (np.sin(x/((rate/lambdah)/np.pi))*127+128)
        bit_stream = bit_stream+chr(int(val))    
    return bit_stream


bit_stream = get_bit_stream()
# TODO: write the bitstream to file.
# Your code here.
```

To run this code, save it as `activity_24.py` and run: 

```bash
$ python3 activity_24.py
```


## Task 3: Basic Audio Processing

Now that we know how to read audio data, the next step is to process it. There will be 3 main parts to this step.
Hint: Lecture 5 and 6 introduce related knowledge. Please read the slides and watch the lecture videos. 

1. **Filtering**: Apply 1 or more of these filters to a signal:
   1. Hamming Filter
   2. Low pass filter
   3. High pass filter
2. **Sampling**: Discrete Fast Fourier Transformation
3. **Normalizing** :  Logarithmic binning, Applying a Mel-Filterbank

Note: This section will be less hands-on than the other sections. I suggest experimenting with these steps first on your laptop/desktop (Try using a jupyter notebook! [Getting started](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/), [VSCode](https://code.visualstudio.com/docs/python/jupyter-support), [Online](colab.research.google.com)) and then moving your code to the raspberry pi (Use scp or git for this!). 

### Task 3.1: Filtering

#### Part 1: Activity

**Online References (Resources I used to prepare this material):**

1. [CS 434 FFT Foundations (Page 26)](http://croy.web.engr.illinois.edu/teaching/notes/1_Fourier_transform.pdf)
2. [Hamming Filter (Window Function) Wiki](https://en.wikipedia.org/wiki/Window_function)
3. [Hamming Filter Motivation](https://www1.udel.edu/biology/rosewc/kaap686/notes/windowing.html)
4. [numpy.hamming](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hamming.html)
5. [High Pass Filter and Low Pass Filter Wiki](https://en.wikipedia.org/wiki/Audio_filter)
6. [High pass and low pass filter Motivation](sites.music.columbia.edu/cmc/MusicAndComputers/chapter4/04_03.php)
7. [Butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter)

Your first task is to generate a band pass filter (The combination of a low pass filter and a high pass filter) to filter out. The main goal is to get an intuitive understanding of what a filter is doing.  You should be able to complete this activity using only the functions given below (but its okay to use other functions as well!).

**Documentation:**

1. [scipy.signal.butter](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html)
2. [scipy.signal.filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)
3. Any other functions in [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)

The following code generates this signal that is the summation of 3 sinusoidal wave. Your goal is to filter out the 5 Hz and the 100 Hz sinusoid so that only the 50 Hz signal remains.

```python
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

def filter_low_and_high(signal, order=2):
	# @TODO: implement this.
  

filtered_50_hz_sin = filter_low_and_high(sum_of_3_sin)  
plot_function(filtered_50_hz_sin, "Desired Output")
```

![index](./img/index.png)

![index2](./img/index2.png)



#### Part 2: Filtering Audio data

A big problem that arises from the use of an Fourier transform to get the frequencies of a real signal is something called **spectral leakage**. So, before we take the FFT of a signal, we need to apply a **hamming filter** to the signal (Read more about how hamming filters help with spectral leakage [here](http://digitalsoundandmusic.com/2-3-11-windowing-functions-to-eliminate-spectral-leakage/) and [here](http://saadahmad.ca/fft-spectral-leakage-and-windowing/)). 

Your goal for this part is to apply a hamming filter to the data that you read in from the PulseAudio stream you made in Task 2 (called `input data` from now on...). This consists of 3 tasks:

1. Reading in the  `input data` .
2. Constructing a hamming window (see documentation of numpy.hamming)
3. Multiplying each value in the hamming window with the input data (point wise).

## Task 3.2: DFT

**Online Resources:**

1. [CS 434 FFT Foundations](http://croy.web.engr.illinois.edu/teaching/notes/1_Fourier_transform.pdf)
2. [3blue1brown Youtube video](https://www.youtube.com/watch?v=spUNpyF58BY)
3. [numpy.fft](https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft)

A discrete Fourier transform takes an input signal and separates it into its discrete frequency components.

Now that we have prepared our filtered `input data`, take a 1 dimensional discrete Fourier transform of the same. (look at the [numpy.fft](https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft) documentation) and then plot it using the `plot_function` defined in the code for Task 3.1. Play around with this! Try to see if you can come up with answers for the following:

1. How does the DFT change if we add a very high frequency sinusoid to the data?
2. What would happen if we didn't apply the hamming filter in Task 3.1.2?

The DFT gives us the amplitude and phase of each frequency present in the signal. From this, we can compute an estimation for the [Power spectral density](https://en.wikipedia.org/wiki/Spectral_density) which can be calculated using this formulas:
$$
P(f_k) = \frac{1}{N} |DFT(f_{k})|^{2}
$$

### Task 3.3: Normalizing the DFT

Yay! Now that we have the frequency decomposition, we can start identifying the components that we might want to visualize.  Our final goal (for this task) is to bin the frequencies so that we can map colors to certain frequencies (red for high pitched vocals, violet for bass, etc...).

| Frequency Range | Frequency Values |
| --------------- | ---------------- |
| Sub-bass        | 20 to 60 Hz      |
| Bass            | 60 to 250 Hz     |
| Low midrange    | 250 to 500 Hz    |
| Midrange        | 500 Hz to 2 kHz  |
| Upper midrange  | 2 to 4 kHz       |
| Presence        | 4 to 6 kHz       |
| Brilliance      | 6 to 20 kHz      |

(table from [here](https://www.teachmeaudio.com/mixing/techniques/audio-spectrum/))

I'll leave the actual implementation of how to bin the data up to you. I'll use this section to explain two (out of many) ways in which you can go about this.

1. **Logarithmic binning**: One of the things that we notice when we take the log-log plot of the power spectrum output (Shown for some sample data in the figure below) is that the frequencies are distributed logarithmically (The number of frequencies in the 0-10Hz interval is half of the number of frequencies in the 10-100Hz and so on). Hence, we can group the frequencies in logarithmically increasing bins. One way of going about this is to split the array on the indices that are the geometric mean of the log intervals. Some pseudo-code for the following:

   ```python
   log_scale = generate_log_10_values(0, log(len(signal)))
   indices = [geometric_mean(log_scale[i], log_scale[i - 1]) for i from 1 to len(log_scale) - 1]
   split_on_indices(signal, indices)
   ```

   

   ![index2](./img/index3.png)

2. **Filter using a Mel-frequency filterbank**:

   Some resources to learn this:

   * [Mel Scale](https://en.wikipedia.org/wiki/Mel_scale)

   * [Getting to know the Mel Spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
   * [Calculating the mel frequency cepstrum coeff's](https://dsp.stackexchange.com/questions/6499/help-calculating-understanding-the-mfccs-mel-frequency-cepstrum-coefficients)

   The basic intuition behind this is that its easier for humans to differentiate between two signals oscillating at 500Hz and 1000Hz than to differentiate between two signals oscillating at 8000Hz and 8500Hz. Hence, we must filter our frequencies appropriately so that frequencies in the mel scale are binned together. All in all, it boils down to taking a dot product of our current power spectrum approximation and the `mel_matrix`. The following code will construct the `mel_matrix` for you:

   ```python
   # implementation adapted from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
   def hz_to_mel(hz):
       return 2595 * np.log10(1 + (hz / 2) / 700)
   
   def mel_to_hz(mel):
       return 700 * (10**(mel / 2595) - 1)
   
   def get_mel_filtermatrix(n_filters=24, fft_size=512, low_hz=0, high_hz=16000, sampling_rate=44100):
      low_mel = hz_to_mel(low_hz)
      high_mel = hz_to_mel(high_hz)
      points_in_mel = np.linspace(low_mel, high_mel, n_filters + 2) 
      points_in_hz = mel_to_hz(points_in_mel)
      center_freq = np.floor((size + 1) * points_in_hz / sampling_rate).astype(int)

      freq_to_mel_matrix = np.zeros((n_filters, int(np.floor(size / 2 + 1))))

      for i in range(1, len(center_freq) - 1):
         low_f = int(center_freq[i - 1])
         center_f = int(center_freq[i])
         high_f = int(center_freq[i + 1])

         inc_slope_idx = np.arange(low_f, center_f) # +ve triangle filter slope
         dec_slope_idx = np.arange(center_f, high_f)
         freq_to_mel_matrix[i - 1, inc_slope_idx] = (inc_slope_idx - low_f) / (center_f - low_f)
         freq_to_mel_matrix[i - 1, dec_slope_idx] = (high_f - dec_slope_idx) / (high_f - center_f)

      return freq_to_mel_matrix
   
   mel_matrix = get_mel_filtermatrix(N_fft_bins)
   ```

   

## Task 4: Visualization

At this point, we are done with all the heavy lifting. You should make sure that your code runs in less than a second to ensure that we can visualize our audio in real time. Our original goal was to output the bins we created in the last step to the LED strip using the Pi's GPIO pins. However, due to COVID-19, we shall instead use a GUI for the visualization phase. **You are welcome to use any software/language/framework in this task,** as long as you provide necessary citations, your code/logic is interpretable, and your choice of software can render the GUI in real time. (You may even print the FFT bins to the terminal). I shall be using  `matplotlib.animation` for this. The reason is two-fold:

1. It has enough documentation that all basic questions can be searched online.
2. It runs on my Pi 3B+ without too much lag.

If you're controlling your Pi via SSH, you can pass in the `-X` flag to view the plot/animation generated.

## Task 4.1 Bass Visualization (Activity)

**Online References**

1. [Animations in matplotlib](https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html)
2. [Lifecycle of a plot in matplotlib](https://matplotlib.org/3.2.1/tutorials/introductory/lifecycle.html)
3. [matplotlib.animation](https://matplotlib.org/3.2.1/api/animation_api.html)
4. [matplotlib.patches.Circle](https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.patches.Circle.html)
5. [Relation between FFT length and frequency resolution](https://electronics.stackexchange.com/questions/12407/what-is-the-relation-between-fft-length-and-frequency-resolution)

Your goal in this subtask is to map the frequencies that correspond to the "base" tones onto the radius of the circle. The table from Task 3.3 above tells us that bass tones usually lie within the 60 to 250 Hz range. If we extract the frequencies corresponding to the bass tones, then the energy in these frequencies (rudimentally) represents the presence of a beat in the signal. We can extract the frequencies in two ways: 

1. We can use a lowpass filter to filter out any frequency above 250 Hz. 
2. We can take the $n$-point discrete Fourier transform of the audio (after applying a hamming filter) and take the slice of the array that corresponds to the "bass" frequencies. (make sure that $n$ = sampling frequency in the call to `np.fft.rfft` to use the 60 Hz and 250 Hz numbers directly. Also, instead of energy, you need to calculate the [spectral energy](https://en.wikipedia.org/wiki/Energy_(signal_processing)) for the fft).

Here is some code to plot a circle in matplotlib to get your started. 

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure(figsize=(5,5))
ax = plt.axes(xlim=(0, 2), ylim=(0, 2))

base = plt.Circle((1, 1), 0.2, fc='b')

def init():
    ax.add_patch(base)
    return base

def loop(i):
    sample = microphone.get_sample() # <- @TODO replace this line with code to read a sample from the microphone.
    if sum(sample) > 0: 
        calc_base =  0.05 * np.random.rand() # <- @TODO replace this line with processed signal.
        base.set_radius(calc_base)
    return base


anim = animation.FuncAnimation(fig, loop, 
                               init_func=init, 
                               frames=10, 
                               interval=10,
                               blit=True)

plt.show()
```



## Task 4.2 Music Visualization

__Online References__:

1. [A Perceptually Meaningful Audio Visualizer](https://medium.com/@delu/a-perceptually-meaningful-audio-visualizer-ee72051781bc)
2. [An audio visualizer for Razer products](https://medium.com/schkn/building-an-audio-visualizer-for-razer-chroma-keyboards-7814cab950ff)

Get creative! We've already implemented (95% of) the tools that we need to make a rudimentary music visualizer. In this sub-task, your goal is to piece together all the different things we have touched on in the past (few) weeks and make something cool! However, there are 2 limitations:

1. Your visualization must be "real time" (Lag between audio and visualization shouldn't exceed 1.5 second max).
2. The audio processing must be done in real time (Shouldn't be reading a file with the Fourier transform of every sample of the mp3).

A couple of pointers if you're running into speed issues:

1. Try reducing the number of frames in `FRAMES_PER_BUFFER`. `FRAMES_PER_BUFFER := SAMPLING_FREQ / FPS`. So, to increase the FPS, we need to reduce the frames per buffer.
2. Try lowering the value of  `frames` and/or `interval`  in the call to `animation.FuncAnimation`.
3. Try rendering less objects!

Have fun!

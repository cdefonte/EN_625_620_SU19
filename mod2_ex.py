from __future__ import print_function, division

import thinkdsp
import thinkplot
import thinkstats2 

import numpy as np

import warnings
warnings.filterwarnings('ignore')

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

PI2 = np.pi * 2

%matplotlib inline

def make_sine(offset):
    signal = thinkdsp.SinSignal(freq=440, offset=offset)
    wave = signal.make_wave(duration=0.5, framerate=10000)
    return wave
	

wave1 = make_sine(offset=0)
wave2 = make_sine(offset=1)

thinkplot.preplot(2)
wave1.segment(duration=0.01).plot()
wave2.segment(duration=0.01).plot()
thinkplot.config(xlabel='Time (s)', ylim=[-1.05, 1.05])

wave1.corr(wave2)

def compute_corr(offset):
    wave1 = make_sine(offset=0)
    wave2 = make_sine(offset=-offset)
    
    thinkplot.preplot(2)
    wave1.segment(duration=0.01).plot()
    wave2.segment(duration=0.01).plot()
    
    corr = wave1.corr(wave2)
    print('corr =', corr)
    
    thinkplot.config(xlabel='Time (s)', ylim=[-1.05, 1.05])
	
slider = widgets.FloatSlider(min=0, max=PI2, value=1)
interact(compute_corr, offset=slider);

offsets = np.linspace(0, PI2, 101)

corrs = []
for offset in offsets:
    wave2 = make_sine(offset)
    corr = np.corrcoef(wave1.ys, wave2.ys)[0, 1]
    corrs.append(corr)
    
thinkplot.plot(offsets, corrs)
thinkplot.config(xlabel='Offset (radians)',
                 ylabel='Correlation', 
                 axis=[0, PI2, -1.05, 1.05])
				 
def serial_corr(wave, lag=1):
    N = len(wave)
    y1 = wave.ys[lag:]
    y2 = wave.ys[:N-lag]
    corr = np.corrcoef(y1, y2, ddof=0)[0, 1]
    return corr

	
signal = thinkdsp.UncorrelatedGaussianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)

signal = thinkdsp.BrownianNoise()
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)

signal = thinkdsp.PinkNoise(beta=1)
wave = signal.make_wave(duration=0.5, framerate=11025)
serial_corr(wave)

np.random.seed(19)

betas = np.linspace(0, 2, 21)
corrs = []

for beta in betas:
    signal = thinkdsp.PinkNoise(beta=beta)
    wave = signal.make_wave(duration=1.0, framerate=11025)
    corr = serial_corr(wave)
    corrs.append(corr)
    
thinkplot.preplot(1)
thinkplot.plot(betas, corrs)
thinkplot.config(xlabel=r'Pink noise parameter, $\beta$',
                 ylabel='Serial correlation', 
                 ylim=[0, 1.05])
				 
def autocorr(wave):
    """Computes and plots the autocorrelation function.

    wave: Wave
    """
    lags = range(len(wave.ys)//2)
    corrs = [serial_corr(wave, lag) for lag in lags]
    return lags, corrs
	
def plot_pink_autocorr(beta, label):
    signal = thinkdsp.PinkNoise(beta=beta)
    wave = signal.make_wave(duration=1.0, framerate=10000)
    lags, corrs = autocorr(wave)
    thinkplot.plot(lags, corrs, label=label)
	
np.random.seed(19)
thinkplot.preplot(3)

for beta in [1.7, 1.0, 0.3]:
    label = r'$\beta$ = %.1f' % beta
    plot_pink_autocorr(beta, label)

thinkplot.config(xlabel='Lag', 
                 ylabel='Correlation',
                 xlim=[-1, 1000], 
                 ylim=[-0.05, 1.05],
                 legend=True)

wave = thinkdsp.read_wave('28042__bcjordan__voicedownbew.wav')
wave.normalize()
wave.make_audio()

spectrum = wave.make_spectrum()
spectrum.plot()
thinkplot.config(xlabel='Frequency (Hz)', ylabel='Amplitude')

spectro = wave.make_spectrogram(seg_length=1024)
spectro.plot(high=4200)
thinkplot.config(xlabel='Time (s)', 
                 ylabel='Frequency (Hz)',
                 xlim=[wave.start, wave.end])

duration = 0.01
segment = wave.segment(start=0.2, duration=duration)
segment.plot()
thinkplot.config(xlabel='Time (s)', ylim=[-1, 1])

spectrum = segment.make_spectrum()
spectrum.plot(high=1000)
thinkplot.config(xlabel='Frequency (Hz)', ylabel='Amplitude')

len(segment), segment.framerate, spectrum.freq_res

def plot_shifted(wave, offset=0.001, start=0.2):
    thinkplot.preplot(2)
    segment1 = wave.segment(start=start, duration=0.01)
    segment1.plot(linewidth=2, alpha=0.8)

    # start earlier and then shift times to line up
    segment2 = wave.segment(start=start-offset, duration=0.01)
    segment2.shift(offset)
    segment2.plot(linewidth=2, alpha=0.4)

    corr = segment1.corr(segment2)
    text = r'$\rho =$ %.2g' % corr
    thinkplot.text(segment1.start+0.0005, -0.8, text)
    thinkplot.config(xlabel='Time (s)', xlim=[start, start+duration], ylim=[-1, 1])

plot_shifted(wave, 0.0001)

end = 0.004
slider1 = widgets.FloatSlider(min=0, max=end, step=end/40, value=0)
slider2 = widgets.FloatSlider(min=0.1, max=0.5, step=0.05, value=0.2)
interact(plot_shifted, wave=fixed(wave), offset=slider1, start=slider2)
None

wave = thinkdsp.read_wave('28042__bcjordan__voicedownbew.wav')
wave.normalize()
duration = 0.01
segment = wave.segment(start=0.2, duration=duration)

lags, corrs = autocorr(segment)
thinkplot.plot(lags, corrs)
thinkplot.config(xlabel='Lag (index)', ylabel='Correlation', ylim=[-1, 1])

low, high = 90, 110
lag = np.array(corrs[low:high]).argmax() + low
lag

period = lag / segment.framerate
period

frequency = 1 / period
frequency

segment.framerate / 102, segment.framerate / 100

N = len(segment)
corrs2 = np.correlate(segment.ys, segment.ys, mode='same')
lags = np.arange(-N//2, N//2)
thinkplot.plot(lags, corrs2)
thinkplot.config(xlabel='Lag', ylabel='Correlation', xlim=[-N//2, N//2])

N = len(corrs2)
lengths = range(N, N//2, -1)

half = corrs2[N//2:].copy()
half /= lengths
half /= half[0]
thinkplot.plot(half)
thinkplot.config(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])

thinkplot.preplot(2)
thinkplot.plot(half)
thinkplot.plot(corrs)
thinkplot.config(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])

diff = corrs - half[:-1]
thinkplot.plot(diff)
thinkplot.config(xlabel='Lag', ylabel='Difference in correlation')

	 
	
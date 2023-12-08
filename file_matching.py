from collections import defaultdict
import csv
import math
import json
import os
import sys
import webbrowser

import librosa

import numpy as np
from scipy.fft import fft, fftfreq
from scipy import fft, signal
import scipy
from scipy.io.wavfile import read

def open_video_player(video, frame):
    video = video[0:-3]
    video += 'mp4'
    full_url = f"https://liu-zy-98.github.io/vidplayer/?video=videos/{video}&start={frame}"
    print(full_url)
    # Open the default browser to the full URL
    webbrowser.open(full_url)

def find_offset(within_file, find_file, window):
    y_within, sr_within = librosa.load(within_file, sr=4410)
    y_find, _ = librosa.load(find_file, sr=sr_within)

    c = signal.correlate(y_within, y_find[:sr_within*window], mode='valid', method='fft')
    peak = np.argmax(c)
    offset = round(peak / sr_within, 2)

    # fig, ax = plt.subplots()
    # ax.plot(c)
    # fig.savefig("cross-correlation.png")

    return offset * 30

def get_residual_clip(audio1, audio2):
    if len(audio1) != len(audio2):
        raise ValueError("Invalid comparison. Not same length")
    
    residual = 0
    for i in range(len(audio1)):        
        residual += sum([abs(audio1[i][j] - audio2[i][j]) for j in range(len(audio1[i]))])
    return residual 

def get_residual_total(input, full):
    n, m = len(input), len(full)

    min_residual = math.inf
    min_index = 0
    for index in range(m - n + 1):
        residual = get_residual_clip(input, full[index : index + n])
        if residual < min_residual:
            min_residual = residual 
            min_index = index
    return min_residual, min_index

def print_match(querypath, input):
    
    min_residual = math.inf
    match = None
    min_index = 0

    for file, full_audio in db.items():
        residual, index = get_residual_total(input, full_audio)
        if residual < min_residual:
            match = file 
            min_residual = residual 
            min_index = index
    

    print(match, min_residual, min_index)
    frame = find_offset('Data/Audio/Audios_DB/' + match, querypath, 5)
    print('frame: ' + str(int(frame)))
    open_video_player(match, frame)

def create_constellation(audio, Fs):
    # Parameters
    window_length_seconds = 2

    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 15
    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples
    song_input = np.pad(audio, (0, amount_to_pad))
    # Perform a short time fourier transform
    frequencies, times, stft = signal.stft(
        song_input, Fs, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True
    )
    constellation_map = defaultdict(list)
    for time_idx, window in enumerate(stft.T):
        # Spectrum is by default complex. 
        # We want real values only
        spectrum = abs(window)
        # Find peaks - these correspond to interesting features
        # Note the distance - want an even spread across the spectrum
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        # Only want the most prominent peaks
        # With a maximum of 15 per time slice
        n_peaks = min(num_peaks, len(peaks))
        # Get the n_peaks largest peaks from the prominences
        # This is an argpartition
        # Useful explanation: https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map[time_idx].append(frequency)
            
    return constellation_map

if __name__ == "__main__":


    with open("Data/Audio/db.json", 'r') as file:
        db = json.load(file)
    
    Fs, song = read(sys.argv[2])
    song = np.transpose(np.transpose(song)[0])
    input = create_constellation(song, Fs)

    print_match(sys.argv[2], input)


    


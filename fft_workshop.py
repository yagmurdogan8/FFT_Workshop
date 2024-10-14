import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Task 1 (Assignment 1)

y, sr = librosa.load('piano.wav')
D = librosa.stft(y) # short time fourier transdorm of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure()
librosa.display.specshow(S_db)
plt.colorbar()
plt.show()


# Task 2 Part 1:

# parameters as stated in the instructions pdf
# uncomment for task 2 part 1 :

# window_size = 512
# n_bands = 8

# uncomment for task 2 part 2 :

window_size = 256
n_bands = 16

band_limits = [(i * 1000, (i + 1) * 1000) for i in range(n_bands)]  # frequency bands as stated in the instructions pfd

# calculate the band energy for each window
def calculate_band_energy(y, sr, band_limits):
    # short time fourier transdorm
    stft = np.abs(librosa.stft(y, n_fft = window_size, hop_length = window_size//2))
    # frequencies corresponding to FFT bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=window_size)
    
    # calculate energy for each band
    energies = []
    for start, end in band_limits:
        band_indices = np.where((freqs >= start) & (freqs < end))[0]
        band_energy = np.sum(stft[band_indices, :], axis=0)
        energies.append(band_energy)
    
    return np.array(energies).T  # had to transpose so each row corresponds to a window

# calculate the energy vectors and save them
energies = calculate_band_energy(y, sr, band_limits)
np.savetxt('piano_energies.txt', energies, fmt='%.5f')  # document that to piano_energies.txt

# Task 2 part 2:

# encode pitch types (up, down and repeat)
def encode_pitches(energies):
    pitches = []
    last_max_band = None
    repeat_count = 0
    
    for i, window_energy in enumerate(energies):
        max_band = np.argmax(window_energy)
        
        if last_max_band is None:
            pitches.append('Start')
        elif max_band > last_max_band:
            pitches.append('U')
            repeat_count = 0
        elif max_band < last_max_band:
            pitches.append('D')
            repeat_count = 0
        else:
            repeat_count += 1
            pitches.append(f'R{repeat_count}')
        
        last_max_band = max_band
    
    return pitches

pitches = encode_pitches(energies)

# final pitches
print("\nFinal Pitches:")
print("\n".join(pitches))
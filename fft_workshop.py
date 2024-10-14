import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Task 1 (Assignment 1)

y, sr = librosa.load('piano.wav')
D = librosa.stft(y) # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure()
librosa.display.specshow(S_db)
plt.colorbar()
plt.show()

# y, sr = librosa.load('piano.wav', sr=16000)


# Task 2 Part 1:

window_size = 512
hop_length = window_size // 2  # 50% overlap
n_fft = window_size
n_bands = 8  # Number of frequency bands
max_freq = 8000  # Maximum frequency for band divisions

# Compute STFT
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
D_mag = np.abs(D) ** 2  # Magnitude squared for power

# Define the frequency bins for the bands
freqs = np.fft.fftfreq(n_fft, d=1/sr)[:n_fft//2]
band_edges = np.linspace(0, max_freq, n_bands + 1)

# Initialize list to store 8-dimensional energy vectors
energy_vectors = []

# Iterate over each window of the STFT
for frame in range(D_mag.shape[1]):
    # Calculate the energy for each band
    energy_band = []
    for i in range(n_bands):
        # Determine the frequency range for this band
        low_freq, high_freq = band_edges[i], band_edges[i + 1]
        # Find indices corresponding to this frequency range
        band_indices = np.where((freqs >= low_freq) & (freqs < high_freq))[0]
        # Sum the energies in this band
        band_energy = np.sum(D_mag[band_indices, frame])
        energy_band.append(band_energy)
    # Add this 8-dimensional vector to the results
    energy_vectors.append(energy_band)

# Write the results to a file
with open('piano_energies.txt', 'w') as f:
    for vec in energy_vectors:
        f.write(' '.join(map(str, vec)) + '\n')
        
# Task 2 part 2:

# Function to find max-energy band index
def max_energy_band(energy_frame):
    band_energies = []
    for i in range(n_bands):
        # Get indices for the current band
        low_freq, high_freq = band_edges[i], band_edges[i + 1]
        band_indices = np.where((freqs >= low_freq) & (freqs < high_freq))[0]
        # Calculate energy in this band
        band_energy = np.sum(energy_frame[band_indices])
        band_energies.append(band_energy)
    return np.argmax(band_energies), band_edges[np.argmax(band_energies)]  # Return max band index and avg freq

# Initialize the results list and repeat counter
results = []
repeat_count = 0
prev_band_index = None

# Process each window
for frame in range(D_mag.shape[1]):
    # Calculate max-energy band for the current frame
    band_index, band_avg_freq = max_energy_band(D_mag[:, frame])
    
    # Compare with the previous window's max-energy band
    if prev_band_index is None:
        # First window, no comparison
        results.append('U')
    elif band_index > prev_band_index:
        results.append('U')
        repeat_count = 0
    elif band_index < prev_band_index:
        results.append('D')
        repeat_count = 0
    else:
        # If the same band, increment repeat counter and note 'Rx'
        repeat_count += 1
        results[-1] = f'R{repeat_count}' if results and 'R' in results[-1] else f'R1'
    
    # Update previous band index
    prev_band_index = band_index

# Display the encoded pitch tendency sequence
print(' '.join(results))
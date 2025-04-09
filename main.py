# import sys
from playsound import playsound
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment

def play_sound(file_path):
    try:
        print(f"Playing '{file_path}' ...")
        playsound(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def show_spectrogram(file_path):
    # Read the audio file
    if file_path.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
        data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
    else:
        sample_rate, data = wavfile.read(file_path)

    # Create a spectrogram
    plt.specgram(data, Fs=sample_rate)
    plt.title('Spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Intensity [dB]')
    plt.show()

def show_transform(file_path):
    # Read the audio file
    if file_path.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
        data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
    else:
        sample_rate, data = wavfile.read(file_path)

    # If stereo, average the channels
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Compute the Fourier Transform
    fft_vals = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), d=1 / sample_rate)

    # Consider only the positive frequencies
    pos_mask = fft_freq >= 0

    plt.figure()
    plt.plot(fft_freq[pos_mask], np.abs(fft_vals[pos_mask]))
    plt.title('Fourier Transform')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.show()

def main():
    audio_file = '/Users/jackvu/Desktop/latex projects/pde/echo_cancel/wilhelm.mp3'
    
    # Play the audio
    play_sound(audio_file)
    
    # Show spectrogram visualization
    show_spectrogram(audio_file)
    
    # Show Fourier transform visualization
    show_transform(audio_file)
    
    print("Playback finished.")

if __name__ == '__main__':
    main()

import librosa
import numpy as np

for i in range(1,11):
    print(i)

# Obtaining Audio Files
    
audio1 = input('Enter the path of the first audio file: ')
audio2 = input('Enter the path of the second audio file: ')
y1, sr1 = librosa.load(audio1, sr=44100)
y2, sr2 = librosa.load(audio2, sr=44100)

# The tempo function returns the estimated tempo (beats per minute) for a given onset strength envelope.
# The onset strength envelope is a vector that stores the strength of the onsets at each time step.
# onset strength represents 

y1_onset_env = librosa.onset.onset_strength(y=y1, sr=sr1)
y1_tempo = librosa.feature.tempo(onset_envelope=y1_onset_env, sr=sr1)

y2_onset_env = librosa.onset.onset_strength(y=y2, sr=sr2)
y2_tempo = librosa.feature.tempo(onset_envelope=y2_onset_env, sr=sr2)

print('Tempo of the first audio file: ', y1_tempo)
print('Tempo of the second audio file: ', y2_tempo)

# Comparing the Features
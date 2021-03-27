
import librosa
import matplotlib.pylab as plt

# filename = '/Users/satyakimallick/IdeaProjects/Awaves_Energy_Levels/dataset/2gb_set/Acidrave - Em - 110 - Yellow Claw - Reckless (Dirty).mp3'
filename = '/Users/satyakimallick/IdeaProjects/Awaves_Energy_Levels/dataset/2gb_set/Drumstep - Dbm - 80 - 4B & Nvrleft - DOPE (Main).mp3'
#filename = '/Users/satyakimallick/IdeaProjects/Awaves_Energy_Levels/wav files/Basshouse - Abm - 75 - Chamillionaire.wav'

#filename = librosa.load(path)

#filename = librosa.example('nutcracker')

y, sr = librosa.load(filename)
# plt.plot(y, 'audio', 'time', 'amplitude')

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

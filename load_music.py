import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

mp3_files_folder = "C:/Users/rogie/Desktop/Music Database"

def load_song(path_to_file):

    example_mp3, sr = librosa.load(path_to_file)

    x = path_to_file.split("/")

    return example_mp3, sr, x[-1]

def load_song_offset(number: int):

    mp3_files = os.listdir(mp3_files_folder)

    path_to_file = mp3_files_folder + '/' + mp3_files[number]

    example_mp3, sr = librosa.load(path_to_file, offset=10)

    return example_mp3, sr, mp3_files[number]


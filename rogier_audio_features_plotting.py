import librosa
import librosa.display
import matplotlib.pyplot as plt
import load_music
import numpy as np


def plot_chroma_stft_cqt(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    chroma_cqt = librosa.feature.chroma_cqt(example_mp3, sr)
    chroma_stft = librosa.feature.chroma_stft(example_mp3, sr)


    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax[0])
    ax[0].set(title='chroma_stft')
    ax[0].label_outer()
    fig.suptitle(song_name)
    img = librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time', ax=ax[1])
    ax[1].set(title='chroma_cqt')
    fig.colorbar(img, ax=ax)
    plt.show()


def plot_chroma_cens(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    chroma_cens = librosa.feature.chroma_cens(example_mp3, sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time', ax=ax)
    fig.suptitle("Chroma_CQ on" + " " + song_name, fontsize=8)
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_mel_frequency_spectogram(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(example_mp3, sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.suptitle('Mel-frequency spectogram on' + ' ' + song_name, fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_mfcc(number):
    example_mp3, sr, song_name = load_music.load_song(number)
    example_mp3 = example_mp3[:220500]
    fig, ax = plt.subplots()
    mfccs = librosa.feature.mfcc(example_mp3, sr)
    print(mfccs.shape)
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    fig.colorbar(img, ax=ax)
    fig.suptitle('MFCC on' + ' ' + song_name, fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_rms(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    S, phase = librosa.magphase(librosa.stft(example_mp3))
    rms = librosa.feature.rms(S=S)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='loga Power spectrogram')
    fig.suptitle('RMS on' + ' ' + song_name, fontsize=8)
    plt.show()


def plot_spectral_centroid(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    cent = librosa.feature.spectral_centroid(example_mp3, sr)
    S, phase = librosa.magphase(librosa.stft(y=example_mp3))

    fig, ax = plt.subplots()
    times = librosa.times_like(cent)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    fig.suptitle('log power spectogram on' + ' ' + song_name, fontsize=8)
    plt.show()


def plot_spectral_bandwidth(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    fig, ax = plt.subplots(nrows=2, sharex=True)

    spec_bw = librosa.feature.spectral_bandwidth(y=example_mp3, sr=sr)
    S, phase = librosa.magphase(librosa.stft(y=example_mp3))

    times = librosa.times_like(spec_bw)
    centroid = librosa.feature.spectral_centroid(S=S)

    ax[0].semilogy(times, spec_bw[0], label='Spectral bandwidth')
    ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')
    ax[1].fill_between(times, centroid[0] - spec_bw[0], centroid[0] + spec_bw[0],
                       alpha=0.5, label='Centroid +- bandwidth')
    ax[1].plot(times, centroid[0], label='Spectral centroid', color='w')
    ax[1].legend(loc='lower right')
    fig.suptitle('Spectral bandwidth on ' + song_name, fontsize=7)
    plt.show()


def plot_spectral_contrast(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    fig, ax = plt.subplots(nrows=2, sharex=True)

    S = np.abs(librosa.stft(example_mp3))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    img1 = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                            ref=np.max),
                                    y_axis='log', x_axis='time', ax=ax[0])
    fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()
    img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
    fig.colorbar(img2, ax=[ax[1]])
    fig.suptitle('Spectral contras on ' + song_name, fontsize=8)
    ax[1].set(ylabel='Frequency bands', title='Spectral contrast')
    plt.show()


def plot_spectral_flatness(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    flatness = librosa.feature.spectral_flatness(y=example_mp3)
    plt.plot(range(0, len(flatness[0])), flatness[0])
    plt.title('Spectral Flatness on ' + song_name)
    plt.show()


def plot_spectral_rolfoff(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    fig, ax = plt.subplots()
    rolloff = librosa.feature.spectral_rolloff(y=example_mp3, sr=sr, roll_percent=0.99)
    rolloff_min = librosa.feature.spectral_rolloff(y=example_mp3, sr=sr, roll_percent=0.01)

    S, phase = librosa.magphase(librosa.stft(example_mp3))

    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)
    ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
    ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
            label='Roll-off frequency (0.01)')
    ax.legend(loc='lower right')
    ax.set(title='log Power spectrogram')
    fig.suptitle('Spectral rolfoff on ' + song_name, fontsize=8)
    plt.show()


def plot_poly_features(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    S = np.abs(librosa.stft(example_mp3))

    p0 = librosa.feature.poly_features(S=S, order=0)
    p1 = librosa.feature.poly_features(S=S, order=1)
    p2 = librosa.feature.poly_features(S=S, order=2)

    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
    times = librosa.times_like(p0)
    ax[0].plot(times, p0[0], label='order=0', alpha=0.8)
    ax[0].plot(times, p1[1], label='order=1', alpha=0.8)
    ax[0].plot(times, p2[2], label='order=2', alpha=0.8)
    ax[0].legend()
    ax[0].label_outer()
    ax[0].set(ylabel='Constant term ')
    ax[1].plot(times, p1[0], label='order=1', alpha=0.8)
    ax[1].plot(times, p2[1], label='order=2', alpha=0.8)
    ax[1].set(ylabel='Linear term')
    ax[1].label_outer()
    ax[1].legend()
    ax[2].plot(times, p2[0], label='order=2', alpha=0.8)
    ax[2].set(ylabel='Quadratic term')
    ax[2].legend()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[3])
    fig.suptitle('Poly features on ' + song_name, fontsize=8)
    plt.show()


def plot_tonnetz(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    harmonic = librosa.effects.harmonic(example_mp3)
    tonnetz = librosa.feature.tonnetz(harmonic, sr)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    img1 = librosa.display.specshow(tonnetz,
                                    y_axis='tonnetz', x_axis='time', ax=ax[0])
    ax[0].set(title='Tonal Centroids (Tonnetz)')
    ax[0].label_outer()
    img2 = librosa.display.specshow(librosa.feature.chroma_cqt(harmonic, sr=sr),
                                    y_axis='chroma', x_axis='time', ax=ax[1])
    ax[1].set(title='Chroma')
    fig.colorbar(img1, ax=[ax[0]])
    fig.colorbar(img2, ax=[ax[1]])
    fig.suptitle('Tonnetz on ' + song_name, fontsize=8)
    plt.show()


def plot_zero_crossing_rate(number):
    example_mp3, sr, song_name = load_music.load_song(number)

    fig, ax = plt.subplots()
    zcr = librosa.feature.zero_crossing_rate(example_mp3)
    plt.plot(range(0, len(zcr[0])), zcr[0])
    fig.suptitle('ZCR on ' + song_name, fontsize=8)
    plt.show()


def plot_tempogram(number):
    example_mp3, sr, song_name = load_music.load_song(number)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=example_mp3, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                          hop_length=hop_length)
    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    # Estimate the global tempo for display purposes
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                               hop_length=hop_length)[0]

    fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
    times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
    ax[0].plot(times, oenv, label='Onset strength')
    ax[0].label_outer()
    ax[0].legend(frameon=True)
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='tempo', cmap='magma',
                             ax=ax[1])
    ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
                label='Estimated tempo={:g}'.format(tempo))
    ax[1].legend(loc='upper right')
    ax[1].set(title='Tempogram')
    x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
                    num=tempogram.shape[0])
    ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    ax[2].set(xlabel='Lag (seconds)')
    ax[2].legend(frameon=True)
    freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                 label='Mean local autocorrelation', basex=2)
    ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                 label='Global autocorrelation', basex=2)
    ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,
                label='Estimated tempo={:g}'.format(tempo))
    ax[3].legend(frameon=True)
    ax[3].set(xlabel='BPM')
    ax[3].grid(True)
    fig.suptitle('Tempogram on ' + song_name, fontsize=8)

    plt.show()


def plot_fourier_tempogram(number):
    example_mp3, sr, song_name = load_music.load_song(number)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=example_mp3, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
                                                  hop_length=hop_length)
    # Compute the auto-correlation tempogram, unnormalized to make comparison easier
    ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                             hop_length=hop_length, norm=None)

    fig, ax = plt.subplots(nrows=3, sharex=True)
    ax[0].plot(librosa.times_like(oenv), oenv, label='Onset strength')
    ax[0].legend(frameon=True)
    ax[0].label_outer()
    librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='fourier_tempo', cmap='magma',
                             ax=ax[1])
    ax[1].set(title='Fourier tempogram')
    ax[1].label_outer()
    librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='tempo', cmap='magma',
                             ax=ax[2])
    ax[2].set(title='Autocorrelation tempogram')
    fig.suptitle('Fourier Tempogram on ' + song_name, fontsize=8)

    plt.show()


for i in range(0, 10):
    plot_fourier_tempogram(i)


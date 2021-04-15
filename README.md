# Song_Energy_Levels



## Update - 15th April, 2020
The current model has improved and gives 95% accuracy on Javyskyla dataset right now. The model is trained to give a integer number between 1 and 6 in a window of 10 seconds of a song with 1 represents low energy and 6 represents highest energy. Tested on:

Chopin - Nocturne op.9 No.2 (Timestamp 0:00 to 0:10 secs) - Model gives a score of 1

Don't You Worry Child (Time stamp 3:35 to 3:45) - Model gives a score of 5.


## Update - 10th April, 2020
Instead of having binary labels between a sad and a happy song, what if the songs ouput a number which if greater than a threshold, is a happy song, and otherwise is a sad song.



## Abstract
When looking at the general music build-up of a party, people usually refer to concepts like ‘flow’ and ‘energy’. Usually, professional DJ’s use the concept of energy to slowly activate their crowd and to reach a climax when the energy of the music and crowd are high. However, in scientific literature the concept of energy is ill defined. On the contrary, in practice, multiple music software tools use it to provide a score on energy level for their users. Multiple outlets describe how DJs should alter their playlists to accommodate energy levels (e.g. MixedInKey)
Therefore this assignment focusses upon defining and extracting the energy level for use within the virtual DJ environment. The main result should be an energy level score which could be implemented within the DJ recommender and to use it as a variable to mix songs upon. However, to get to such a score, we first need to define what energy is in music terms.

### Available Definitions
As definitions are unavailable in scientific literature, we look for practical explanations. MixedInKey does not explicitly defines energy levels, but scales them from level 1 until level 10. With level 1 representing ‘boring’ songs that do not have a beat and would be more in the chillout genre. Level 10 songs are highly energetic, have huge drops and are the climax of a set.

The energy level per song is available in the Spotify API. Mixingguide states that energy is not defined by only the amplitude or BPM of a track. So it is highly likeable that energy or energy level is a compounded variable containing multiple independent variables.


## Progress
So far I have a CNN model which is applied on the mel spectrograms of 360 songs from the Jyvaskyla dataset. The model performs poorly (53%) accuracy at the moment and requires improvement.

## Things tried so far
* I have tried using 1D CNN and 2D CNN on the mel-spectrogram (1D CNN by reducing one of the dimensions of the spectrogram) with no significant result.
* Trying out only dense layer based models
* Trying out only binary classification to classify happy songs against sad songs.
* Trying out MFCC based model


## Outlook
* Using Temporal CNN
* Trying other different other kind of spectrograms to see if they give significant results
* Trying out datasets other that the Jyvaskyla dataset
* Trying out LSTMs
* Trying out Emotion AI models




Google Colab notebook - https://colab.research.google.com/drive/1vUPBy2Gp_kdt6LAP0fVooGqEGRlFMwBT?usp=sharing

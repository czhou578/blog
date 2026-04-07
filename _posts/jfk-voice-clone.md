
I decided to create a system that would train on a corpus of JFK's speeches from his term in office, and then have that replicated voice read various speeches, lyrics, and
more. This has been a project dream of mine for many years.

## Technologies

I first had to pick out the tech stack I was going to use. Python was an easy choice for ML purposes. In terms of the finetuning model, I could have developed this myself, but I
decided to use the F5 TTS open source voice cloning model from GitHub. The reason why was that through Reddit comments and my online research, many people mentioned this as an ideal 
choice. I wasn't willing to pay for ElevenLabs or a third party proprietary API. 

For GPU, I rented an RTX 4000 Ada GPU on Runpod for about $0.27/hr. I was debating whether or not to use the more powerful RTX 4090 or RTX 5090, but as it turned out, through my
experiments, the budget GPU actually performed very well in terms of inference time. I spent a total of approximately $2 to fully finish this project, which I'm very happy about.

## Data Cleaning

In order to get the voice of JFK to be as good as possible, I found a 4 hour clip of JFK's speeches from various events on YouTube (yes, that does exist), and I converted it to a large mp3
file. This took a while since a lot of online YouTube to mp3 converters don't accept clips of that long. I ended up using the `yt-dlp` library from Python, which converted the
massive mp3 file into 16 kHz mono WAV. 

Then, the downloaded audio went through the following pipeline:

### Denoising
In the original clip, because the recordings were made in a live environment, there were many instances of clapping, background noise, and other disturbances. To clean these
out of the final chunked training audio, the massive clip was fed through the `resemble` library in Python, which performs the audio cleaning in conjunction with the `torchaudio`
library. 

### Transcription and Segmentation

Next was the transcription, since the training process needed labels of the spoken audio for validation purposes. For this step, I utilized the WhisperX model to apply VAD,
transcribe with timestamps, and return the list of segments, which are just the start and end timestamps, and the text spoken in between these two. The VAD removes the silences
and the non-speech segments.

### Slice and Export

Finally, I called a function to slice and export the audio and the csv file holding the metadata, including transcriptions. This ensured a nice result.

## Finetuning

For the finetuning step, I asked Claude to write and then modify a script in order to perform the training. I first cloned the F5 GitHub repo to my project and added it as a 
git submodule. I then created a virtual environment in the project and installed all my dependencies like PyTorch and all the libraries needed to run my local scripts and the
F5 module. I then prepared the dataset, copying all the chunked .wav files to the correct location as expected by F5, and also made sure that if this was the first time that
I pulled from remote repo, that there was actual data in the wav folder and the csv files, and not their respective Git submodule pointers.

I then converted metadata.csv + wavs into raw.arrow and duration.json, which the F5 library needs for efficient training.

I then modified the accelerate config so that the accelerate library specifically would be ready for the training.

Then, I officially launched the finetuning, which would perform the specified epochs, using the hyperparameters defined at the top of the file, and then save checkpoints every
so often until the training is done. 

## Inference

I then ran the `inference.py` which is a script that would create a .wav file in the outputs folder based on the command line or text file arguments with the speech that
is going to be read. You can select the wav file that would serve as the reference audio clip, and then update the corresponding reference text with the text from that clip.

## Problems Encountered

## Takeaways

## Conclusion

# STEP 4: Run the official dataset preparation script
# This converts metadata.csv + wavs into raw.arrow + duration.json
# which F5-TTS needs internally for efficient training



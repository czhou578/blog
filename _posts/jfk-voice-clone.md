
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

At the end, I uploaded several of the checkpoints that I used to HuggingFace, since Git charges quite an exorbitant amount of money to store files in its large file storage system. This proved useful later, as I decided to use one of my saved checkpoints (which had the lowest loss) to generate all of the final .wav files.

## Running Trials and Experiments

I asked Claude to give me a simple but realistic plan for me to improve on my trial results. It proposed a set of experiments relating to batch size, number of epochs, and the learning rate. I ran all of the experiments except for one, which was increasing the batch size to 8000 frames (afraid that my 20GB of VRAM wouldn't be enough, and it wasn't worth switching setups). 

## Problems Encountered

There were numerous challenges that I had to overcome to get the final result. Here is a list:

1. Python Environments: Ah, yes, this should be self explanatory. Every time that I got a new VM on Runpod and had to install my environment, it would take at least 15-20 minutes for all of the dependencies to install. This was a huge time sink, and I'm not sure how to exactly improve on this or speed it up.
2. Checkpoints: Each checkpoint file generated was about 5GB, which quickly ate up all of my VM storage. Thankfully, Huggingface made this easier because of their generous
limits on storage. But I also had several trial runs invalidated since they used previous checkpoints from previous runs, thus not doing anything. I ended up having to remove all the checkpoints later on to allow trials to truly train.
3. When I originally made my .txt file copy of Taylor Swift's song Blank Space, the model voice speed read the entire file without much pausing, and also inserted random
words like "government" into the recording. I found out through some AI-assisted debugging that the speed reading was due to the way the segmentation was working on the inference side.

The lyrics that I copy and pasted from online didn't have ending punctuation, so the segmentation was treating the entire half of the song as one line. Obviously, I wanted it to be one line, one synthesis. That required changing the behavior of the script to account for this. In addition, the insertion of the word "government" was due to the referenced text variable in my inference script not completely matching the real audio transcription of the referenced audio clip. 

## Takeaways

Also, I did test the speech on a very clear audio clip of JFK, without any microphone noises, or missed audio. For some reason, it didn't sound quite like him to me. But keep in mind that the microphone quality back in the 1960's weren't the best, but those recordings are all that we have. So when the audio quality improved a lot, it would
sound very different to the ear. 

## Conclusion

I had a lot of fun with this project, and was able to have a JFK-like voice read content that was both funny and humorous. I am impressed with how open source TTS software
can accomplish in contrast with the more premiumm and private third party offerings. I have included all of my source code (here)[https://github.com/czhou578/jfk-voice-clone/tree/master]. 


---
layout: post
title: "Building a Local Voice Agent on CPU"
date: 2026-04-05
---

I have seen online the growing prevalence of people running local AI models on their personal devices and interacting with them through voice. For a long time, I ignored the hype simply because I didn't believe that my 2022 Lenovo Thinkpad X1 Carbon PC could be capable of running any AI model on a usefulness basis.

But after seeing the new Alibaba Qwen 3.5 family of models drop and seeing people running it on their phones, I became intrigued at how my PC could run it, especially since I do have 16 GB of RAM and 1 TB of SSD storage. Here was my experience attempting to create a voice agent that would leverage Qwen to perform basic browser operations like opening up and replaying YouTube videos, launching search queries, and controlling my browser like an agent. I used pure Python for development.

# Setup

I started by installing Ollama, which is a tool that allows you to run large language models locally. I installed it on my Windows PC using their official installer in PowerShell. 

Next, I installed the Qwen 3.5 model using the following command:

```bash
ollama run qwen3.5:4b
```

I chose Ollama because according to Grok, it is one of the most popular tools for running local AI models and it is tailored to developers. 

The actual install for Qwen 3.5 didn't take very long, and I was able to open its console up in my Windows terminal and send commands to it pretty easily. I averaged around 14 tokens per second, which on my Intel CPU, wasn't as bad as I expected. My goal was just to have the tokens/sec be sufficient enough that there wouldn't be too much latency. My priority wasn't to have the model write essays, but to execute short and succinct commands. 

My goal was to create a fully end to end voice agent, that would take in my spoken input, translate late it into text, get the response from Qwen, and then recite it back to me. 

# Agent Evolution

I started off using pyttsx3 for text to speech, since it was recommended by Grok as a popular choice. I also began with SpeechRecognition, which is a Python library for speech recognition and a wrapper around many major speech recognition APIs from Google and more. Google's Web Speech API happened to be a free service to use so I went with that. 

The problem was that the Google Speech API does have a rate limit which could be exceeded, prompting errors when calling it programmatically, and also requires internet use, which I didn't like because I preferred something that I could run locally. 

So I switched over to Vosk, which was suggested by Claude after some prompting. 

Vosk is a local toolkit for speech recognition that supports over 20 languages, runs on CPU, and are small in terms of model size (50mb according to their documentation). That ended up working just as well with not much latency. 

# Browser Automation

I had some previous knowledge of using frameworks like Playwright to automate the browser, so I ended up integrating that into my project. Structurally, I added the browser logic as a "skill" into my project, where there was a global class called BrowserManager which contained all the methods for the browser automation, like the initialization lifecycle. It also contains the methods for Playwright to perform operations in the browser like navigate_to, which opens a new tab and navigates to a URL. 

I also had to add a system prompt for Qwen in order to help it understand the browser task. It basically told Qwen to output either [SEARCH] or [Navigate] at the beginning of each answer that is brower related. Based on which one it returns, a different Playwright method would be called. 

Here is what the prompt looked like:

```
SYSTEM_PROMPT = """You are a helpful voice assistant named Qwen. 
You must strictly follow these exact command formats for actions:

- To search Google: [SEARCH] query here
- To search YouTube: [YOUTUBE] query here
- To restart a YouTube video: [YOUTUBE_REPLAY]
- To play the first YouTube video result: [YOUTUBE_CLICK_FIRST]
- To open a website: [NAVIGATE] example.com

Example 1:
User: "Search for cat videos on YouTube"
Assistant: [YOUTUBE] cat videos

Example 2:
User: "Go to reddit"
Assistant: [NAVIGATE] reddit.com

Example 3:
User: "How are you today?"
Assistant: I am doing well, thank you!

For general questions, answer verbally in 1-2 short, natural sentences without abbreviations. 
IMPORTANT RULE: When you output a command, you MUST NOT output any other text. Output literally ONLY the command format. Do NOT invent new brackets."""

```

# Improving TTS And Running the Agent

After continuously playing around with the model, I realized that from the TTS side of things, the voice lacked emotion and sounded robotic. In addition, there was also a noticeable latency (over 5 seconds) every turn when waiting for the model response. I looked into alternatives and found out about Piper TTS, which is a fast, local, and high quality TTS engine that is based on the VITS model. I also did look into Kokoro TTS, which I have actual work-related experience in but the setup was going to take much longer. 

To integrate Piper into my project, I had to implement a queue based architecture rather then the single engine approach I had before. The reason for this was so that in the background, the TTS engine could generate the audio without blocking the main loop. 

Here is a snippet of this code:

```python
tts_queue = queue.Queue()

def tts_worker():
    """Background thread for continuous Text-to-Speech processing."""
    PIPER_MODEL_PATH = "en_US-lessac-medium.onnx"
    use_piper = False
    
    try:
        if os.path.exists(PIPER_MODEL_PATH):
            piper_engine = PiperVoice.load(PIPER_MODEL_PATH)
            piper_sample_rate = piper_engine.config.sample_rate
            use_piper = True
            print("[System] Loaded Piper TTS Model.")
        else:
            raise FileNotFoundError("Piper model not found locally.")
    except Exception as e:
        print(f"[System] Piper error fallback to pyttsx3: {e}")
        # We must initialize pyttsx3 inside the thread loop for safety in some OS environments
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 160)
        
    while True:
        text = tts_queue.get()
        if text is None:
            break
            
        print(f"\n[Qwen] {text}")
        
        if use_piper:
            try:
                stream = sd.OutputStream(samplerate=piper_sample_rate, channels=1, dtype='int16')
                stream.start()
                for chunk in piper_engine.synthesize(text):
                    stream.write(chunk.audio_int16_array)
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"[System] Piper playback failed: {e}")
        else:
            tts_engine.say(text)
            tts_engine.runAndWait()
            
        tts_queue.task_done()

# Start TTS background thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

```

*I also experienced this at work when developing such systems using GPU's, which is that the response to the very first turn takes a long time, since the model needs to be loaded into memory. To fix this issue for this project, I had a silent request be made to Qwen at the very beginning so that it would be ready for the first real user request.

--------------------------------------------------------------------------------------------------------------------------------------

I had to open up Brave browser in developer mode in order to have the browser operations to work. I don't know if this is required for other browsers like Safari or Chrome, but it was just adding a flag to the cmd line to lauch the browser. 

At this point, I could speak into the microphone and tell the voice agent to open YouTube and play a video! The latency was still very noticeable, but the browser automation worked as expected. 

# Memory

Towards the end, I saw a need for the agent to remember everything in a session. Previously, if I told the agent to open up a YouTube video and then play it, the first action would be done, but the second would be skipped. I would have no way of prompting the agent to accurately go off of the last command's result. 

Due to the inherent limitations of my hardware and how I prioritized simplicity, I decided to create a simple in-memory solution, where a simple array in the main thread is created that would store the agent and user messages one after the other. 

Here is a simplified version of what it looked like:

```python

# Initialize memory
memory = []

# Add user message
memory.append({"role": "user", "content": user_input})

# Get response from Qwen
response = qwen.generate(messages=memory)

# Add assistant message
memory.append({"role": "assistant", "content": response})

```

The one caveat was that I could only save the last 10 messages, otherwise the history would get too long and the pollution would start to affect the model's behavior. In my case, it didn't matter that much since I wasn't performing long chains of thought or operations that would run a long time without my input. 

# Whisper Model

As a final test, I decided to ask Gemini if there were any other possible options for STT in python that is free and highly reliable with low latency. It eventually gave me several options but recommended Faster-Whisper, which is a Python implementation of OpenAI's Whisper model that is optimized for speed and efficiency. 

I ended up replacing Vosk with Faster-Whisper using int8 quantization on CPU, which performed satisfactorily in terms of transcribing my voice into text for the agent, only taking 2-3 seconds now. 

For the system that I wanted to build, this kind of quantization was the best tradeoff between speed and accuracy. 

Here is a snippet of my code:

```python

class STTManager:
    def __init__(self):
        print("\n[System] Loading Faster-Whisper Model (base.en)...")
        # Run on CPU with int8 quantization for speed on typical desktop CPUs
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
        
        print("[System] Initializing Microphone...")
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Adjust if it's too sensitive or not sensitive enough
        self.recognizer.dynamic_energy_threshold = False
        
        # Increase pause threshold so it doesn't aggressively cut off the end of your sentence
        # if you pause slightly before saying "YouTube"
        self.recognizer.pause_threshold = 1.5
        self.recognizer.non_speaking_duration = 0.5
        
        self.microphone = sr.Microphone(sample_rate=16000)
        
        # Adjust for ambient noise once on startup
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
        self.is_listening = False
        self._stop_listening_func = None

    def listen_for_speech(self):
        """Blocks and yields text once recognized."""
        if not self.is_listening:
            self.start_listening()
        with self.microphone as source:
            while self.is_listening:
                try:
                    # Listen for a single phrase (blocks until silence is detected)
                    audio_data = self.recognizer.listen(source, timeout=1, phrase_time_limit=15)
                    
                    if not self.is_listening:
                        break # In case we got paused while waiting for speech
                        
                    # Convert the raw audio bytes directly into a normalized float32 numpy array
                    # Whisper expects 16kHz audio, which our sr.Microphone is already set to
                    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Transcribe
                    segments, _ = self.model.transcribe(audio_np, beam_size=5, condition_on_previous_text=False)
                    
                    text = "".join([segment.text for segment in segments]).strip()
                    
                    if text:
                        return text
                        
                except sr.WaitTimeoutError:
                    # Just loops around and keeps listening if nobody spoke
                    pass
                except Exception as e:
                    print(f"\n[STT Error] {e}")
```

In my final STT code, I have a class called STTManager that handles the STT pipeline. It uses the Whisper model to transcribe speech to text after using the microphone to capture audio and converting that into a numpy array that directly feeds into the Whisper model. 

# Takeaways

It is actually very difficult to get a bare bones voice agent working. You have to contend with the limitations of the hardware you are using, the model latency, the orchestration of the entire voice to response pipeline, the fallback behavior if the tool call doesn't work, and so forth. 

Here are things that I didn't try adn would be worth experimeting with in the future:

1. Upgrading hardware somehow (this is the obvious one and would result in faster performance).
2. Figuring out a way to not use flags like [SEARCH] or [NAVIGATE] in the prompt. If the agent was to get more complicated tasks, tracking these flags would be much more difficult. 
3. Browser frameworks like Playwright are not the only option. Selenium and others also do the same work, but with subtle differences. It would be interesting to benchmark the effectiveness of different browser automation frameworks and see how it adds up. I don't know if Qwen would be better at using one or the other. 
4. I have heard of browsers out there that were built for agentic AI, but I didn't look into them for this project. Would they be better with this similar setup? 
5. Increasing the parameter of the model would be a good test of actual reasoning, but that is again tied to the first point of hardware limitations. 
6. Would figuring out a way to increase the tokens per second make the model stronger at these tasks? 

Even though I was able to get the basic idea down, putting it all in code proved to be much more difficult, and time consuming, since one problem could've had multiple sources. Prompting the agent to do something is more of an art then something that can be empirically measured. When I was starting out, I tended to believe that most of the problems were due to bad prompts, when in realiy, the tool call could've been failing. 

# Other Interesting Musings

- I used AI to write most of the code for this project (Gemini 3.1 Pro and Claude Sonnet 4.6), but I did encounter some interesting aspects in this process. First, when downloading the model, the AI that I used would oftentimes write a separate Python script to download the model, instead of just doing it in the client. This is what happened with the Whisper model downloading. 

- In the beginning, I was having trouble having Playwright immediately open the tab in my browser when I told the agent to do so. During the debugging process with AI, a Python script called `test_cdp.py` was created that would send a message to Chromium and would return a success when a link navigation works. This was very useful in quickly resolving the issue. 

There is promise though, and in the future, it would be amazing if I could build an entirely local and customized version of such a system for myself without having to worry too much about reliability. But that is truly for the future.

Thanks for reading!

CZ
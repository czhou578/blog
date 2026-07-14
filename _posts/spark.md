---
layout: post
title: "My first Nvidia DGX Spark"
date: 2026-07-14
---

I recently got an Nvidia DGX Spark from my local Micro Center in Santa Clara. As someone who has never spent more then 1.5k on any electronics device before, I'm now the proud owner of a 5k+ device. In this post, I want to explain why I got it. 

Preface:

My original interest in local AI models started earlier this year after I started experimenting with agentic coding and how AI models could make my work as a programmer more efficient. After seeing the staggered releases of GLM and Kimi and so many others, I realized that it was necessary for me to become compute independent. 

These AI labs are going to increase prices for their coding plans due to the pressure from investors to turn a profit. Not to mention that using the cloud for everything exposes your data and credentials to a potential highest bidder during a data breach. 


First Impressions:

The setup was very quick and simple. I had to just plug the power adapter in, connect it to wifi, and I was able to set the ssh credentials and install all repos and dependencies with no issue. I was even able to install vLLM from the build in under 10 minutes, which was incredible. Previously, I tried to do the same on RunPod and it took probably 4 hours with multiple failed attemps and wasted time. 

Projects:

I began by installing various versions of the most powerful model that could fit in my device with room to spare. I started with no intention of finetuning or doing manual quantization. It turned out to be Qwen 3.6 35B model. I tried to run it various ways, with vLLM and llma.cpp, and under different providers. 

I finally got vLLM working with Qwen at around 60 tok/sec, which I was pretty happy about.


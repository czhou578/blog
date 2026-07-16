---
layout: post
title: "Training GPT-2 on Nvidia DGX Spark"
date: 2026-07-16
---

After getting my DGX Spark, I wanted to test its capabilities by training a GPT-2 model. I used the nanoGPT framework by Andrej Karpathy, which is a lightweight and efficient implementation of GPT-2.

I'm going to walk you through the steps I took to set up the training environment, prepare the dataset, and run the training process on the DGX Spark.

## Setting Up the Environment

I used Andrej Karpathy's nanoGPT framework, which in his `Reproducing GPT-2` video, explains clearly how to set up the script and optimizers to train his GPT-2 replica. I largely followed his instructions but with some minor tweaks to fit the size of model I wanted to train. I used PyTorch and the HuggingFace Transformers library to handle the model architecture and training loop. I also made sure to install the necessary dependencies, including CUDA and cuDNN in my uv environment, to leverage the GPU capabilities of the DGX Spark.

## Dataset Preparation

I asked ChatGPT for datasets to try. Karpathy in his GPT-2 training video used the Tiny Shakespeare dataset, which is a small dataset of Shakespeare's works. My attempts at trying to find more interesting datasets brought me to the TinyStories and the WikiText datasets. Both are available on HuggingFace as a free download and contain a variety of text data that can be used for training language models. I downloaded both datasets onto my local disk from HuggingFace, created `.bin` files, and imported them during training as necessary. 

I do recommend using as creative of datasets as possible, but for the sake of this specific experiment, it doesn't matter as much. The goal is to get output that resembles regular speech rather then complete gibberish.

## Experiment 1: Overfitting a Batch

In my first experiment, I wanted to see if I could overfit a single batch of data. The reason is that if the model can overfit a single batch, it means that the model is capable of learning and memorizing. I used the Tinystories dataset split into training, validation, and test splits.  I limited the test token number to 20,000 and the validation / test splits to 5000 tokens. The script that I used to run the training can be found here: prepare.py

For this run, I ran using the following hyperparameters:

# model
n_layer = 2
n_head = 4
n_embd = 128
block_size = 128

# training
batch_size = 16
learning_rate = 1e-3
max_iters = 2000

# dropout
dropout = 0.0

The model wasn't too big and the batch size was small, so I expected it to quickly overfit. Overfitting is more of a smoke test to see if the model is capable of learning and memorizing. I ran this on my DGX Spark and after 2.5 hours, it achieved a loss of 2.3536.

To generate samples, I used the following command:

```bash
python sample.py \
    --out_dir=out \
    --dataset=tinystories10m \
    --num_samples=5 \
    --max_new_tokens=200
```

The sample file (linked here) contains code to sample from the model checkpoint that is generated at the end of every successful trial run.

This is what it gave back to me:

```text
fter that, the big dog came to the park. The dog loved to play and play in the park. The dog saw Sue and Sue under the bed. Sue was scared, but she did not help.
Sue closed the bushes and said, "I want to play with you!" They played and laughed and had fun. When the dog got to Tom, they decided to play together. Sue was a nice girl. They played a lotion with Sue, and they played together.<|endoftext|>Once upon a time, there was a girl named Lucy. She had a friend named Jerry. Tom loved to play with Sue and Sue. They had fun computer all day long. They would laugh and enjoy their dance.
One day, Tom's mom came to play with all the little ones. But Tom hurt and did not mind. Tom was sad. He thought, "I am, Tom. I love to feel better."
Tom tried to use the computer, but he could not help. Tom did not want to play with Lily. So, Tom tried to balance on the computer. He was not happy to take the computer with him. He tried to push the computer, but he was difficult. Tom was sad, and he did not want to hurt Sara.
Tom tried to climb the computer, but he hurt his knee. He fell down and started to worry. The computer did not move. The computer made Tim's face and came back.
Sara and Tom were sad. They did not want to go. They did not know what to do. They said they were sorry and lost. They had a bad ending.<|endoftext|>Once upon a time, there was a little dog named Spot. Spot lived in a small house with many toys. One day, Tim went to the store with his mom. They were on the street.
Tim's mom saw grown-up, and said, "Lily, can we have enough food with this crayons!" They looked at each other and saw a big, red bucket. They were curious and excited.
"Look, the box is in the grass!" Tim said. He wanted to pick the box and open it, but it was difficult to get it. So, he started to break it. He pulled and pulled, and the box.
"Oh no, what are you doing?" Lily asked.
Lily said, "We need to cut the box, Lily."
"Give it
```

It looked pretty good, but the story didn't make much sense, and there were end of text tokens sprinkled in here and there. Regardless, the model achieved a very low loss and was able to largely memorize the batch. 

## Experiment 2: 

For the next experiment, I wanted to see if I could train the model on a more larger dataset without truncating any number of tokens. I decided to use the TinyStories dataset in full, which is about 10 million tokens. 

I changed the hyperparameters to the following:

# model
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024

# training
batch_size = 12
learning_rate = 6e-4
max_iters = 2000

For the sampling, I also added the following parameters:

```bash
sample_max_new_tokens = 100
sample_start = "\n"
sample_temperature = 0.8
sample_top_k = 200
metrics_log_file = 'metrics.jsonl' # structured, machine-parseable log (one JSON object per line)
train_log_file = 'train_baseline.log' # human-readable log, mirrors stdout
```

The final training run took about 4 hours, and it did come with a caveat*. The model was able to achieve a final loss of 0.24

## Experiment 3

For the final experiment, I tested the model on the WikiText dataset, which contains about 





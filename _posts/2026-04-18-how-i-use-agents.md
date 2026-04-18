---
layout: post
title: "How I use AI Agents for coding in 2026"
date: 2026-04-18
---

![alt text](https://czhou578.github.io/blog/images/ai-ide.png)

Agentic workflows are slowly becoming the norm for software development. My current company generously provides all developers with a subscription to Google AI Ultra, which gives you access to the Antigravity IDE with no rate limits, and the maximum priority for requests. 

Previously, when I was still using GitHub Copilot in VSCode, my AI workflow mainly revolved around adding files and terminal selections into context, typing in a query, waiting for a response, and then manually applying the changes / asking for clarifications. 

I rarely found myself turning on agent mode since I wanted to maintain maximum control over what was accepted and what wasn't. In a way, I thought that accepting agents into my mainstream coding would be like trying to defuse a landmine every time I tried to move forward.

But Antigravity agents on net balance have been **very helpful**. 

Through the use of Claude and Gemini, I've realized that a large number of bugs that I encounter can be fixed relatively easily with a few targeted prompts, sometimes with only one prompt. As a full stack developer, I have been able to quickly implement UI designs (basically not writing Tailwind CSS at all anymore) and also plan out the architecture of new features. 

Even better, if I am not confident of it doing a large code change, I can always ask the agent to break it down into smaller steps, and I can review each step individually, or generate a plan that I can comment on before letting it go. You don't even necessarily need to link every single file of interest in its context; the smarter models can figure that out by themselves.

---

Here are a few things that agents are good at doing, through my experience:

**1. Generating architecture deep-dives**

This one is probably the most useful of them all. I can add several files to context and ask Claude or Gemini to give me a first principles explanation of the code from any file, and tell me how it works together and the overall data flow. I've been able to quickly use this to refresh my memory on code that I haven't touched for a long time, or explore new repositories. 

The GitHub website has an Agents tab that you can use to ask an agent about a codebase. I have used this feature many times to understand the codebase of open source projects I'm interested in contributing to, and even the codebases of other projects at work. 

I truly think there is no more excuse for any developer to not be able to understand any codebase, no matter how large or complex it is, now that you can have AI explain it to you.  

**2. Adding logging to existing code**

This is a task that I used to dread. I would have to go through the codebase and add logging statements to the code, and then I would have to test the code to make sure that the logging was working correctly. But with AI, if you are able to narrow down the area of concern, you can ask the LLM to add targeted logging to identify issues. 

It can go overboard with the emojis, but I have found that the emojis actually help me more quickly identify the data flow and errors in large log files with tens of thousands of lines.

**3. Doing UI work**

I barely write my own CSS anymore. I can just describe what I want in plain English, and Claude / Gemini will generate the Tailwind CSS code for me. In terms of design, I can ask Gemini to come up with possible UI designs that fit specific criteria, and have it implement that automatically. 

My background is not in design, and I am going off of gut instinct when it comes to the design portion, but I think even designers working with LLM's can produce a lot more wireframes that are plausible then otherwise. Gemini does seem to be better at UI then the other LLM's for some reason. 

I find myself being more concerned with how the frontend logic works rather then the designs themselves, which for me is very refreshing. 

**4. Refactoring monolithic components**

For one of my projects, I had a manager component in the backend that was responsible for handling websocket connections from the frontend, sending video, and audio chunks to a pipeline and a separate microservice, and handling all the responses. It was hard to reason about, and I was honestly dreading refactoring it. 

I asked Claude to simply refactor this component into multiple components, while keeping the code simple & functional. It proceeded to give me a plan for refactoring into two components, that would each be responsible for only a part of the original component. It ended up doing that completely correctly on the first try. 

I stressed over testing the system for a while, but it ended up working out perfectly to specification. Unbelievable.

**5. Doing DevOps work**

When I wanted to dockerize one of my projects, I had to create dockerfiles and a docker-compose file. With the complexity revolving around using multiple AI models, and also making sure the final setup was easy to use for a developer, I was facing a big uphill climb. 

Thankfully, I was able to ask Gemini to give me dockerfiles for all the services, and the compose file. It was able to save all the model weights to a local volume so that in development, we wouldn't have to download the model weights every time we wanted to run the system. It saved so much time and greatly improved the developer experience, not to mention enhancing the ease of deployment to prod. 

**6. Doing security audits and identifying performance optimizations**

I was able to ask Claude Opus 4.6 to generate a comprehensive security audit of my codebases, and it was able to identify several vulnerabilities that I was not aware of. It also gave me suggestions on how to fix them, and I was able to take that advice effectively. I think that for a general purpose scan, it is useful.

But be aware that sometimes it will highlight changes that it deems extremely urgent, but in reality are not that big of a deal with the scope of the project. That requires human judgement and good aptitude to distinguish. 

---

Now on the negative side:

1. **Bad Frontend Habits**

On the frontend, it is much easier for agents to keep adding states and ref's in React, which can easily accumulate and become hard to reason about. It seems to be the default behavior for agents to do this, and needs a lot of human supervision. 

Cleanups have been relatively easy for me, but in the early stages, it's definitely a hit and miss. 

2. **Changing Models in the Middle of a Conversation**

Changing models in the middle of a conversation can lead to a loss of context, and reconciling different arguments can be difficult. If Claude suggested one change, but then Gemini reversed it, it is hard to tell which one is correct, and even more difficult to reverse. 

Besides trivial errors, it really is a hope and prayer that Claude and Gemini are on the same page. It is important to prioritise diversity of thought, but sometimes a consensus between models is neede for productivity.

3. **Unnecessary File Creation**

In addition, agents have a habit of creating files that are not needed on occasions. You have to be very clear and explicit about which files to add. More often then not for testing purposes, it will just create a new script to test something, and then not delete it. 

I've found that agents cannot actually deal with Jupyter Notebooks effectively for some reason. They will often make syntax errors when editing code cells, and sometimes just create a Python script to run the code instead. I don't know if I'm missing something here. 

4. **Terminal Management**

While I do appreciate agents spinning up a terminal and running commands to test their changes, it can be very annoying to have to keep track of the terminals that have spawned. I often times have existing terminals in play, and conflicts in this sense can be hard to manage. 

5. **UI when creating plans**

This is definitely nitpicking at this point but when I ask Gemini to generate a plan for example, in Antigravity IDE, it creates a document that is not formatted correctly, and seems very wonky. In comparison, Claude's plan actually looks like a real Markdown preview that you can see on GitHub repos, and not some notepad-quality document. 

---

While a lot of people have been playing around with `AGENT.md` files and massive configurations for agentic workspaces, I still have not felt the need to do so. Agents have a context and I don't want to pollute their memories. I can see where this can be useful for certain cases like formatting code, and maybe it can do it automatically in the style of ruff in Python, or prettier in JavaScript. But is that really worth it when running a script that formats both frontend and backend can be done in a few seconds? 

All in all, the improvement in agentic coding over the past few months has been real, and I'm excited to see what the future in agentic coding holds. 

I've often compared agentic coding to defusing planted bombs in a minefield. Do a bad job, and the mine can explode in your face, creating a big fat mess. But the size of the mines are definitely shrinking, and we should be excited about that.

CZ
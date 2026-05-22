---
layout: post
title: "Adding Paged Attention to NanoGPT"
date: 2026-05-22
---

In the previous post, I introduced prefix caching to NanoGPT. We are going to continue along that line of thinking and introduce **paged attention** to NanoGPT. This is an optimization that allows us to reduce memory fragmentation and improve cache efficiency.

Paged Attention is most notably used in vLLM for managing the KV cache of requests. 
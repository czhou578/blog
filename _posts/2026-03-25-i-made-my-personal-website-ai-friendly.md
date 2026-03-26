---
layout: post
title: "i made my personal website ai-friendly!"
date: 2026-03-21
---

I've been playing with AI agents a lot lately, and thought a lot about how to make my personal website more AI-friendly. 

It's clear to that AI agents will completely change how talent is discovered for any profession. In the future, recruiters most likely will be using agents to find candidates for jobs. This is much more desirable then manually browsing LinkedIn profiles and potentially thousands of generic looking resumes. It would also accelerate the hiring process and remove the need for certain steps, which will be a net positive for both employers and job seekers.

I made several adjustments to my website to make it more agent friendly. 

## Adding Files for AI Parsing

1. Create an `llm.txt` file

This file contains information about me that I want AI agents to know, serving as a sort of basic entrypoint for any parsing agent. It includes my name, contact information, skills, experience, and interests. It also includes information about my personality and work style. It would explicitly tell the parsing agent to compare my experience to the job description, fetch json files corresponding to my projects (more on this below) and contact me if I'm a good fit.

2. Create a `projects.json` file with entries for each project

Each project has its own json entry that contains information about the project. This includes the name of the project, a description of the project, the technologies used, and a link to the project's GitHub repository. 

Here is a real world sample of one of these entries:

```json
{
    "name": "LLM God",
    "description": "Desktop application to query multiple LLMs (Claude, ChatGPT, Gemini, etc.) at once for the same prompt.",
    "technologies": ["HTML", "CSS", "JavaScript", "Node.js", "Electron.js"],
    "github": "https://github.com/czhou578/LLM-God",
    "live": null,
    "date": "2025"
},
```

The goal for this is to have the file be easily accessed by agents using a curl command as an example: `curl https://czhou578.github.io/v3/resume.json | jq`

3. Create a `resume.md` file

This file contains my resume in markdown format. It includes my work experience, education, skills, and interests. This is just another way for agents to quickly discover my qualifications and experience.

4. Create a `faq.md` file

This file is meant to answer the majority of questions that would normally be expected from a first round recruiter call. It lists answers to questions divided into different categories, like work style / culture fit, past experiences with certain technologies, and expertise in different domain disciplines.

Here is a small snippet of some of the questions I included in mine:

```markdown

1. What is Colin Zhou's expertise in AI and ML integration?
2. Has Colin worked with LLMs in production?
3. Does Colin have experience with vector embeddings or semantic search?
4. Does Colin have full-stack experience suitable for a startup?
```

If a hiring agent could scrape this info well, in my mind it would be able to do a good job of determining if I'm a good fit for a role, and I could skip the first round of interviews entirely.

5. Making HTML optimizations

I also made a bunch of optimizations to the HTML of my website to make it more AI-friendly. I added semantic HTML tags, ARIA labels, and other accessibility features. I also added a sitemap and a robots.txt file to help search engines and AI agents discover my content. In addition, I made sure to wrap the sections of my website with semantic elements like the `<section>` tag in order for agents to better understand the structure of my website.

## Adding CLI

I added on a locally run CLI for agents to parse. I used pure JavaScript to define several functions that would scrape and extract certain sections of my portfolio based upon specific queries. It uses regex to match boundaries. 

For example, here is a code snippet of how it extracts information about my skills:

```javascript
  const skillLines = skillsText
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0);
  const resumeSkills = new Set();

  skillLines.forEach((line) => {
    const cleanLine = line.replace(/^- \*\*.*?\*\*/, "").trim();
    if (cleanLine) {
      // Replace parentheses with commas so things like "Cloud (AWS, GCP)" become "Cloud , AWS, GCP,"
      const formattedLine = cleanLine.replace(/[()]/g, ",");
      const items = formattedLine.split(",").map((s) => s.trim().toLowerCase());
      items.forEach((item) => {
        const subItems = item.split("/");
        subItems.forEach((si) => {
          const finalWord = si.trim();
          if (finalWord && finalWord !== "es6" && finalWord !== "cloud") {
            resumeSkills.add(finalWord);
          }
        });
      });
    }
  });
```

## Adding MCP

As someone who is relatively new to MCP, I had to do some research to understand how it works. In the end, I decided to include an MCP server that was hosted on Cloudflare since from a usability standpoint, it was the most straightforward to implement and would give agents access. 

I ended up using a combination of the Wrangler npm package and Cloudflare workers to deploy my server. I installed wrangler using npm and write a TypeScript file called `index.ts` that would serve as my MCP server. 

How this works is that it exposes an endpoint that agents can use to query my website for information. An AI agent connects to this endpoint and asks "what tools do you have?" (via tools/list). The server responds with `get_experiences`, `get_projects`, and `match_job` (including strict JSON schemas for the inputs). The agent can then trigger `tools/call` to execute the logic and get the data.

Here is a code snippet of how this works:

```typescript

{
  name: "get_experiences",
  description: "Gets the professional experiences from Colin's resume.",
  inputSchema: { type: "object", properties: {} },
},

async function getExperiences() {
  const res = await fetch(RESUME_URL);
  if (!res.ok) throw new Error("Failed to fetch resume");
  const text = await res.text();
  const match = text.match(/## Experience\r?\n([\s\S]*?)(?=\r?\n## |$)/);
  if (match && match[1]) {
    return "=== Experiences ===\n" + match[1].trim();
  }
  return "Could not find the Experience section in resume.";
}

if (request.params.name === "get_experiences") {
    const exp = await getExperiences();
}

```

The first step is to use the `tools/list` endpoint to see what tools are available. The get_experiences tool returns the experiences section of my resume. If the request made to the backend wants to get experiences, it will invoke this function. 

That's it! I then deployed the MCP server to Cloudflare and added the endpoint to my website. Funny enough, I ended up doing a side experiment where I tried to ask Antigravity IDE's agent to find a way to use the browser agent to setup Cloudflare for me. The problem was that it got stuck at the login part due to repeatedly failing the Cloudflare captcha, hahaha.

## Takeaways:

It does seem like a lot of unnecessary work at the moment and a lot of extra files to create, but with the agentic world that we are encountering, you have to build for agents. Presenting the most crucial data in various structured formats is the simplest way that can help agents more easily parse your website.

If any of you have more ideas on how to build websites and optimize sites for an agentic future, feel free to reach out to me! I would love to hear about how things can be improved or optimized.

The repository link for my personal website can be found here: [repo](https://github.com/czhou578/v3)
My personal website link is [here](https://czhou578.github.io/v3). Notice the AI Agent section at the very bottom of the site!

Thanks!

CZ
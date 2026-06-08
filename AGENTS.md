# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

CV Wiki — an open-source knowledge base for computer vision learning, interview prep, and engineering practice. Built with Astro Starlight, deployed to Vercel or GitHub Pages.

## Tech Stack

- **Framework**: Astro + Starlight (documentation site generator)
- **Content**: Markdown/MDX files in `src/content/docs/`
- **Language**: Chinese (content is written in Chinese)
- **Deploy**: Vercel (preferred) or GitHub Pages

## Commands

```bash
npm create astro@latest -- --template starlight   # initialize project
npm run dev          # start dev server
npm run build        # production build
npm run preview      # preview production build
```

## Project Structure

- `astro.config.mjs` — site config: title, sidebar navigation, integrations
- `src/content/docs/` — all content pages; each `.md`/`.mdx` file becomes a page
- `public/` — static assets
- `desc.md` — project planning document (original design decisions)

## Content Architecture

Content is organized into modules: basics, image-processing, detection, segmentation, deployment, interview. Sidebar is manually configured in `astro.config.mjs` (not auto-generated).

Each article follows a standard template:
1. One-sentence summary
2. Problem it solves
3. Core idea
4. Math definition
5. Algorithm flow
6. Code example (Python/PyTorch/OpenCV)
7. Interview answer
8. Common follow-up questions
9. Engineering practice
10. Common mistakes
11. References

## Writing Conventions

- First occurrence of a Chinese term should include English: e.g., 非极大值抑制（Non-Maximum Suppression, NMS）
- Formulas must explain each variable
- Interview answers: 1-3 paragraphs, oral-able
- Code examples: Python + PyTorch preferred, OpenCV for image processing
- Article status tags: draft, review, stable

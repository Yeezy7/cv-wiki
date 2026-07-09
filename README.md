<div align="center">

# AI Wiki

**Open-source knowledge base for AI learning, interview prep, and engineering practice**

[![GitHub stars](https://img.shields.io/github/stars/Yeezy7/ai-wiki?style=flat-square)](https://github.com/Yeezy7/ai-wiki/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Yeezy7/ai-wiki?style=flat-square)](https://github.com/Yeezy7/ai-wiki/network/members)
[![License](https://img.shields.io/github/license/Yeezy7/ai-wiki?style=flat-square)](https://github.com/Yeezy7/ai-wiki/blob/main/LICENSE)
[![Deploy](https://img.shields.io/badge/deployed-live-brightgreen?style=flat-square)](https://yeezy7.github.io/ai-wiki/)

[![Deploy to GitHub Pages](https://github.com/Yeezy7/ai-wiki/actions/workflows/deploy.yml/badge.svg)](https://github.com/Yeezy7/ai-wiki/actions/workflows/deploy.yml)

[**Live Demo**](https://yeezy7.github.io/ai-wiki/) · [**Quick Start**](#quick-start) · [**Contributing**](CONTRIBUTING.md)

</div>

---

## ✨ Features

- 📚 **Systematic Content** — From basics to advanced, covering CV, LLM, and multimodal
- 🎯 **Interview Ready** — Standard answers and frequently asked questions for each topic
- 💻 **Code Examples** — Complete implementations in Python + PyTorch/OpenCV
- 🚀 **Production Ready** — Deployment optimization, performance tuning, common pitfalls
- 📖 **Clear Structure** — Modular organization with full-text search

## 📂 Modules

| Module | Topics |
|--------|--------|
| **Computer Vision** | CNN, Object Detection, Image Segmentation, Image Processing, Model Deployment |
| **Large Language Models** | Transformer, RAG, Fine-tuning, RLHF |
| **Multimodal** | CLIP, ViT, Grounding |

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm 9+

### Local Development

```bash
# Clone the repo
git clone https://github.com/Yeezy7/ai-wiki.git
cd ai-wiki

# Install dependencies
npm install

# Start dev server
npm run dev
```

Visit http://localhost:4321 to view the site.

### Available Commands

```bash
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run build:all    # Build main site and all subsites
npm run check        # Validate content and links
```

## 🏗️ Project Structure

```
ai-wiki/
├── src/
│   ├── content/docs/    # Documentation (Markdown/MDX)
│   ├── components/      # Shared components
│   └── styles/          # Global styles
├── sites/               # Subsites (CV, LLM, Multimodal)
├── public/              # Static assets
├── astro.config.mjs     # Site configuration
└── package.json
```

## 📝 Content Format

Article frontmatter:

```yaml
---
title: Article Title
description: One-line description
category: cv | llm | multimodal
tags: [tag1, tag2]
status: draft | review | stable
order: 1
---
```

## 🛠️ Tech Stack

- [Astro](https://astro.build/) — Modern static site generator
- [Starlight](https://starlight.astro.build/) — Documentation site theme
- [GitHub Pages](https://pages.github.com/) — Free hosting

## 🤝 Contributing

Contributions are welcome! Please see [Contributing Guide](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**[Live Demo](https://yeezy7.github.io/ai-wiki/)** · **[GitHub](https://github.com/Yeezy7/ai-wiki)**

</div>

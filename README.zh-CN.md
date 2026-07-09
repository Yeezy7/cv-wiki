<div align="center">

# AI Wiki

**面向 AI 学习、面试和工程实践的开源知识库**

[![GitHub stars](https://img.shields.io/github/stars/Yeezy7/ai-wiki?style=flat-square)](https://github.com/Yeezy7/ai-wiki/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Yeezy7/ai-wiki?style=flat-square)](https://github.com/Yeezy7/ai-wiki/network/members)
[![License](https://img.shields.io/github/license/Yeezy7/ai-wiki?style=flat-square)](https://github.com/Yeezy7/ai-wiki/blob/main/LICENSE)
[![Deploy](https://img.shields.io/badge/deployed-live-brightgreen?style=flat-square)](https://yeezy7.github.io/ai-wiki/)

[![Deploy to GitHub Pages](https://github.com/Yeezy7/ai-wiki/actions/workflows/deploy.yml/badge.svg)](https://github.com/Yeezy7/ai-wiki/actions/workflows/deploy.yml)

[**在线访问**](https://yeezy7.github.io/ai-wiki/) · [**快速开始**](#快速开始) · [**贡献指南**](CONTRIBUTING.md)

</div>

---

## ✨ 特性

- 📚 **系统化内容** — 从基础到进阶，覆盖 CV、LLM、多模态等核心领域
- 🎯 **面试导向** — 每个知识点附带标准回答与高频追问
- 💻 **代码实战** — Python + PyTorch/OpenCV 完整示例
- 🚀 **工程落地** — 包含部署优化、性能调优、常见坑位
- 📖 **结构清晰** — 模块化组织，支持全文搜索

## 📂 内容模块

| 模块 | 内容 |
|------|------|
| **计算机视觉** | CNN、目标检测、图像分割、图像处理、模型部署 |
| **大语言模型** | Transformer、RAG、Fine-tuning、RLHF |
| **多模态** | CLIP、ViT、Grounding |

## 🚀 快速开始

### 环境要求

- Node.js 18+
- npm 9+

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/Yeezy7/ai-wiki.git
cd ai-wiki

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问 http://localhost:4321 查看站点。

### 可用命令

```bash
npm run dev          # 启动开发服务器
npm run build        # 构建生产版本
npm run preview      # 预览生产构建
npm run build:all    # 构建主站与所有子站
npm run check        # 校验内容与链接
```

## 🏗️ 项目结构

```
ai-wiki/
├── src/
│   ├── content/docs/    # 文档内容 (Markdown/MDX)
│   ├── components/      # 共享组件
│   └── styles/          # 全局样式
├── sites/               # 子站 (CV、LLM、多模态)
├── public/              # 静态资源
├── astro.config.mjs     # 站点配置
└── package.json
```

## 📝 内容规范

文章 frontmatter 格式：

```yaml
---
title: 文章标题
description: 一句话描述
category: cv | llm | multimodal
tags: [tag1, tag2]
status: draft | review | stable
order: 1
---
```

## 🛠️ 技术栈

- [Astro](https://astro.build/) — 现代化静态站点生成器
- [Starlight](https://starlight.astro.build/) — 文档站点主题
- [GitHub Pages](https://pages.github.com/) — 免费托管

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md) 了解详情。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

<div align="center">

**[在线访问](https://yeezy7.github.io/ai-wiki/)** · **[GitHub](https://github.com/Yeezy7/ai-wiki)**

</div>

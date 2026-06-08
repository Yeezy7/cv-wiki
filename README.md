# AI Wiki

面向 AI 学习、面试和工程实践的开源知识库。

## 项目简介

AI Wiki 是一个综合性的 AI 知识库，涵盖多个领域：

- **计算机视觉** — CNN、目标检测、图像分割、模型部署
- **大语言模型** — Transformer、RAG、Fine-tuning、RLHF
- **多模态** — CLIP、ViT、Grounding

每个知识点都做到：
- **能学懂** — 从直觉出发，配合图示和代码
- **能面试** — 附带标准回答和高频追问
- **能落地** — 包含工程实践和常见坑
- **能查阅** — 结构化组织，快速定位

## 内容模块

### 计算机视觉
- 深度学习基础（CNN、BatchNorm、激活函数、损失函数）
- 目标检测（R-CNN、YOLO、NMS、mAP）
- 图像分割（U-Net、DeepLab、Mask R-CNN）
- 图像处理（滤波、边缘检测、形态学操作）
- 模型部署（ONNX、TensorRT、量化）

### 大语言模型
- Transformer 架构
- 注意力机制
- RAG 检索增强生成
- Fine-tuning 微调技术
- RLHF 人类反馈强化学习

### 多模态
- CLIP 对比学习
- ViT 视觉 Transformer
- Grounding 视觉定位

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build

# 预览生产版本
npm run preview

# 同步共享组件到各子站
npm run sync:shared

# 强制刷新根站和子站内容索引
npm run sync:content

# 检查共享组件是否漂移，并校验文章 frontmatter 和内部链接
npm run check

# 构建主站与所有子站，并合并输出
npm run build:all
```

项目使用 npm workspaces 管理 `sites/*` 子站脚本。`npm run build:all` 会先同步共享组件、样式和中文翻译，再运行工程校验、刷新内容索引，最后构建主站及所有子站。

共享组件以根站 `src/components/`、`src/styles/custom.css`、`src/content/i18n/zh-CN.json` 为来源。修改共享 UI 后先运行 `npm run sync:shared`，提交前运行 `npm run check`。

## 可选配置

评论区使用 Giscus，仅在普通文章页展示。未配置时文章页会显示“评论功能尚未启用”的说明，不会加载空配置脚本。

```bash
PUBLIC_GISCUS_REPO=Yeezy7/ai-wiki
PUBLIC_GISCUS_REPO_ID=你的 Giscus repo id
PUBLIC_GISCUS_CATEGORY=Announcements
PUBLIC_GISCUS_CATEGORY_ID=你的 Giscus category id
```

AI 问答助手源码暂时保留，但当前不会在页面中渲染。后续如果要开启，建议先补后端代理或明确浏览器端密钥方案。

## 内容规范

所有文章必须包含以下 frontmatter 字段：

```yaml
title: 标题
description: 一句话描述
category: cv
tags: [tag-a, tag-b]
status: stable
order: 1
```

`status` 只能使用 `draft`、`review`、`stable`。`npm run validate:content` 会检查根站和所有子站内容。

内部链接建议使用带 `/ai-wiki` 前缀的站点绝对路径，或使用相对路径。`npm run validate:links` 会检查 Markdown/MDX 中的内部页面、静态资源和锚点链接。

## 技术栈

- [Astro](https://astro.build/) — 现代化的静态站点生成器
- [Starlight](https://starlight.astro.build/) — 文档站点主题
- Markdown / MDX — 内容格式

## 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md)。

## 许可证

MIT License

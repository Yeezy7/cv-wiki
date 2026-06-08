# AI Wiki 待办事项

## 已完成功能

- [x] 多领域架构（CV、LLM、多模态）
- [x] 独立领域侧边栏
- [x] 文章元信息（最后更新时间、编辑链接）
- [x] 评论入口（Giscus 可配置，未配置时显示说明）
- [x] GitHub 贡献面板（编辑当前页、新增文章、提交 Issue）
- [x] 移动端适配优化
- [x] 返回顶部按钮
- [x] 自定义样式系统
- [x] 搜索功能增强（快捷键 ⌘K、搜索高亮）
- [x] 图片懒加载
- [x] SEO 优化（Starlight Head + 结构化数据）
- [x] 文章目录导航优化（滚动高亮）
- [x] 共享组件同步脚本（根站组件同步到各子站）
- [x] Starlight UI 中文化配置
- [x] 共享组件漂移检查（`npm run check:shared`）
- [x] 文章 frontmatter 校验（`npm run validate:content`）
- [x] 内部链接与锚点校验（`npm run validate:links`）
- [x] npm workspaces 子站脚本管理
- [x] 全站内容索引刷新脚本（`npm run sync:content`）

## 待配置功能

### Giscus 评论

- **优先级**：中
- **复杂度**：低
- **描述**：当前评论组件已经接入可配置逻辑，但需要补充 Giscus 的仓库 ID 和分类 ID。
- **配置项**：
  - `PUBLIC_GISCUS_REPO`
  - `PUBLIC_GISCUS_REPO_ID`
  - `PUBLIC_GISCUS_CATEGORY`
  - `PUBLIC_GISCUS_CATEGORY_ID`

### AI 问答

- **优先级**：低
- **复杂度**：中
- **状态**：暂不启用
- **描述**：当前保留 `AIChat.astro` 源码，但不挂载到页面。后续如果开启，应先补后端代理或明确浏览器端密钥方案，避免把真实 API Key 暴露给访问者。

### 真实用户权限 / 在线后台

- **优先级**：低
- **复杂度**：高
- **描述**：静态站不再保留假的 GitHub OAuth 和浏览器端 Token 后台。若以后需要真实后台，应新增后端服务或 GitHub App。

## 待实现功能

### 知识图谱

- **优先级**：中
- **复杂度**：高
- **描述**：可视化展示知识点之间的关系
- **实现方案**：
  - 使用 D3.js 或 Cytoscape.js 进行可视化
  - 定义知识点之间的依赖关系
  - 支持交互式探索（缩放、拖拽、点击）
  - 按领域分组展示
- **技术要点**：
  - 需要定义知识点关系数据结构
  - 响应式设计，适配移动端
  - 与现有文章页面联动

## 内容待补充

### 大语言模型

- [ ] 注意力机制详解
- [ ] RAG 检索增强生成
- [ ] Fine-tuning 微调技术
- [ ] RLHF 人类反馈强化学习

### 多模态

- [ ] ViT 视觉 Transformer
- [ ] Grounding 视觉定位

### 面试题库

- [ ] LLM 面试题
- [ ] 多模态面试题

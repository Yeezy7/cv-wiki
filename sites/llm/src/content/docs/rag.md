---
title: RAG 检索增强生成
description: Retrieval-Augmented Generation 的原理、架构与工程实践
category: llm
tags: [llm, rag, retrieval, embedding, vector-database]
status: stable
order: 4
---

# RAG 检索增强生成

## 一句话解释

RAG（Retrieval-Augmented Generation）通过在生成答案前先检索相关文档，将外部知识注入大模型，解决知识过时和幻觉问题。

## 它解决什么问题

大语言模型有两个核心局限：

1. **知识截止**：模型只记得训练数据中的知识，无法获取最新信息。
2. **幻觉**：模型会自信地生成错误答案，尤其是涉及细节时。

RAG 的思路很直接：与其让模型"回忆"知识，不如先帮它"查资料"。就像考试可以带参考书，有参考资料的回答通常更准确。

## 核心思想

RAG 的工作流程分为三步：

```
用户问题
   │
   ▼
[检索] → 从知识库中找到相关文档片段
   │
   ▼
[增强] → 将检索结果拼接到 Prompt 中
   │
   ▼
[生成] → LLM 基于上下文生成答案
```

### 关键组件

| 组件 | 作用 | 常用方案 |
|------|------|----------|
| 文档加载 | 读取原始文档 | LangChain DocumentLoader |
| 文本切分 | 将长文档切成片段 | RecursiveCharacterTextSplitter |
| 向量化 | 将文本转为向量 | OpenAI Embedding、BGE、GTE |
| 向量数据库 | 存储和检索向量 | FAISS、Milvus、Pinecone |
| 检索器 | 执行相似度搜索 | 向量检索 + 重排序 |
| 生成器 | 基于上下文生成答案 | GPT-4、Claude、Llama |

### 文本切分策略

切分质量直接影响检索效果。常见策略：

- **固定长度切分**：按字符数切分，简单但可能切断语义
- **递归切分**：按段落 → 句子 → 字符递归切分，保持语义完整
- **语义切分**：基于嵌入相似度判断边界，效果最好但计算开销大

推荐参数：chunk_size=500-1000，chunk_overlap=50-100。

### 检索策略

- **向量检索**：计算问题与文档片段的嵌入相似度，召回 Top-K
- **关键词检索**：BM25 等传统方法，适合精确匹配
- **混合检索**：向量 + 关键词，兼顾语义和精确匹配
- **重排序**：用 Cross-Encoder 对召回结果重新排序，提升精度

## 代码示例

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = DirectoryLoader("./knowledge_base", glob="**/*.md")
documents = loader.load()

# 2. 切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = splitter.split_documents(documents)

# 3. 向量化并存入向量数据库
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. 创建检索链
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_type="mmr",        # 最大边际相关性，增加多样性
        search_kwargs={"k": 5}    # 返回 Top 5
    ),
    return_source_documents=True,
)

# 5. 提问
result = qa_chain({"query": "CNN 的感受野怎么计算？"})
print(result["result"])
print(result["source_documents"])  # 查看引用来源
```

## 面试标准回答

**"RAG 和微调怎么选"**

RAG 适合知识频繁更新、需要引用来源的场景（如知识库问答、客服系统）。微调适合需要改变模型行为风格、提升特定任务能力的场景（如代码生成、特定领域对话）。两者可以结合：先微调让模型适配领域风格，再用 RAG 注入最新知识。

**"RAG 的检索效果怎么评估"**

从三个层面评估：(1) 检索质量——Recall@K（Top-K 中包含正确文档的比例）、MRR（正确文档的排名）；(2) 生成质量——答案准确性、完整性、是否忠于检索内容；(3) 端到端效果——最终答案的正确率。建议先单独评估检索，再评估生成。

**"RAG 有哪些常见失败模式"**

主要三类：(1) 检索失败——相关文档没被召回，可能是切分粒度不对或嵌入模型不匹配；(2) 排序失败——相关文档被召回但排名靠后，被截断；(3) 生成失败——检索到了正确文档，但 LLM 没有正确利用，或产生了幻觉。排查时先定位是哪个环节出了问题。

## 高频追问

**Q1: chunk_size 怎么设？**

没有万能值，取决于文档类型和任务。太小会丢失上下文，太大会引入噪声。建议从 500-800 字开始，在评估集上对比不同粒度的效果。FAQ 类文档可以用较小粒度（200-300），技术文档用较大粒度（800-1500）。

**Q2: 向量检索和 BM25 哪个好？**

各有优势。向量检索擅长语义匹配（"怎么训练模型"能匹配到"模型训练方法"），BM25 擅长精确匹配（专有名词、编号）。最佳实践是混合检索：先用两种方法分别召回，再合并去重。

**Q3: 如何处理多轮对话的 RAG？**

需要将对话历史融入检索 query。常见做法：(1) 将最近 N 轮对话和当前问题拼接成完整 query；(2) 用 LLM 将对话历史压缩成一个独立的检索 query；(3) 维护对话上下文的摘要，作为检索的额外信息。

**Q4: RAG 的成本怎么控制？**

主要成本在 Embedding 和 LLM 调用。优化方法：(1) 缓存常见问题的检索结果；(2) 使用开源嵌入模型（如 BGE、GTE）替代 OpenAI Embedding；(3) 先用小模型（如 GPT-4o-mini）生成，复杂问题再升级；(4) 减少不必要的上下文长度。

## 工程实践

### 1. 知识库更新策略

- 增量更新：只对新增/修改的文档重新向量化
- 版本管理：保留历史版本，支持回滚
- 去重：避免重复文档影响检索结果

### 2. 引用溯源

答案中应包含引用来源，方便用户验证：

```python
# 在 Prompt 中要求引用
prompt = """
基于以下参考资料回答问题。回答中引用来源时使用 [1][2] 标记。

参考资料：
{context}

问题：{question}
"""
```

### 3. 监控指标

- 检索命中率：用户问题是否能检索到相关文档
- 答案准确率：生成的答案是否正确
- 用户满意度：是否需要人工介入

## 常见误区

1. **"RAG 可以完全替代微调"** — 不能。RAG 擅长注入知识，但不擅长改变模型行为。需要模型"说人话"或"按格式输出"时，微调更合适。

2. **"检索越多文档越好"** — 过多文档会引入噪声，分散模型注意力。通常 Top 3-5 就够了。

3. **"向量数据库万能"** — 向量检索不擅长精确匹配（如产品编号、电话号码）。需要结合关键词检索。

4. **"切分粒度越小越好"** — 太小的片段缺乏上下文，模型无法理解。需要在粒度和上下文之间平衡。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Documentation](https://python.langchain.com/docs/tutorials/rag)
- [Building RAG Applications](https://www.pinecone.io/learn/retrieval-augmented-generation/)

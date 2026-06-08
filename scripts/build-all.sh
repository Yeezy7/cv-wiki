#!/bin/bash

# 构建所有站点并合并输出

set -euo pipefail

echo "开始构建所有站点..."

echo "同步共享组件..."
bash scripts/sync-shared.sh

echo "运行工程校验..."
npm run check

echo "刷新内容索引..."
npm run sync:content

# 构建主站
echo "构建主站..."
npm run build

# 构建 CV 站点
echo "构建 CV 站点..."
npm run build:cv

# 构建 LLM 站点
echo "构建 LLM 站点..."
npm run build:llm

# 构建多模态站点
echo "构建多模态站点..."
npm run build:multimodal

# 构建面试题库站点
echo "构建面试题库站点..."
npm run build:interview

# 合并所有站点
echo "合并所有站点..."
./scripts/merge-sites.sh

echo "所有站点构建完成！"

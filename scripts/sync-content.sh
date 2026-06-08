#!/bin/bash

# 强制刷新根站和各子站的 Astro content store，避免增量缓存残留。

set -euo pipefail

WORKSPACES=("ai-wiki-cv" "ai-wiki-llm" "ai-wiki-multimodal" "ai-wiki-interview")

echo "刷新主站内容索引..."
npm run astro -- sync --force

for workspace in "${WORKSPACES[@]}"; do
  echo "刷新 $workspace 内容索引..."
  npm exec --workspace "$workspace" -- astro sync --force
done

echo "内容索引已刷新。"

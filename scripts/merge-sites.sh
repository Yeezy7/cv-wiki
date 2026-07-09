#!/bin/bash

# 合并所有子站构建输出到主站 dist 目录

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT_DIR/dist"
SITES=(cv llm multimodal interview)

echo "开始合并站点..."

for site in "${SITES[@]}"; do
  src="$ROOT_DIR/sites/$site/dist"
  if [[ ! -d "$src" ]]; then
    echo "错误：$src 不存在，请先构建子站。" >&2
    exit 1
  fi
  rm -rf "$DIST/$site"
  mkdir -p "$DIST/$site"
  cp -R "$src/." "$DIST/$site/"
  echo "已合并 $site"
done

echo "合并完成！dist 目录结构："
ls -la "$DIST/"

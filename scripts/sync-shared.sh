#!/bin/bash

# 将根站共享组件以符号链接形式挂载到各子站，消除文件复制和同步漂移。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-sync}"
SITES=("cv" "llm" "multimodal" "interview")
COMPONENTS=(
  "AdminPanel.astro"
  "BackToHome.astro"
  "BackToTop.astro"
  "Comments.astro"
  "CommentsWrapper.astro"
  "Head.astro"
  "LazyImage.astro"
  "MarkdownContent.astro"
  "Mermaid.astro"
  "SEOHead.astro"
  "SidebarPanel.astro"
  "SiteTitle.astro"
  "UserAuth.astro"
)
REMOVED_COMPONENTS=(
  "AIChat.astro"
)

if [[ "$MODE" != "sync" && "$MODE" != "--check" ]]; then
  echo "用法：bash scripts/sync-shared.sh [--check]" >&2
  exit 1
fi

# 确保符号链接指向正确目标；--check 模式下仅报告不修复。
check_or_link() {
  local target_file="$1"
  local link_target="$2"

  if [[ "$MODE" == "--check" ]]; then
    if [[ ! -L "$target_file" ]]; then
      echo "缺少符号链接：$target_file" >&2
      return 1
    fi
    local actual
    actual="$(readlink "$target_file")"
    if [[ "$actual" != "$link_target" ]]; then
      echo "符号链接指向错误：$target_file -> $actual（期望 $link_target）" >&2
      return 1
    fi
  else
    rm -f "$target_file"
    ln -s "$link_target" "$target_file"
  fi
}

has_drift=0

for site in "${SITES[@]}"; do
  component_target="$ROOT_DIR/sites/$site/src/components"
  style_target="$ROOT_DIR/sites/$site/src/styles"
  i18n_target="$ROOT_DIR/sites/$site/src/content/i18n"

  if [[ "$MODE" == "sync" ]]; then
    mkdir -p "$component_target" "$style_target" "$i18n_target"
    for component in "${REMOVED_COMPONENTS[@]}"; do
      rm -f "$component_target/$component"
    done
  fi

  for component in "${COMPONENTS[@]}"; do
    check_or_link "$component_target/$component" "../../../../src/components/$component" || has_drift=1
  done

  check_or_link "$style_target/custom.css" "../../../../src/styles/custom.css" || has_drift=1
  check_or_link "$i18n_target/zh-CN.json" "../../../../../src/content/i18n/zh-CN.json" || has_drift=1
  check_or_link "$ROOT_DIR/sites/$site/src/content/config.ts" "../../../../src/content/config.ts" || has_drift=1
done

if [[ "$MODE" == "--check" ]]; then
  if [[ "$has_drift" -ne 0 ]]; then
    echo "共享文件检查失败。请运行 npm run sync:shared 后重试。" >&2
    exit 1
  fi

  echo "共享文件检查通过。"
else
  echo "共享组件、样式和中文翻译已通过符号链接挂载到子站。"
fi

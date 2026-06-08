#!/bin/bash

# 同步或检查根站共享文件到各子站，避免多站实现漂移。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-sync}"
SITES=("cv" "llm" "multimodal" "interview")
COMPONENTS=(
  "AIChat.astro"
  "AdminPanel.astro"
  "BackToHome.astro"
  "BackToTop.astro"
  "Comments.astro"
  "CommentsWrapper.astro"
  "Head.astro"
  "LazyImage.astro"
  "MarkdownContent.astro"
  "SEOHead.astro"
  "SearchEnhance.astro"
  "SidebarPanel.astro"
  "SiteTitle.astro"
  "TableOfContentsHighlight.astro"
  "UserAuth.astro"
)

if [[ "$MODE" != "sync" && "$MODE" != "--check" ]]; then
  echo "用法：bash scripts/sync-shared.sh [--check]" >&2
  exit 1
fi

check_file() {
  local source_file="$1"
  local target_file="$2"

  if [[ ! -f "$target_file" ]]; then
    echo "缺少共享文件：$target_file" >&2
    return 1
  fi

  if ! cmp -s "$source_file" "$target_file"; then
    echo "共享文件未同步：$target_file" >&2
    return 1
  fi
}

has_drift=0

for site in "${SITES[@]}"; do
  component_target="$ROOT_DIR/sites/$site/src/components"
  style_target="$ROOT_DIR/sites/$site/src/styles"
  i18n_target="$ROOT_DIR/sites/$site/src/content/i18n"

  if [[ "$MODE" == "sync" ]]; then
    mkdir -p "$component_target" "$style_target" "$i18n_target"
  fi

  for component in "${COMPONENTS[@]}"; do
    source_file="$ROOT_DIR/src/components/$component"
    target_file="$component_target/$component"

    if [[ "$MODE" == "--check" ]]; then
      check_file "$source_file" "$target_file" || has_drift=1
    else
      cp "$source_file" "$target_file"
    fi
  done

  if [[ "$MODE" == "--check" ]]; then
    check_file "$ROOT_DIR/src/styles/custom.css" "$style_target/custom.css" || has_drift=1
    check_file "$ROOT_DIR/src/content/i18n/zh-CN.json" "$i18n_target/zh-CN.json" || has_drift=1
  else
    cp "$ROOT_DIR/src/styles/custom.css" "$style_target/custom.css"
    cp "$ROOT_DIR/src/content/i18n/zh-CN.json" "$i18n_target/zh-CN.json"
  fi
done

if [[ "$MODE" == "--check" ]]; then
  if [[ "$has_drift" -ne 0 ]]; then
    echo "共享文件检查失败。请运行 npm run sync:shared 后重试。" >&2
    exit 1
  fi

  echo "共享文件检查通过。"
else
  echo "共享组件、样式和中文翻译已同步到子站。"
fi

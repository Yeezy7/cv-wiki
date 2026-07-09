# 任务计划：解决404问题

## 目标
解决运行 `bash ./scripts/build-all.sh` 后访问 `http://localhost:4321/ai-wiki/` 出现404的问题

## 阶段1：分析问题
- [ ] 检查构建脚本和配置
- [ ] 分析dist目录结构
- [ ] 验证base路径配置

## 阶段2：测试验证
- [ ] 运行构建脚本
- [ ] 启动preview服务器
- [ ] 验证访问路径

## 阶段3：修复问题
- [ ] 根据分析结果进行修复
- [ ] 验证修复效果

## 发现记录
1. astro.config.mjs中设置了 `base: '/ai-wiki'`
2. dist目录中没有 `ai-wiki` 子目录，文件直接放在根目录
3. index.html中引用了 `/ai-wiki/` 路径
4. 子站（如cv）的base路径是 `/ai-wiki/cv`
5. 主站package.json中preview命令是 `astro preview`

## 错误记录
| 错误 | 尝试 | 解决方案 |
|------|------|----------|
| 无 | 无 | 无 |

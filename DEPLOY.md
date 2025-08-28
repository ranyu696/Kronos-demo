# GitHub Pages 自动部署指南

## 部署步骤

### 1. 启用 GitHub Pages

1. 进入你的 GitHub 仓库设置（Settings）
2. 找到 "Pages" 部分
3. Source 选择 "GitHub Actions"
4. 保存设置

### 2. 配置 GitHub Actions

已创建的 `.github/workflows/update-predictions.yml` 会：
- 每小时自动运行一次（在第5分钟）
- 获取 ETH/USDT 数据
- 生成预测图表
- 更新 HTML 页面
- 自动部署到 GitHub Pages

### 3. 首次手动触发

1. 进入仓库的 Actions 标签
2. 选择 "Update ETH Predictions" workflow
3. 点击 "Run workflow" 按钮手动触发

### 4. 访问你的页面

部署成功后，访问：
```
https://shiyu-coder.github.io/Kronos-demo/
```

## 自动更新机制

### 工作流程

1. **定时触发**：每小时第5分钟自动执行
2. **数据获取**：从 Binance 公共 API 获取 ETH/USDT 数据
3. **模型预测**：使用 Kronos 模型生成 24 小时预测
4. **图表生成**：创建包含历史和预测的图表
5. **页面更新**：更新 HTML 中的预测概率和时间戳
6. **自动部署**：将更新推送到 GitHub Pages

### 文件说明

- `update_predictions_once.py`：单次运行版本（用于 GitHub Actions）
- `update_predictions.py`：持续运行版本（用于本地或服务器）
- `.github/workflows/update-predictions.yml`：GitHub Actions 配置

### 监控运行状态

1. 在仓库的 Actions 标签页查看运行历史
2. 每个运行都会显示状态（成功/失败）
3. 点击具体运行查看详细日志

## 本地测试

在推送到 GitHub 前，可以本地测试：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行单次更新
python update_predictions_once.py

# 查看生成的文件
open index.html
```

## 故障排除

### 如果 Actions 失败

1. **检查 Actions 日志**：
   - 进入 Actions 标签
   - 点击失败的运行
   - 查看具体错误信息

2. **常见问题**：
   - API 限制：Binance API 可能有频率限制
   - 模型下载：首次运行需要下载模型文件
   - 依赖问题：确保 requirements.txt 包含所有依赖

3. **手动重试**：
   - 在 Actions 页面点击 "Re-run all jobs"

### 如果页面不更新

1. 检查 GitHub Pages 设置是否正确
2. 确认 workflow 成功运行
3. 清除浏览器缓存后刷新页面
4. 查看仓库的 gh-pages 分支（如果使用）

## 自定义配置

### 修改更新频率

编辑 `.github/workflows/update-predictions.yml`：

```yaml
on:
  schedule:
    # 每2小时运行一次
    - cron: '5 */2 * * *'
    # 每天运行一次（UTC 00:05）
    - cron: '5 0 * * *'
```

### 修改预测参数

编辑 `update_predictions_once.py` 中的配置：

```python
Config = {
    "SYMBOL": 'ETHUSDT',      # 交易对
    "INTERVAL": '1h',          # K线间隔
    "PRED_HORIZON": 24,        # 预测时长
    "N_PREDICTIONS": 30,       # Monte Carlo 采样次数
}
```

## 安全提示

- 不要在代码中硬编码 API 密钥
- 如需使用私有 API，使用 GitHub Secrets
- 定期检查 Actions 使用量避免超限
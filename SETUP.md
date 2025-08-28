# Kronos Demo 配置和运行指南

## 系统要求

- Python 3.8 或更高版本
- 至少 4GB 可用内存
- 稳定的网络连接（用于访问 Binance API 和下载模型）

## 安装步骤

### 1. 克隆或下载项目

```bash
git clone <你的仓库地址>
cd Kronos-demo
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 模型准备

首次运行时，程序会自动从 Hugging Face 下载 Kronos 模型到 `../Kronos_model` 目录。这可能需要几分钟时间，取决于网络速度。

如果自动下载失败，可以手动下载：
- 访问 https://huggingface.co/NeoQuasar/Kronos-small
- 访问 https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base

## 运行方式

### 方式一：单次运行（测试用）

```bash
python update_predictions.py
```

这将执行一次完整的预测流程：
1. 获取 BTC/USDT 历史数据
2. 生成 24 小时价格预测
3. 创建预测图表
4. 更新 HTML 页面
5. 自动提交到 Git（如果配置了仓库）

### 方式二：持续运行（生产环境）

脚本默认会持续运行，每小时自动更新一次预测：

```bash
# 在后台持续运行
nohup python update_predictions.py > kronos.log 2>&1 &

# 或使用 screen/tmux
screen -S kronos
python update_predictions.py
# Ctrl+A+D 分离会话
```

## 配置说明

在 `update_predictions.py` 中的 `Config` 字典可调整以下参数：

```python
Config = {
    "MODEL_PATH": "../Kronos_model",    # 模型存储路径
    "SYMBOL": 'BTCUSDT',                # 交易对
    "INTERVAL": '1h',                    # K线时间间隔
    "HIST_POINTS": 360,                  # 使用的历史数据点数
    "PRED_HORIZON": 24,                  # 预测时长（小时）
    "N_PREDICTIONS": 30,                 # Monte Carlo 采样次数
    "VOL_WINDOW": 24,                    # 波动率计算窗口
}
```

## 查看结果

1. **本地查看**：直接在浏览器打开 `index.html` 文件

2. **部署到 GitHub Pages**：
   - 在 GitHub 仓库设置中启用 Pages
   - 选择主分支作为源
   - 访问 `https://<用户名>.github.io/<仓库名>/`

## 故障排除

### 1. 模型下载失败
- 检查网络连接
- 尝试使用代理
- 手动下载模型文件

### 2. Binance API 连接问题
- 确保能访问 api.binance.com
- 检查防火墙设置

### 3. 内存不足
- 减少 `N_PREDICTIONS` 参数值
- 使用更小的 `HIST_POINTS`

### 4. Git 推送失败
- 确保已配置 Git 凭据
- 检查仓库权限

## 停止运行

```bash
# 查找进程
ps aux | grep update_predictions.py

# 终止进程
kill <进程ID>
```

## 日志监控

```bash
# 实时查看日志（如果使用了 nohup）
tail -f kronos.log
```
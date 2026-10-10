# 本地大模型命令行助手

> AI 协会创智部二面 · 实战题
> 作者：[你的名字]（GitHub: [你的用户名]）

一个跑在**自己电脑上**的命令行 AI 助手：不联网、不调用云端 API，直接和本地部署的大模型对话；
再加上 YOLO 目标检测，它还能"看图说话"。

题目要求"完成本地大模型部署 → 搭建自己的智能体 → 将模型作为 provider，通过 API 接入自己的应用"。
本仓库记录了这个过程的**完整实现、踩坑记录与解决思路**，而不只是最后的代码。

---

## 当前进度

| 状态 | 内容 |
|---|---|
| ✅ | 本地大模型部署（Ollama + Qwen2.5-1.5B，纯 CPU 运行） |
| ✅ | CLI 应用：通过 HTTP API 调用本地模型，支持流式输出 |
| ✅ | YOLO 目标检测：本地 CPU 推理跑通，并接入 CLI（`-i 图片` 看图说话） |
| ⏳ | 在魔搭 ModelScope 免费 GPU 上完成一次训练（脚本已备好 `train/train_yolo.py`，待执行） |
| ⏳ | 创意作品整合（计划中） |

---

## 硬件与软件环境

| 项目 | 配置 |
|---|---|
| 机型 | Dell Inspiron 5548（2015 年） |
| CPU | Intel i5-5200U（双核四线程，支持 AVX2） |
| 内存 | 16 GB |
| 显卡 | Intel HD 5500 + AMD Radeon R7 M260（**均不可用于推理，见"已知问题"**） |
| 系统 | Windows 10 专业版 21H2（19044） |
| Python | 3.11.5 |
| Ollama | 0.40.2 |
| 模型 | `qwen2.5:1.5b`（约 1 GB，Q4 量化） |

**选型说明（为什么是 1.5B）**：本机是 11 年前的低功耗双核笔记本，CPU 是唯一可用的算力（两块显卡都不具备可用算力）。
在 CPU 上跑 7B 级模型体验无法接受，而 0.5B 又经常答非所问。实测 `qwen2.5:1.5b` 在本机可稳定达到约 **13 tokens/秒**，
是这个硬件配置下"跑得动"与"说得像人话"的平衡点。

---

## 快速开始（复现步骤）

```cmd
:: 1. 安装 Ollama
::    下载 https://ollama.com/download/windows 并安装

:: 2. 拉取模型（约 1 GB）
ollama pull qwen2.5:1.5b

:: 3. 【关键】关闭 Vulkan，强制使用 CPU
::    不设这一步，本机会直接崩溃，详见"已知问题"
setx OLLAMA_VULKAN 0

:: 4. 启动服务（新开一个终端窗口，保持不关）
ollama serve

:: 5. 安装 Python 依赖
::    先装 CPU 版 PyTorch（约 200 MB），否则默认会下载 2.5 GB 的 CUDA 版本
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
py -3.11 -m pip install -r requirements.txt

:: 6. 纯文字对话
python src/ai.py "用一句话解释什么是机器学习"

:: 7. 看图说话（首次运行会自动下载 6 MB 的 YOLO 预训练权重）
python src/ai.py -i data/samples/bus.jpg "这张图里有什么"
```

### 使用示例

```cmd
:: 看图说话：YOLO 负责"看"，大模型负责"说"
python src/ai.py -i data/samples/bus.jpg "这张图里有什么？请用中文回答"

:: 基本提问（默认流式输出，字会一个个蹦出来）
python src/ai.py "用三句话介绍一下你自己"

:: 设定角色
python src/ai.py "帮我写一个冒泡排序" --system "你是一个资深 Python 工程师，回答要简洁"

:: 关闭流式输出（等全部生成完再显示）
python src/ai.py "你好" --no-stream

:: 换一个小模型（更快，但更笨）
python src/ai.py "你好" --model qwen2.5:0.5b
```

---

## 训练（在魔搭免费 GPU 上）

**为什么不在这台笔记本上训练**：本机是双核 CPU、没有可用于训练的显卡（见"已知问题"）。
训 50 轮 coco128 在本机需要数小时，而在免费 GPU 上约 5 分钟。

**为什么选魔搭**：本机**无法访问 Google Colab**（实测超时），网上大多数 YOLO 教程需要跳过。
魔搭（ModelScope）提供免费 GPU 额度，环境基于 PAI-DSW，预装 CUDA 与 PyTorch。

**步骤**：

1. 在 [modelscope.cn](https://www.modelscope.cn/) 注册，进入「我的 Notebook」
2. 新建 Notebook，选择**带 GPU 的环境**，启动
3. 在 JupyterLab 的代码格里执行：

```python
!pip install ultralytics -q
!git clone <本仓库地址> && cd <仓库名>
!python train/train_yolo.py --data coco128.yaml --epochs 50 --batch 16 --device 0
```

4. 训练完成后下载 `runs/train/weights/best.pt`，放进本机 `weights/` 目录
5. 用自己训练的权重推理：

```cmd
python src/ai.py -i data/samples/bus.jpg "这张图里有什么" --weights weights/best.pt
```

**训练参数的选择理由**（题目要求"参数自己决定"，这里给出依据）：

| 参数 | 取值 | 理由 |
|---|---|---|
| 基础模型 | `yolov8n` | 最小最快的版本，本机 CPU 推理能跑得动（约 3.3 秒/张） |
| `epochs` | 50 | 小数据集下足够收敛；过少欠拟合，过多过拟合 |
| `imgsz` | 640 | YOLO 的标准输入尺寸，精度与速度的通用平衡点 |
| `batch` | 16 | 24G 显存下宽裕；若改在本机 CPU 训练则必须调小 |
| `device` | 0 | 使用第 0 号 GPU；本机 CPU 训练填 `cpu` |

> ⚠️ 魔搭的免费额度按使用时长消耗，**训练完请及时停止实例**，并把权重下载到本地。

---

## 已知问题与解决方案

### 问题 1：`ollama run` 崩溃，报 `0xc0000005`

**现象**：模型下载成功，但一运行就报
`500 Internal Server Error: llama-server process has terminated: exit status 0xc0000005`

**定位过程**（这是本项目最耗时的部分，完整记录见 `logs/2026-10-09.md`）：

1. 排除模型损坏 —— sha256 校验通过，重复下载无效
2. 排除"虚拟机缺 AVX 指令集" —— `systeminfo` 确认是物理机，CPU 为 Broadwell 架构，支持 AVX2
3. 排除 VC++ 运行库缺失 —— `vcruntime140.dll` 已是 14.44 版本
4. **读 Windows 事件日志（事件 ID 1000）**，拿到崩溃模块：
   ```
   错误应用程序名称: llama-server.exe
   错误模块名称:     vulkan-1.dll
   异常代码:         0xc0000005
   ```
5. **读 Ollama 引擎日志**，发现它在做一件不该做的事：
   ```
   load_tensors: offloaded 19/29 layers to GPU
   load_tensors:      Vulkan0 model buffer size =  605.40 MiB
   ```
6. **在系统 DriverStore 中找到这块"显卡"的真身**：`amd-vulkan64.json` → `amdvlk64.dll`（2020 年的实现）

**根因**：Ollama 默认把本机的 **AMD R7 M260**（2013 年、2 GB 显存、驱动停留在 2020 年）识别为可用加速卡，
把 19/29 层模型权重搬上去，搬运过程中驱动崩溃，整个推理引擎随之挂掉。**与模型、CPU、内存、Ollama 版本均无关。**

**解决方案**：设置环境变量，禁止 Ollama 使用 Vulkan：

```cmd
setx OLLAMA_VULKAN 0
```

**验证方法**：先做 6 组对照实验确认（`OLLAMA_VULKAN=0`、`OLLAMA_VULKAN=false`、`OLLAMA_LLM_LIBRARY=cpu` 三组均成功；
`GGML_VK_VISIBLE_DEVICES`、`VK_DRIVER_FILES` 两组无效），再固定使用 `OLLAMA_VULKAN=0`。

### 问题 2：Ollama 桌面程序内部服务起不来

**现象**：新版 Ollama 自带桌面界面（`ollama app.exe`），日志报
`ollama server not ready, continuing anyway`，11434 端口始终无监听。

**当前处理**：改用命令行方式常驻服务 `ollama serve`。此问题不影响本项目的 CLI 使用。

---

## 实测性能

### 大模型（qwen2.5:1.5b）

| 指标 | 数值 |
|---|---|
| 模型加载时间 | 约 5 秒（首次） |
| 生成速度 | 约 12.9 tokens/秒 |
| 运行方式 | 纯 CPU（无 GPU 卸载） |
| 内存占用 | 约 1.4 GB（模型 934 MB + 运行时缓冲） |

### YOLO（yolov8n，COCO 80 类预训练权重）

| 指标 | 数值 |
|---|---|
| 单张图片推理 | 约 3.3 秒 |
| 权重文件大小 | 6.25 MB |
| 运行方式 | 纯 CPU |
| 实测样例 | `bus.jpg` 检出公交车 + 3 名行人 + 停车标志 |

---

## 项目结构

```
my_project/
├── README.md              # 本文件：环境、复现步骤、踩坑记录
├── requirements.txt       # Python 依赖
├── .gitignore             # 排除模型权重与训练产物（这类文件不能提交）
├── src/
│   ├── ai.py              # CLI 主程序（大模型对话 + 看图说话）
│   └── detector.py        # YOLO 目标检测模块（只负责"看"）
├── train/
│   └── train_yolo.py      # YOLO 训练脚本（在魔搭免费 GPU 上运行）
├── weights/               # 模型权重（已被 .gitignore 排除，首次运行自动下载）
├── data/
│   └── samples/           # 测试用图片
└── logs/                  # 工程日志：进度、卡点、思路
```

---

## 后续计划

- [x] **YOLO 本地推理跑通**：官方 `yolov8n` 预训练权重，纯 CPU 推理
- [x] **图文合并**：`python src/ai.py -i 图片 "问题"` —— YOLO 负责"看"，LLM 负责"说"
- [ ] **完成一次自己的训练**：在魔搭 ModelScope 免费 GPU 上训练自定义数据集
- [ ] **创意作品**：围绕"看图说话"整合成一个完整作品

---

## AI 使用说明（如实声明）

本项目的开发过程**大量使用了 AI 辅助**（对话式 AI 编程助手），具体分工如下：

- **AI 完成**：定位 `0xc0000005` 崩溃根因（指导查看 Windows 事件日志与 Ollama 引擎日志）、设计并执行 6 组环境变量对照实验、提供 CLI 代码骨架与 README/日志模板、协助编写 YOLO 推理模块与图文流水线
- **本人完成**：执行全部命令、判断与采纳 AI 建议、核对 AI 结论（曾发现 AI 关于"Ollama 版本过旧"的判断有误，经查官方仓库确认 0.40.2 为最新版）、按自己的理解补全代码、撰写与本项目的实际决策

**为什么如实说明**：题目明确表示"即便全程 AI 也不扣分，能用 AI 完成得很好反而是加分项"。
我认为更有价值的是**能够判断 AI 给的信息对不对**——本项目中出现过一次 AI 判断错误（误判版本问题），
是通过自己查证官方仓库纠正的。

---

## 参考链接

- [Ollama 官方网站](https://ollama.com/) ｜ [Windows 下载](https://ollama.com/download/windows)
- [Ollama HTTP API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama 官方仓库（版本查询）](https://github.com/ollama/ollama/releases)
- [Python argparse 教程](https://docs.python.org/zh-cn/3/howto/argparse.html)

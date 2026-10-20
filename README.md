# 本地小助手

一个跑在自己电脑上的网页小工具：**本地大模型对话 + 日程管理**。
模型用 Ollama 在本机跑，日程数据存在本机 JSONL 里，不联网、不调云端 API。

> 最初是课程/面试作业，现在按"自己每天用"的标准维护。
> 2026-10 之前的图像识别、相册分类、Agent 部分已经移除，原因和旧代码见文末「历史」。

---

## 现在有什么

| 页面 | 能干什么 |
|---|---|
| 💬 对话 | 调本地 `qwen2.5:1.5b`，流式输出（打字机效果） |
| 📅 日程表 | 一句话加日程、预览确认、冲突检测、周视图、编辑/删除、导出 `.ics` 到手机 |

日程这块是重点，它不调用大模型：

- 说一句「下周四晚上七点在东区 A201 开班会，大概一小时」→ 解析成结构化事件
- **先预览、逐条检查冲突，你确认后才落库**（日程错了人是按着它去开会的）
- 时间重叠、赶路不够、准备时间撞车 → 硬冲突；一天排太满 → 只警告
- 冲突时给出替代时间（同一天改时刻 → 只挪日期 → 日期时刻都改，三档）
- 周视图按课表的习惯展示，支持编辑、删除
- 导出 `.ics`，手机日历导入后锁屏就能看

---

## 快速开始

```cmd
cd /d D:\desktop\my_project
```

**1. 安装并启动 Ollama**（模型约 1 GB）

```cmd
ollama pull qwen2.5:1.5b
ollama serve
```

**2. 关键：本机必须关掉 Vulkan**（否则 Ollama 会调用老 AMD 显卡驱动并崩溃，详见「已知问题」）

```cmd
setx OLLAMA_VULKAN 0
```

**3. 安装 Python 依赖**（现在只需要 Flask + requests）

```cmd
py -3.11 -m pip install -r requirements.txt
```

**4. Vue 本体没有提交进仓库**（保持仓库体积），首次使用先下载一次：

```cmd
curl -o web\static\vue.global.prod.js https://cdn.staticfile.org/vue/3.4.21/vue.global.prod.min.js
```

**5. 启动**

```cmd
python web\app.py
```

浏览器打开 <http://127.0.0.1:5000>。
如果「对话」页用不了，说明 `ollama serve` 没启动；日程页不依赖模型，可以单独用。

命令行也能聊天：

```cmd
python src\ai.py "用一句话解释什么是机器学习"
python src\ai.py "你好" --no-stream
```

---

## 数据放在哪

- 日程：`data/schedule.jsonl`，追加式一行一条，**已被 git 忽略**（个人日程不会进仓库）
- 编辑/删除也是追加一条记录（`update` / `remove`），不会重写整个文件，方便回看和手改
- 备份就是复制这一个文件

---

## 项目结构

```
my_project/
├── web/
│   ├── app.py                 # Flask 后端：对话 + 日程接口
│   └── static/                # index.html + app.js + style.css + Vue 本体
├── scheduler/                 # 日程内核（不依赖模型）
│   ├── model.py               # 事件模型 + 校验
│   ├── store.py               # 追加式 JSONL 存储
│   ├── conflict.py            # 冲突检测 / 找空档 / 替代方案
│   ├── parser.py              # 自然语言 → 结构化事件（规则解析）
│   ├── ics.py                 # 导出 .ics（按 RFC 5545 字节折行）
│   ├── cli.py                 # 命令行：demo / week / verify / gaps / add / ics
│   └── tests/                 # 内核、解析、API 测试
├── src/ai.py                  # 命令行对话（可选）
├── data/schedule.jsonl        # 本机日程数据（不入库）
├── logs/                      # 早期工程日志（历史记录）
└── requirements.txt
```

`weights/`、`data/` 里的旧图片、`output/` 是图像部分留下的东西，程序已经不用了，可以自行删除。

---

## 测试

```cmd
py -3.11 scheduler\tests\test_scheduler.py
py -3.11 scheduler\tests\test_parser.py
py -3.11 scheduler\tests\test_api.py
```

（这台机器没装 pytest，测试文件自带 runner，直接跑即可。）

---

## 已知问题

### Ollama 一跑就崩溃，报 `0xc0000005`

本机是 2015 年的 Dell，AMD R7 M260 的 Vulkan 驱动（2020 年版本）会让 Ollama 把模型层搬到显卡上然后崩溃。
根因定位过程见 `logs/2026-10-09.md`（事件日志 + Ollama 引擎日志 + 6 组对照实验）。

解决：`setx OLLAMA_VULKAN 0`，然后重开终端、重启 `ollama serve`。

### 其他

- CPU 推理约 13 tokens/秒，长回答会慢，属于硬件上限
- 手机上目前只能在同一个 WiFi 下访问电脑的 `http://电脑IP:5000`，而且电脑要开着
- 编辑重复日程会改**整个系列**，不能只改某一次（下一步要做的事）
- 周视图在窄屏上是横向滚动，还没有做手机优先布局

---

## 历史

这个仓库在 2026-10 之前还包含：

- YOLO 看图说话（检测框 + 大模型描述）
- 自训练 5 类相册分类（独立验证集 Top-1 94.6%）
- Agent Harness（让 1.5B 小模型自己调用工具整理相册）
- 课表照片识别（本地 OCR / 云端视觉模型两条路）

这些功能在 2026-10-20 被整体移除：它们依赖 200 MB+ 的 torch/ultralytics，而且日常并不用。
完整旧代码不再放在仓库历史里（历史已清扫，避免每次 clone 都拖着用不上的模块），
需要时看分支 `archive/pre-2026-10-cleanup`（`git show archive/pre-2026-10-cleanup:src/agent.py`）；当时的踩坑记录仍保留在 `logs/2026-10-*.md`。

---

## AI 使用说明

开发过程使用了 AI 辅助（对话式编程助手）。作者负责：提需求、执行全部命令、核对结论、决定取舍；
AI 负责：排查问题、写代码骨架、整理文档。已知的一次 AI 误判（"Ollama 版本过旧"）由作者查官方仓库后纠正。

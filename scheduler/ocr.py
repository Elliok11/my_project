"""课表照片 → 文字（可插拔的识别后端）

【为什么做成"可插拔 + 默认不可用"，而不是硬写一个能跑的方案】
    课表照片识别有两条路，都需要额外装东西：

        ocr  本地 OCR 库（RapidOCR / PaddleOCR 之类）
             优点：离线、免费、隐私不出本机
             缺点：要装几百 MB 的东西；而且 OCR 只给文字，不理解版面，
                   教务系统那种网格课表识别出来是一堆散字，还要再拼

        vlm  多模态模型直接看图（本地 1.5B 跑不动，得调云端或本地大模型）
             优点：理解版面，"第几节课在哪"这种结构能直接读出来
             缺点：要联网、要额度、图片要传出去

    两个都需要成本。所以这里**不硬凑**：谁装了就用谁，都没装就明确说"不可用"，
    并告诉用户怎么装。**工具假装自己能干，比承认不能干更糟** ——
    用户会以为是自己拍的照不对，而不是功能没配。

【这也是第 4 天那个 A/B 对比的接口】
    两条路都留好了位置（run_backend 按名字分发），
    到时候用同一批课表照片跑两边，比"字段抽全率"，就有结论了。
"""
from __future__ import annotations

import re
from pathlib import Path

# 每个后端的"怎么装"说明。界面上会原样显示给用户。
HOW_TO = {
    "ocr": (
        "本地 OCR：pip install rapidocr-onnxruntime\n"
        "（约 100 MB，纯 CPU 可跑，离线、不上传图片）"
    ),
    "vlm": (
        "多模态模型：在项目根目录建一个 .env 文件（已被 .gitignore 排除），填三行：\n"
        "  VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n"
        "  VLM_API_KEY=你的key\n"
        "  VLM_MODEL=qwen-vl-ocr\n"
        "注意：这条路会把课表照片发到远端，别传含个人信息的图（学号姓名先打码）。"
    ),
}


def _ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _load_env_file() -> dict:
    """读项目根目录的 .env（.gitignore 已排除，不会进仓库）。

    【为什么不用 setx 设系统环境变量】第一版就是这么要求的，对一个学生来说
    这是四步操作 + 要重开终端，实际用起来一定会卡住。
    而"把 key 写在项目里的一个文件、代码自己读"只要一步，而且换机器时
    拷贝项目就带过去了（当然别忘了那个文件不进 git）。
    """
    values: dict[str, str] = {}
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _credential(name: str, env_file: dict) -> str:
    """取一项配置：先看真正的环境变量，再看 .env。环境变量优先。

    两种都支持是因为部署环境不一样：本机开发用 .env 方便，
    服务器上通常走环境变量（配置文件不该出现在生产机上）。
    """
    import os
    return os.environ.get(name) or env_file.get(name, "")


def _vlm_config() -> dict:
    env_file = _load_env_file()
    return {
        "base_url": _credential("VLM_BASE_URL", env_file).rstrip("/"),
        "api_key": _credential("VLM_API_KEY", env_file),
        "model": _credential("VLM_MODEL", env_file) or "qwen-vl-ocr",
    }


def _vlm_available() -> bool:
    config = _vlm_config()
    return bool(config["base_url"] and config["api_key"])


def available_backends() -> dict:
    """哪些后端能用。前端据此禁用按钮，而不是让用户点了才报错。"""
    available = []
    if _ocr_available():
        available.append("ocr")
    if _vlm_available():
        available.append("vlm")
    return {"available": available, "how_to": HOW_TO}


def run_backend(name: str, image_path: Path) -> dict:
    """按名字分发到具体后端。两个后端都返回同样的结构：

        {"text": "识别出来的原始文字", "note": "给用户看的说明"}

    统一返回结构的好处：不管走哪条路，**下游的解析器只认文字**，
    所以新增一个后端不需要改解析器，也不需要改前端。
    """
    if name == "ocr":
        return _run_ocr(image_path)
    if name == "vlm":
        return _run_vlm(image_path)
    raise ValueError(f"不认识的识别后端：{name}")


def _run_ocr(image_path: Path) -> dict:
    """本地 OCR。不只是"认字"，还要**认表**。

    【为什么必须做版面还原（这是实测出来的）】
        直接把 OCR 出来的文字按顺序拼起来喂给解析器，网格课表会 **0 条成功**。
        原因不是它认错字 —— 字全对，一个都没错 —— 而是它按列读：

            高等数学   数据结构   高等数学      ← 三列第一节的课名连在一起
            第1-2节    08:00-09:40              ← 行号在别处
            东区A301   东区A305   东区A301       ← 地点又是一串

        于是"高等数学"配 A301 还是 A305、那串"1-16周"属于哪门课，全丢了。
        **丢的不是字，是"谁和谁在一个格子里"。**

        好消息：OCR 本来就返回每个文本框的**坐标**，只是第一版把坐标扔了。
        有了坐标就能反推行列，把格子还原回来 —— 实测 9 门课全对。
    """
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    result, _ = engine(str(image_path))
    raw = result or []
    if not raw:
        return {"text": "", "note": "图上没识别到任何文字。"}

    boxes = []
    for item in raw:
        points, text = item[0], str(item[1]).strip()
        if not text:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        boxes.append({
            "text": text,
            "x": (min(xs) + max(xs)) / 2,
            "y": (min(ys) + max(ys)) / 2,
        })

    grid_text = _rebuild_grid(boxes)
    if grid_text is not None:
        return {
            "text": grid_text,
            "note": f"本地 OCR 识别出 {len(boxes)} 个文本框，并还原成表格（"
                    f"{len(grid_text.splitlines())} 条日程）。"
                    "请核对下面的内容再点「解析」——表格还原有可能串行。",
        }

    # 不是网格表（比如手机备忘录那种一行一件事），按阅读顺序拼起来就行
    boxes.sort(key=lambda b: b["y"])
    return {
        "text": "\n".join(b["text"] for b in boxes),
        "note": f"本地 OCR 识别出 {len(boxes)} 行文字。"
                "没识别出表格结构，按阅读顺序拼接 —— 请核对后再解析。",
    }


# ── 表头识别 ──
# 找"星期X / 周X / 礼拜X"这种列标题
_WEEKDAY_LABELS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7}
_WEEKDAY_LABELS_INV = {v: k for k, v in _WEEKDAY_LABELS.items()}
_HEADER_RE = re.compile(r"^(?:星期|周|礼拜)\s*([一二三四五六日天])$")
# 节次：真课表里"第1-2节"和光秃秃的"1"两种写法都常见
_PERIOD_RE = re.compile(r"^第?\s*\d{1,2}\s*(?:[-—~、,，]\s*\d{1,2}\s*)?节$|^第?\s*\d{1,2}$")
# 时间：单行的时间点（"08:00"）或时间区间（"08:00-09:40"）
_CLOCK_RE = re.compile(r"^(\d{1,2}:\d{2})$")
_TIME_RANGE_RE = re.compile(r"^(\d{1,2}:\d{2})\s*[-—~至到]\s*(\d{1,2}:\d{2})$")
# 周次说明："1-16周" / "第9-16周"
_WEEKS_RE = re.compile(r"^第?\s*\d+\s*[-—~、,，]?\s*\d*\s*周$")
# 房间号：A301 / B202 这种"字母+数字"，或者真课表里"场地 3605"这种四位数
_ROOM_RE = re.compile(r"[A-Za-z]\s?\d{1,4}")
# 真课表用"场地 3605"标注教室（这是最可靠的来源，比猜四位数准）
_FIELD_ROOM_RE = re.compile(r"场\s*地\s*[:：]?\s*(\d{3,4}[A-Za-z]?)")
# 这一列是"上午/中午/下午/晚上"的分组标签，不是日期
_DAY_PART_WORDS = {"上午", "中午", "下午", "晚上", "早晨", "夜间"}


def _rebuild_grid(boxes: list[dict]) -> str | None:
    """把带坐标的文本框还原成"星期X 时间 事件名 地点"，一行一件事。

    返回 None 表示"这看起来不是网格课表"，交给调用方按普通文字处理。

    【这一版是照着真实课表重写的，第一版太天真】
        第一版假设：节次写作 "第1-2节"、时间写作 "08:00-09:40" 一个框、
        一个格子只有 3 个文本框。结果拿真实教务系统课表一跑，**整段还原失效**，
        界面变成"每个文本框一条日程"，200 多条全是"缺开始时间"。

        真课表长这样（每个假设都破了）：
          节次列：'1' / '08:00' / '08:45'  ← 三个独立文本框，不是"第1-2节"
          内容格：字特别多、折行特别多 → 6~10 个文本框挤在一个格子里
          多一列：'上午'/'中午'/'下午'/'晚上' 的分组标签

        所以现在的做法是**从数据里找锚点，而不是假设格式**：
          列锚点 = 星期几表头的 x 中心
          行锚点 = 左侧"节次列"里所有时间点的 y 中心
          然后把每个内容框按"最近的锚点"归位
    """
    # ── 列锚点：星期几表头
    columns: dict[int, float] = {}
    for box in boxes:
        match = _HEADER_RE.match(box["text"])
        if match:
            columns[_WEEKDAY_LABELS[match.group(1)]] = box["x"]
    if len(columns) < 2:
        return None
    header_y = max(b["y"] for b in boxes if _HEADER_RE.match(b["text"]))
    left_edge = min(columns.values())          # 表头左边 = 内容区左边界

    # ── 行锚点：左侧节次列里的时间点。
    #    "左侧"用表头最左边的 x 判断 —— 节次/时间段列一定在"星期一"左边。
    #    这一步是关键：**时间信息本身就在左侧列里**，它不属于任何一个格子，
    #    而是"整行"的属性。第一版没抓住这点，所以每个格子都拿不到时间。
    clock_boxes = []
    for box in boxes:
        if box["y"] <= header_y or box["x"] >= left_edge:
            continue
        match = _CLOCK_RE.match(box["text"])
        if match:
            clock_boxes.append((box["y"], match.group(1), box["x"]))
    if not clock_boxes:
        return None
    clock_boxes.sort()

    # 同一个小节里"开始/结束"两个时间点，取开始的那个（每行只留一个）
    row_anchors: list[tuple[float, str]] = []
    for y, clock, x in clock_boxes:
        if row_anchors and y - row_anchors[-1][0] < 30:
            continue                            # 这一行的"结束时间"框，跳过
        row_anchors.append((y, clock))
    if not row_anchors:
        return None

    def nearest_column(x: float) -> int | None:
        day = min(columns, key=lambda d: abs(columns[d] - x))
        return day if abs(columns[day] - x) < 300 else None

    def nearest_row(y: float) -> int:
        """按 y 找这一格属于哪一行。

        【这个函数改了三版，把每次的症状记下来】
            ① 固定容差 60px  → 高格子（字多、折行多）的内容溢出后超出容差，被丢掉；
                              同时短格子的空行会被邻行的溢出内容填满
            ② 取消容差、一律取最近 → 还是串行。因为行锚点间距本身很小
                              （08:00 在 y=125、08:45 在 y=182），
                              一个 y=165 的框"离 08:45 更近"，就被判给了下一节
            ③ 现在：**用下一行的锚点当上边界**
                              表格里一行的内容可能因为字多而往下溢出，
                              但**绝对不会溢出到下一行的起始时间线以下**。
                              所以 "y < 下一个锚点" 就是这一行的硬边界。
                              这也解释了为什么第一节课的内容总串到第二节课 ——
                              第二节课格子若为空，第一节课的溢出就落进去了。
        """
        for index, (anchor_y, _clock) in enumerate(row_anchors):
            if index + 1 < len(row_anchors):
                if anchor_y <= y < row_anchors[index + 1][0]:
                    return index
            elif y >= anchor_y:
                return index
        # 落在第一个锚点之上的（表头附近），归给第一行
        return 0 if y < row_anchors[0][0] else len(row_anchors) - 1

    # ── 归位
    cells: dict[tuple[int, int], list[dict]] = {}
    for box in boxes:
        text = box["text"]
        if box["y"] <= header_y:
            continue
        if box["x"] < left_edge:                # 左侧节次/时间段列
            continue
        if text in _DAY_PART_WORDS or _HEADER_RE.match(text) or _CLOCK_RE.match(text):
            continue
        day = nearest_column(box["x"])
        row = nearest_row(box["y"])
        if day is None:
            continue
        cells.setdefault((row, day), []).append(box)

    # ── 每个格子拼成一行
    lines = []
    for (row, day), items in sorted(cells.items()):
        items.sort(key=lambda b: (b["y"], b["x"]))
        blob = " ".join(b["text"] for b in items)
        room = _extract_room(blob)
        name = _extract_course_name(items, room)
        start = row_anchors[row][1]
        if not name:
            continue
        line = f"星期{_WEEKDAY_LABELS_INV[day]} {start} {name}"
        if room:
            line += f" {room}"
        lines.append(line)
    return "\n".join(lines) if lines else None


def _extract_room(blob: str) -> str:
    """从格子内容里挖出教室。

    真课表的教室信息其实散在好几处，按可靠性从高到低试：
        ① "场地：3604"           ← 表格自带的字段，最可靠
        ② "A301"                 ← 字母+数字的房间号
        ③ 独立出现的 3~4 位数字   ← 真课表里 "食品 516" 这种，教室就是裸数字
    第 ③ 条容易和学分/周次搞混，所以**要排除紧跟"学分"的数字**。
    """
    match = _FIELD_ROOM_RE.search(blob)
    if match:
        return match.group(1)
    match = _ROOM_RE.search(blob)
    if match:
        return match.group(0)
    # 裸数字：取最后一个"不是学分/周次/节次"的 3~4 位数
    # （真课表里教室常写在内容末尾，比如 "场地:食品楼 516"）
    candidates = re.findall(r"(?<![\d\-/:：])(\d{3,4})(?![\d\-周])", blob)
    for candidate in reversed(candidates):
        if candidate in blob and f"学分 {candidate}" not in blob and f"学分{candidate}" not in blob:
            return candidate
    return ""


def _extract_course_name(items: list[dict], room: str) -> str:
    """从格子的一堆文本框里挑出课名。

    【真课表为什么难】一个格子里的文字长这样：
        程序设计基础
        1-2节/5-16周/场地 3605/课堂名称 26人工智能1-2.26人工
        智能(智创班)1/考试方式:未安排/学分:3
      课名在第一行，后面全是元数据。所以策略是：
        · 去掉纯数字（节次号）、纯时间、场地号
        · 去掉任何含"学分/考试方式/课堂名称/周次"的段落
        · 剩下的拼起来 —— 长课名折行成 2~3 段，要拼回去
    """
    parts = []
    for box in items:
        text = box["text"].strip()
        if not text or _WEEKS_RE.match(text) or _CLOCK_RE.match(text):
            continue
        if re.fullmatch(r"\d{1,2}", text):                  # 节次号
            continue
        if re.fullmatch(r"[\d\s,，、\-—/:：()（）]+", text):  # 纯数字标点
            continue
        if text.startswith("场") or re.fullmatch(r"\d{3,4}[A-Za-z]?", text):
            continue
        if room and room in text and len(text) <= len(room) + 4:
            continue
        # 元数据段落：出现了就整段丢掉
        if any(word in text for word in ("学分", "考试方式", "课堂名称", "课堂 名称", "节/")):
            continue
        parts.append(text)
    name = " ".join(parts).strip()
    return re.sub(r"\s+", " ", name)


def _looks_like_name(text: str) -> bool:
    """粗判一段文字更像"课名"还是"地点"。

    只用于在同一个格子里决定"哪个是地点"。判错了也无所谓 ——
    地点填错用户一眼就看得出来，而下一条会把正确的地点再放进去。
    """
    return len(text) > 6 and not any(ch.isdigit() for ch in text)


def _shrink_for_upload(image_path: Path, max_side: int = 1400) -> tuple[bytes, str]:
    """把图片压到长边不超过 max_side，返回 (字节内容, MIME 里的格式名)。

    【为什么要压】
        实测：一张 1080x1526 的课表，deepseek-flash 读了 **82 秒**。
        视觉模型是按图片分辨率折算 token 的，图越大越慢越贵。
        而课表这种"文字+表格"的图，压到长边 1400 后字依然够清楚 ——
        这一刀砍的是速度，不是准确率。

    【为什么不无脑转 JPEG】
        转换会丢一次质量。如果原图已经是 JPEG，就只缩放、不换格式；
        如果是 PNG，缩放后仍存 PNG（保证不引入 JPEG 压缩噪点，
        因为 OCR 对文字边缘的噪点很敏感）。
        质量参数按格式分别给：JPEG 用 85 已经看不出差别，PNG 用 optimize。
    """
    from io import BytesIO

    from PIL import Image

    format_name = (image_path.suffix.lstrip(".").lower() or "jpeg")
    mime = "jpeg" if format_name in ("jpg", "jpeg") else format_name
    raw = image_path.read_bytes()

    try:
        with Image.open(BytesIO(raw)) as img:
            width, height = img.size
            longest = max(width, height)
            if longest <= max_side:
                return raw, mime            # 本来就不大，原样上传，别白丢一次质量

            scale = max_side / longest
            resized = img.resize((max(1, int(width * scale)), max(1, int(height * scale))),
                                 Image.LANCZOS)
            buffer = BytesIO()
            if mime == "jpeg":
                resized.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
            elif mime == "png":
                resized.save(buffer, format="PNG", optimize=True)
            else:
                resized.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
                mime = "jpeg"
            return buffer.getvalue(), mime
    except Exception:                                              # noqa: BLE001
        # 压缩失败不该让整个功能挂掉 —— 原图上传照样能用，只是慢一点
        return raw, mime


def _run_vlm(image_path: Path) -> dict:
    """把图片发给多模态模型，要求它吐出可直接解析的日程文字。

    关键：**让模型输出"人话"，而不是 JSON**。
    因为它输出的人话正好喂给我方已有的解析器（parser.py），
    那条路已经被 35 个测试覆盖过了。让模型直接吐 JSON 等于绕开所有测试 ——
    而且模型吐的 JSON 一旦格式飘了，报错信息会指向解析器，让你查错方向。
    """
    import base64
    import time

    import requests
    from requests.exceptions import ConnectionError as ReqConnectionError
    from requests.exceptions import SSLError as ReqSSLError

    config = _vlm_config()
    if not (config["base_url"] and config["api_key"]):
        raise RuntimeError("多模态模型没配置：请在项目根目录的 .env 里填 VLM_BASE_URL 和 VLM_API_KEY")

    payload_bytes, mime = _shrink_for_upload(image_path)
    encoded = base64.b64encode(payload_bytes).decode("ascii")
    original_kb = image_path.stat().st_size // 1024
    sent_kb = len(payload_bytes) // 1024

    prompt = (
        "这是一张课程表截图。请把它转写成一行一件事的文字，每行格式：\n"
        "星期几 开始时间 课程名 教室\n"
        "例如：\n"
        "星期三 09:55 国家安全教育与军事理论 3212\n"
        "要求：\n"
        "1. 只输出这些行，不要表头、不要解释、不要 markdown 代码块\n"
        "2. 开始时间用 24 小时制的 HH:MM\n"
        "3. 课程名写完整，不要截断\n"
        "4. 空的格子（没有课）不要输出\n"
        "5. 如果某个格子里的信息看不清，那一行宁可不写，也不要猜\n"
    )
    body = {
        "model": config["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{encoded}"}},
            ],
        }],
        "temperature": 0,
    }
    url = f"{config['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {config['api_key']}"}

    # ── 重试 ──
    # 【为什么必须重试，而不是把错误抛给用户】
    #   实测：这台机器上到 api.deepseek.com 的 TLS 握手**时通时不通**
    #   （连续 5 次探测全通，但用户点一次上传就撞上 SSLEOFError）。
    #   这是网络中间设备（机器上装了深信服的上网行为管理）偶发打断 TLS 造成的，
    #   客户端解决不了根因 —— 但"再试一次"几乎总能成功。
    #   所以：偶发错误要重试，而且要把"这是网络抖动"和"你的 key 不对"
    #   在报错信息里区分开，否则用户会去怀疑 key。
    attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=240)
        except (ReqSSLError, ReqConnectionError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(1.5 * attempt)          # 退避一下再试，别立刻重撞
                continue
            raise RuntimeError(
                f"连不上模型服务（{type(exc).__name__}），重试 {attempts} 次都失败。\n"
                "这多半是**网络中间设备打断了 TLS 握手**，不是 key 的问题"
                "（判断依据：TCP 能连通，但 HTTPS 握手偶尔中断）。\n"
                "可以试试：① 关掉代理/上网行为管理软件后重试；"
                "② 过一两分钟再点一次；③ 在本页切到「本地 OCR」（离线，不走网络）。"
            ) from exc

        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
            note = (f"由多模态模型（{config['model']}）读图生成 {len(lines)} 条，"
                    f"请核对后再点「解析」——模型会看错行，但看得出来。")
            if sent_kb < original_kb:
                note += f"（图片已从 {original_kb}KB 压到 {sent_kb}KB 以加快速度）"
            return {"text": "\n".join(lines), "note": note}

        # 服务端错误（5xx / 429）也重试；4xx 是请求本身有问题，重试没意义
        if resp.status_code >= 500 or resp.status_code == 429:
            last_error = RuntimeError(resp.text[:200])
            if attempt < attempts:
                time.sleep(2 * attempt)
                continue

        # 把服务商的原始错误带出来 —— 否则用户只看到"请求失败"，
        # 分不清是 key 不对、模型没开通、还是图片太大。
        hint = ""
        if resp.status_code == 401:
            hint = "\n（401 = 密钥无效或已过期。去服务商后台确认一下 API key。）"
        elif resp.status_code == 404:
            hint = f"\n（404 = 模型名 {config['model']} 在这个平台上不存在，检查 .env 里的 VLM_MODEL。）"
        elif resp.status_code == 400 and "image" in resp.text.lower():
            hint = "\n（400 且提到 image = 图片格式或大小不被接受，可以试着先裁小一点。）"
        raise RuntimeError(f"模型服务返回 {resp.status_code}：{resp.text[:300]}{hint}")

    raise RuntimeError(f"模型服务调用失败：{last_error}")

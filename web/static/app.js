/* 相册 AI 工具箱 —— Vue 3 前端逻辑
 *
 * 设计要点：
 *   1. 对话用 fetch + ReadableStream 做流式输出（打字机效果）
 *   2. 相册分类的"置信度阈值"放在前端实时计算 —— 拖滑块能立刻看到
 *      图片在「其他」和正常分类之间移动，不用重新上传
 */
const { createApp, ref, computed, onMounted } = Vue

createApp({
  setup() {
    /* ══════════ 通用 ══════════ */
    const tabs = [
      { id: 'chat',     name: '对话',     icon: '💬' },
      { id: 'vision',   name: '看图说话', icon: '🔍' },
      { id: 'album',    name: '相册分类', icon: '🗂️' },
      { id: 'schedule', name: '日程表',   icon: '📅' },
      { id: 'agent',    name: 'Agent',    icon: '🤖' },
    ]
    const tab = ref('chat')
    const dragging = ref(false)
    const status = ref({ ollama: false, model: '' })
    const agentTools = ref([])          // 可用工具清单（Agent 页用，这里提前声明避免时序问题）
    const agentSuggestions = ref([])    // 动态建议（后端扫描 data/ 目录生成）
    const statusClass = computed(() => (status.value.ollama ? 'ok' : 'bad'))

    onMounted(async () => {
      try {
        status.value = await (await fetch('/api/status')).json()
      } catch (e) {
        status.value = { ollama: false, model: '' }
      }
      try {
        agentTools.value = await (await fetch('/api/agent/tools')).json()
      } catch (e) {
        agentTools.value = []
      }
      try {
        agentSuggestions.value = await (await fetch('/api/agent/suggestions')).json()
      } catch (e) {
        agentSuggestions.value = []
      }
      try {
        schedBackends.value = await (await fetch('/api/schedule/backends')).json()
      } catch (e) {
        schedBackends.value = { available: [], how_to: {} }
      }
      await loadWeek('')
    })

    /**
     * 统一的 POST 请求：即使后端返回 HTML 错误页，也能给出可读提示。
     * （Flask 默认对 413/500 返回 HTML，直接 .json() 会报 "Unexpected token '<'"）
     */
    async function postJSON(url, options) {
      const resp = await fetch(url, options)
      const text = await resp.text()
      try {
        return JSON.parse(text)
      } catch (e) {
        const hint = resp.status === 413
          ? '上传的图片总量太大，请分批上传（比如一次 30 张）'
          : '服务器返回了非 JSON 内容'
        return { error: `${hint}（HTTP ${resp.status}）` }
      }
    }

    /* ══════════ ① 对话（流式输出）══════════ */
    const messages = ref([])
    const chatInput = ref('')
    const sending = ref(false)

    async function sendChat() {
      const text = chatInput.value.trim()
      if (!text || sending.value) return

      messages.value.push({ role: 'user', content: text })
      chatInput.value = ''
      const reply = { role: 'bot', content: '', streaming: true }
      messages.value.push(reply)
      sending.value = true

      try {
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: text }),
        })
        if (!resp.ok) {
          const raw = await resp.text()
          let msg = raw
          try { msg = JSON.parse(raw).error || raw } catch (e) { /* 保持原文 */ }
          reply.content = '[错误] ' + msg
          return
        }
        const reader = resp.body.getReader()
        const decoder = new TextDecoder('utf-8')
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          reply.content += decoder.decode(value, { stream: true })
          messages.value = [...messages.value]      // 触发视图更新
        }
      } catch (err) {
        reply.content += '\n\n[请求失败] ' + err
      } finally {
        reply.streaming = false
        sending.value = false
        messages.value = [...messages.value]
      }
    }

    /* ══════════ ② 看图说话 ══════════ */
    const visionImage = ref('')
    const visionBoxes = ref([])
    const visionSize = ref({ w: 1, h: 1 })
    const visionDesc = ref('')
    const visionLoading = ref(false)
    const visionError = ref('')

    /** 把检测框坐标换算成百分比，这样图片缩放时框也跟着走 */
    function boxStyle(box) {
      const W = visionSize.value.w || 1
      const H = visionSize.value.h || 1
      const [x1, y1, x2, y2] = box.box
      return {
        left:   (x1 / W * 100) + '%',
        top:    (y1 / H * 100) + '%',
        width:  ((x2 - x1) / W * 100) + '%',
        height: ((y2 - y1) / H * 100) + '%',
      }
    }

    async function runVision(file) {
      if (!file) return
      visionLoading.value = true
      visionError.value = ''
      visionImage.value = ''
      visionBoxes.value = []
      visionDesc.value = ''

      const fd = new FormData()
      fd.append('image', file)
      try {
        const data = await postJSON('/api/vision', { method: 'POST', body: fd })
        if (data.error) { visionError.value = data.error; return }
        visionImage.value = data.image
        visionBoxes.value = data.boxes || []
        visionSize.value = { w: data.width || 1, h: data.height || 1 }
        visionDesc.value = data.description || ''
      } catch (err) {
        visionError.value = '请求失败：' + err
      } finally {
        visionLoading.value = false
      }
    }
    function onVisionPick(e) { runVision(e.target.files[0]); e.target.value = '' }
    function onVisionDrop(e) { dragging.value = false; runVision(e.dataTransfer.files[0]) }

    /* ══════════ ③ 相册分类 ══════════ */
    const albumRaw = ref([])          // 后端返回的原始结果（含置信度）
    const albumLoading = ref(false)
    const albumError = ref('')
    const albumProgress = ref('')     // 分批上传进度，例如 "40/120"
    const minConf = ref(0.6)
    const albumRunId = ref('')        // 本次上传的批次号，用于「保存结果」
    const saveState = ref('')         // '' | 'saving' | 保存成功后的路径
    const saveError = ref('')

    /** 生成批次号，例如 20261011_213045 */
    function newRunId() {
      const d = new Date()
      const p = (n) => String(n).padStart(2, '0')
      return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`
           + `_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
    }

    /** 按当前阈值算出每张图归到哪一类（阈值改了会自动重算） */
    const albumResults = computed(() => albumRaw.value.map(r => ({
      ...r,
      category: r.conf < minConf.value ? '其他' : r.raw_category,
      uncertain: r.conf < minConf.value,
    })))

    /** 分类统计（也随阈值实时变化） */
    const albumStats = computed(() => {
      const order = ['截图', '文档', '风景', '表情包', '美食', '其他']
      const counts = {}
      albumResults.value.forEach(r => {
        counts[r.category] = (counts[r.category] || 0) + 1
      })
      return order.filter(c => counts[c]).map(c => ({ category: c, count: counts[c] }))
    })

    async function runAlbum(files) {
      if (!files || !files.length) return
      albumLoading.value = true
      albumError.value = ''
      albumRaw.value = []
      saveState.value = ''
      saveError.value = ''
      albumRunId.value = newRunId()

      // 自动分批上传：一次丢几百张也不会超过服务器上限，
      // 而且每批回来就立刻上屏，不用等全部传完
      const BATCH_SIZE = 40
      const list = Array.from(files).slice(0, 500)
      const collected = []

      try {
        for (let i = 0; i < list.length; i += BATCH_SIZE) {
          const chunk = list.slice(i, i + BATCH_SIZE)
          albumProgress.value = `${Math.min(i + BATCH_SIZE, list.length)}/${list.length}`
          const fd = new FormData()
          chunk.forEach(f => fd.append('images', f))
          fd.append('run_id', albumRunId.value)

          const data = await postJSON('/api/classify', { method: 'POST', body: fd })
          if (data.error) { albumError.value = data.error; return }
          collected.push(...data.results)
          albumRaw.value = [...collected]        // 逐批上屏
        }
      } catch (err) {
        albumError.value = '请求失败：' + err
      } finally {
        albumLoading.value = false
        albumProgress.value = ''
      }
    }

    /** 把当前界面上的分类结果（含阈值调整后的「其他」）保存到 output 文件夹 */
    async function saveAlbum() {
      if (!albumRunId.value || !albumResults.value.length) return
      saveState.value = 'saving'
      saveError.value = ''

      const items = albumResults.value.map(r => ({
        stored_name: r.stored_name,
        category: r.category,
      }))
      const data = await postJSON('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: albumRunId.value, items }),
      })
      if (data.error) { saveState.value = ''; saveError.value = data.error; return }
      saveState.value = data.path
    }
    function onAlbumPick(e) { runAlbum(e.target.files); e.target.value = '' }
    function onAlbumDrop(e) { dragging.value = false; runAlbum(e.dataTransfer.files) }

    /* ══════════ ⑤ 日程表 ══════════
     *
     * 这块的定位和 Agent 页不一样：
     *   Agent 页 = 让模型自己决定调哪个工具（展示 harness 的容错能力）
     *   日程页   = 解析 → **用户肉眼确认** → 才落库（展示"让代码裁决"这条原则）
     * 两者共用同一个排程内核。
     *
     * 为什么不做成"直接加进去"：日程一旦解析错了，人是按着它去开会的。
     * 所以中间必须有一道人工确认，把解析的不确定性摊在用户眼前。
     */

    // 周视图的时间轴范围：07:00 ~ 23:00，一屏看完一天的作息
    const GRID_HOURS = 16
    const HOUR_PX = 52
    const GRID_START_HOUR = 7

    const schedWeek = ref(null)
    const schedAnchor = ref('')          // 当前周里的任意一天

    /**
     * 把 Date 转成 "YYYY-MM-DD"，**用本地时区**。
     *
     * 【为什么不能直接用 toISOString().slice(0,10)】
     *   toISOString() 返回的是 **UTC** 时间。本机是 UTC+8，
     *   于是"本地周一 00:00"会被转成"UTC 周日 16:00"，日期整整少一天：
     *
     *       new Date('2026-10-12T00:00:00')      → 本地 Mon Oct 12 00:00 GMT+0800
     *       同一个时刻 .toISOString()             → 2026-10-11T16:00:00.000Z
     *       .slice(0, 10)                        → "2026-10-11"   ← 变成周日了
     *
     *   而服务端拿到日期后会算"这一周的周一"，于是一来一回就偏了一周。
     *   实测症状：点「上一周」会往前跳两周，点「下一周」看起来没反应。
     *
     *   这类 bug 的坑在于**只在东时区出现**（西半球 offset 是负的，
     *   本地 00:00 转 UTC 是同一天，反而正常）。所以"我这儿测着没问题"
     *   一点都不保险 —— 凡是要把本地日期转字符串，都得走这个方法。
     */
    function toLocalDateString(d) {
      const pad = (n) => String(n).padStart(2, '0')
      return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
    }
    const schedText = ref('')
    const schedParsing = ref(false)
    const schedUploading = ref(false)
    const schedPreview = ref([])
    const schedError = ref('')
    const schedBackends = ref({ available: [], how_to: {} })
    // 默认选中哪个识别后端。**默认选云端视觉模型** —— 实测：
    // 网格课表（教务系统那种）本地 OCR 结构还原只有 7/19 行，云端 12/12 全对。
    // 本地 OCR 的价值在于离线、免费、不上传图片，所以留着让用户自己切。
    const schedBackend = ref('vlm')
    // 用本地时区拿"今天"。不能写成 new Date().toISOString().slice(0,10) ——
    // 那在东八区会在**每天早上 8 点前**返回前一天（详见 toLocalDateString 的注释）。
    const schedToday = ref(toLocalDateString(new Date()))

    const pickedCount = computed(() => schedPreview.value.filter(p => p.picked).length)

    async function loadWeek(anchor) {
      schedError.value = ''
      const url = '/api/schedule/week' + (anchor ? '?date=' + anchor : '')
      const data = await postJSON(url)
      if (data.error) { schedError.value = data.error; return }
      schedWeek.value = data
      schedAnchor.value = data.monday
      if (data.today) schedToday.value = data.today
    }



    function shiftWeek(step) {
      if (!schedAnchor.value) return
      const d = new Date(schedAnchor.value + 'T00:00:00')
      d.setDate(d.getDate() + step * 7)
      loadWeek(toLocalDateString(d))
    }
    function gotoThisWeek() { loadWeek(schedToday.value) }

    /** 计算一条日程在周视图里的位置（按时间绝对定位） */
    function evStyle(item) {
      const [h, m] = item.occ_start.split(':').map(Number)
      const offsetMin = (h * 60 + m) - GRID_START_HOUR * 60
      const top = Math.max(0, offsetMin / 60 * HOUR_PX)
      return { top: top + 'px', height: evHeight(item) + 'px' }
    }
    function evHeight(item) {
      const [sh, sm] = item.occ_start.split(':').map(Number)
      const [eh, em] = item.occ_end.split(':').map(Number)
      const minutes = Math.max(20, (eh * 60 + em) - (sh * 60 + sm))
      return minutes / 60 * HOUR_PX
    }

    /** 解析结果逐条去问内核"这个时间行不行" —— 冲突摆出来，别等加完才发现 */
    async function checkVerdicts() {
      for (const p of schedPreview.value) {
        if (!p.start) continue
        const data = await postJSON('/api/schedule/check', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(p),
        })
        if (!data.error) p.verdict = data.verdict
      }
      schedPreview.value = [...schedPreview.value]
    }

    async function parseScheduleText() {
      const text = schedText.value.trim()
      if (!text || schedParsing.value) return
      schedParsing.value = true
      schedError.value = ''
      try {
        const data = await postJSON('/api/schedule/parse', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })
        if (data.error) { schedError.value = data.error; return }
        schedPreview.value = data.items.map(it => ({ ...it, picked: it.ok }))
        schedText.value = ''
        await checkVerdicts()
      } finally {
        schedParsing.value = false
      }
    }

    /** 上传课表照片 → 后端识别出文字 → 走同一条解析路 */
    async function uploadSchedulePhoto(file) {
      if (!file) return
      schedUploading.value = true
      schedError.value = ''
      // 后端没得选（只装了一个）时用那个；有得选就用用户挑的
      const list = schedBackends.value.available || []
      const backend = list.includes(schedBackend.value) ? schedBackend.value : list[0]
      if (!backend) {
        schedError.value = '没有可用的识别后端。'
        schedUploading.value = false
        return
      }
      const fd = new FormData()
      fd.append('image', file)
      fd.append('backend', backend)
      try {
        const data = await postJSON('/api/schedule/photo', { method: 'POST', body: fd })
        if (data.error) {
          // 把后端的"怎么开"一起显示出来，别只说一句"不可用"
          schedError.value = data.error + (data.how_to && data.how_to[backend]
            ? '  开启方式：' + data.how_to[backend] : '')
          return
        }
        schedText.value = data.text || ''
        if (data.note) schedError.value = ''
        await parseScheduleText()
      } finally {
        schedUploading.value = false
      }
    }
    function onSchedulePhoto(e) {
      uploadSchedulePhoto(e.target.files[0])
      e.target.value = ''
    }

    /** 后端选择器上的一句话说明 —— 把取舍讲清楚，别让用户盲选 */
    function backendHint(name) {
      if (name === 'vlm') return '看得懂版面，网格课表也能还原；图片会发到云端（约 1 分钟）'
      if (name === 'ocr') return '离线、免费、不上传图片；但网格课表会串行（约 20 秒）'
      return ''
    }

    function useAlternative(item, alt) {
      item.start = alt.start
      item.duration_min = Math.max(5, (new Date(alt.end) - new Date(alt.start)) / 60000)
      // 改完立刻重算裁决，让用户看到"现在没问题了"
      postJSON('/api/schedule/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(item),
      }).then(data => {
        if (!data.error) { item.verdict = data.verdict; schedPreview.value = [...schedPreview.value] }
      })
    }

    async function addParsed() {
      const items = schedPreview.value.filter(p => p.picked)
      if (!items.length) return
      const data = await postJSON('/api/schedule/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
      })
      if (data.error) { schedError.value = data.error; return }
      schedPreview.value = []
      // 加完跳回"这一周"，否则用户加了下周的日程但还停在原来那一周，会以为没生效
      const first = data.added[0]
      await loadWeek(first ? first.start.slice(0, 10) : schedToday.value)
    }

    async function delEvent(item) {
      if (!confirm(`删除「${item.title}」？`)) return
      const data = await postJSON('/api/schedule/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: item.id }),
      })
      if (data.error) { schedError.value = data.error; return }
      await loadWeek(schedAnchor.value)
    }

    /* ══════════ ④ Agent（harness 演示）══════════ */
    const agentInput = ref('')
    const agentRunning = ref(false)
    const agentMessages = ref([])       // 左侧对话
    const agentSteps = ref([])          // 右侧执行过程
    const agentError = ref('')
    const agentExamples = [
      '帮我整理一下 data/my_photos 这个文件夹',
      '看看 data/samples/bus.jpg 里有什么',
      'data/my_photos 里有多少张照片',
    ]

    function askExample(text) {
      agentInput.value = text
      runAgent()
    }

    /** 把一条流式事件渲染到界面上 */
    function handleAgentEvent(ev, reply) {
      switch (ev.type) {
        case 'start':
          break
        case 'thinking':
          agentSteps.value.push({ kind: 'thinking', text: `第 ${ev.step} 步：模型思考中…` })
          break
        case 'tool_call':
          agentSteps.value.push({ kind: 'call', text: `调用工具 ${ev.name}`, args: ev.args })
          break
        case 'tool_result':
          agentSteps.value.push({ kind: ev.ok ? 'ok' : 'fail', text: ev.summary })
          break
        case 'nudge':
        case 'repeat':
        case 'summarizing':
          agentSteps.value.push({ kind: 'note', text: ev.text || '收尾中…' })
          break
        case 'answer':
          reply.content = ev.text
          agentMessages.value = [...agentMessages.value]
          break
        case 'error':
          agentError.value = ev.text
          break
      }
      agentSteps.value = [...agentSteps.value]
    }

    async function runAgent() {
      const question = agentInput.value.trim()
      if (!question || agentRunning.value) return

      agentMessages.value.push({ role: 'user', content: question })
      agentInput.value = ''
      agentSteps.value = []
      agentError.value = ''
      agentRunning.value = true
      const reply = { role: 'bot', content: '' }
      agentMessages.value.push(reply)

      try {
        const resp = await fetch('/api/agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        })
        if (!resp.ok) {
          const raw = await resp.text()
          let msg = raw
          try { msg = JSON.parse(raw).error || raw } catch (e) { /* 保持原文 */ }
          agentError.value = msg
          return
        }

        // 逐行读 NDJSON 流：每读到一行就立刻渲染，所以界面是"实时"的
        const reader = resp.body.getReader()
        const decoder = new TextDecoder('utf-8')
        let buffer = ''
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop()                       // 最后一段可能是半行，留到下次
          for (const line of lines) {
            if (!line.trim()) continue
            let ev
            try { ev = JSON.parse(line) } catch (e) { continue }
            handleAgentEvent(ev, reply)
          }
        }
      } catch (err) {
        agentError.value = '请求失败：' + err
      } finally {
        agentRunning.value = false
        agentMessages.value = [...agentMessages.value]
      }
    }

    return {
      tabs, tab, dragging, status, statusClass,
      messages, chatInput, sending, sendChat,
      visionImage, visionBoxes, visionDesc, visionLoading, visionError,
      boxStyle, onVisionPick, onVisionDrop,
      albumResults, albumStats, albumLoading, albumProgress, albumError, minConf,
      albumRunId, saveState, saveError, saveAlbum,
      onAlbumPick, onAlbumDrop,
      GRID_HOURS, HOUR_PX,
      schedWeek, schedText, schedParsing, schedUploading, schedPreview, schedError,
      schedBackends, schedBackend, backendHint, pickedCount, evStyle, evHeight,
      shiftWeek, gotoThisWeek, parseScheduleText, onSchedulePhoto, useAlternative,
      addParsed, delEvent,
      agentTools, agentSuggestions, agentInput, agentRunning, agentMessages, agentSteps,
      agentError, agentExamples, askExample, runAgent,
    }
  },
}).mount('#app')

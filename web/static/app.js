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
      { id: 'chat',   name: '对话',     icon: '💬' },
      { id: 'vision', name: '看图说话', icon: '🔍' },
      { id: 'album',  name: '相册分类', icon: '🗂️' },
      { id: 'agent',  name: 'Agent',    icon: '🤖' },
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
      agentTools, agentSuggestions, agentInput, agentRunning, agentMessages, agentSteps,
      agentError, agentExamples, askExample, runAgent,
    }
  },
}).mount('#app')

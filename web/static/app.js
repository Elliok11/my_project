/* 本地小助手 —— Vue 3 前端逻辑
 *
 * 设计要点：
 *   1. 对话用 fetch + ReadableStream 做流式输出（打字机效果）
 *   2. 周视图在前端算时间轴位置，后端只返回"这一周有哪些事件"
 *   3. 解析结果先预览、逐条问内核"行不行"，用户确认后才落库
 */
const { createApp, ref, computed, onMounted } = Vue

createApp({
  setup() {
    /* ══════════ 通用 ══════════ */
    const tabs = [
      { id: 'chat',     name: '对话',   icon: '💬' },
      { id: 'schedule', name: '日程表', icon: '📅' },
    ]
    const tab = ref('chat')
    const status = ref({ ollama: false, model: '' })
    const statusClass = computed(() => (status.value.ollama ? 'ok' : 'bad'))

    onMounted(async () => {
      try {
        status.value = await (await fetch('/api/status')).json()
      } catch (e) {
        status.value = { ollama: false, model: '' }
      }
      await loadWeek('')
    })

    /**
     * 统一的 POST 请求：即使后端返回 HTML 错误页，也能给出可读提示。
     * （Flask 默认对 500 返回 HTML，直接 .json() 会报 "Unexpected token '<'"）
     */
    async function postJSON(url, options) {
      const resp = await fetch(url, options)
      const text = await resp.text()
      try {
        return JSON.parse(text)
      } catch (e) {
        return { error: `服务器返回了非 JSON 内容（HTTP ${resp.status}）` }
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

    /* ══════════ ② 日程表 ══════════
     *
     * 解析 → 用户肉眼确认 → 才落库。日程一旦错了，人是按着它去开会的，
     * 所以中间必须有一道人工确认。冲突检测全部由 scheduler/conflict.py 裁决。
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
    const schedPreview = ref([])
    const schedError = ref('')
    const schedEditing = ref(null)
    const schedSaving = ref(false)
    const schedNotice = ref('')          // 保存成功但有话说（比如冲突）
    const schedEditingSeriesStart = ref('')   // 编辑中的重复日程：系列真实起点
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

    /**
     * 点开编辑面板。
     *
     * 【为什么重复日程要把开始时间换成"这一周的那一天"】
     *   表单里显示的是 schedEditing.start，而后端按这个日期判断星期几是否和 weekdays 对得上。
     *   如果拿系列原始起点（比如 9 月的某个周一），在 10 月这一周点编辑，
     *   用户既看不出来改的是哪一次，拖动日期也大概率跟 weekdays 冲突、直接 400。
     *   所以把日期换成本周这一次的日期（时刻不变），星期几自然落在 weekdays 里。
     */
    function editEvent(item) {
      let start = item.start
      if (item.recurrence === 'weekly' && item.occ_start) {
        const day = (schedWeek.value && schedWeek.value.days || [])
          .find(d => d.items && d.items.some(it => it.id === item.id && it.occ_start === item.occ_start))
        if (day) start = day.date + 'T' + item.occ_start
      }
      // 【表单里的 start 是"这一次的日期"，不能直接当系列起点提交】
      //   直接提交的话，后端会以为系列要从这一天开始重复，
      //   前面几周那几次就消失了。所以把系列起点单独记住，保存时还原回去。
      schedEditing.value = { ...item, start }
      schedEditingSeriesStart.value = item.start
      schedNotice.value = ''
      schedError.value = ''
    }

    async function saveEditedEvent() {
      const draft = schedEditing.value
      if (!draft || schedSaving.value) return
      schedSaving.value = true
      schedError.value = ''

      const payload = { ...draft, start: schedEditingSeriesStart.value || draft.start }
      const data = await postJSON('/api/schedule/edit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      schedSaving.value = false
      if (data.error) { schedError.value = data.error; return }
      schedEditing.value = null
      await loadWeek(schedAnchor.value)
      // 冲突不阻止保存（和"添加"保持一致）：存进去，再把问题摆出来
      schedNotice.value = data.issues && data.issues.length
        ? '已保存，但和已有日程有冲突：' + data.issues.join('；')
        : ''
    }

    return {
      tabs, tab, status, statusClass,
      messages, chatInput, sending, sendChat,
      GRID_HOURS, HOUR_PX,
      schedWeek, schedText, schedParsing, schedPreview, schedError,
      schedEditing, schedSaving, schedNotice, pickedCount, evStyle, evHeight,
      shiftWeek, gotoThisWeek, parseScheduleText, useAlternative,
      addParsed, delEvent, editEvent, saveEditedEvent,
    }
  },
}).mount('#app')

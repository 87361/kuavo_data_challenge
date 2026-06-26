function formatLogLine(log) {
  const date = log.time ? new Date(log.time) : new Date();
  const time = Number.isNaN(date.getTime()) ? "--:--:--" : date.toLocaleTimeString();
  const level = String(log.level || "info").toUpperCase().padEnd(5, " ");
  const source = String(log.source || "app");
  return `[${time}] ${level} ${source}: ${log.message || ""}`;
}

export function createTerminalController(ctx) {
  const { state, els } = ctx;
  const maxLines = 500;
  const lines = [];

  function isNearBottom() {
    const el = els.terminalLog;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  }

  function renderLine(line, { preserveScroll = false } = {}) {
    const shouldScroll = !preserveScroll && isNearBottom();
    lines.push(line);
    while (lines.length > maxLines) lines.shift();
    els.terminalLog.textContent = lines.join("\n");
    if (shouldScroll) els.terminalLog.scrollTop = els.terminalLog.scrollHeight;
  }

  function log(level, source, message, fields = {}) {
    renderLine(formatLogLine({ level, source, message, ...fields }));
  }

  function clear() {
    lines.length = 0;
    els.terminalLog.textContent = "";
  }

  async function pollLogs() {
    const response = await fetch(`/api/logs?after=${state.terminalSeq}`);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const payload = await response.json();
    for (const item of payload.logs || []) {
      state.terminalSeq = Math.max(state.terminalSeq, Number(item.seq) || 0);
      renderLine(formatLogLine(item));
    }
    if (Number.isFinite(Number(payload.next_seq))) {
      state.terminalSeq = Math.max(state.terminalSeq, Number(payload.next_seq));
    }
  }

  function start() {
    log("info", "terminal", "Application log panel ready");
    els.clearTerminal.addEventListener("click", clear);
    window.addEventListener("lerobot-log", (event) => {
      const detail = event.detail || {};
      log(detail.level || "info", detail.source || "app", detail.message || "");
    });
    window.addEventListener("error", (event) => {
      log("error", "frontend", event.message || "Uncaught frontend error");
    });
    window.addEventListener("unhandledrejection", (event) => {
      const reason = event.reason instanceof Error ? event.reason.message : String(event.reason || "Unhandled promise rejection");
      log("error", "frontend", reason);
    });
    state.terminalPoll = setInterval(() => {
      pollLogs().catch((error) => log("error", "logs", error.message));
    }, 1500);
    pollLogs().catch((error) => log("error", "logs", error.message));
  }

  return { clear, log, pollLogs, start };
}

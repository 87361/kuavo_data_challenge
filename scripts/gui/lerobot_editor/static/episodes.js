import { STORAGE, clamp, storedString } from "./utils.js";

export function createEpisodeController(ctx) {
  const { state, els, api, setStatus } = ctx;

  async function loadDatasets() {
    setStatus("Scanning");
    const payload = await api("/api/datasets");
    state.datasets = payload.datasets;
    els.datasetSelect.innerHTML = "";
    for (const item of state.datasets) {
      const option = document.createElement("option");
      option.value = item.path;
      option.textContent = `${item.version} ${item.editable ? "" : "(view only)"} ${item.path}`;
      option.disabled = !item.editable;
      els.datasetSelect.appendChild(option);
    }
    const firstEditable = state.datasets.find((item) => item.editable);
    if (firstEditable) els.datasetSelect.value = firstEditable.path;
    setStatus(`${state.datasets.length} datasets`);
  }

  function defaultExportPath(datasetPath) {
    const stamp = new Date().toISOString().replaceAll(":", "").replace(/\..+/, "").replace("T", "_");
    const name = datasetPath.split("/").filter(Boolean).slice(-3, -1).join("_") || "lerobot";
    return `/mnt/data/kuavo_tianchi/lerobot_edits/${name}_edited_${stamp}/lerobot`;
  }

  function renderEpisodeNav() {
    const total = state.dataset?.total_episodes || 0;
    const complete = ctx.progress?.isCurrentEpisodeComplete?.() ? " complete" : "";
    els.episodeJump.disabled = !state.dataset;
    els.prevEpisode.disabled = !state.dataset || state.episodeIndex <= 0;
    els.nextEpisode.disabled = !state.dataset || state.episodeIndex >= total - 1;
    els.episodeJump.max = String(Math.max(0, total - 1));
    els.episodeJump.value = String(state.episodeIndex || 0);
    els.episodeIndicator.textContent = total ? `Episode ${state.episodeIndex + 1} / ${total}${complete}` : "Episode - / -";
  }

  async function openDataset() {
    const path = els.datasetSelect.value;
    if (!path) return;
    ctx.video.stopPlayback();
    ctx.analysis.stopAnalysisPolling();
    setStatus("Loading dataset");
    state.dataset = await api("/api/open", { method: "POST", body: JSON.stringify({ path }) });
    state.progress = state.dataset.progress || null;
    state.analysisResult = null;
    state.history = [];
    state.future = [];
    els.exportPath.value = state.progress?.last_export_path || defaultExportPath(path);
    if (!els.urdfPath.value.trim()) {
      els.urdfPath.value = storedString(STORAGE.urdfPath, "") || state.dataset.urdf_path || "";
    }
    els.datasetName.textContent = state.dataset.path.split("/").slice(-3).join("/");
    ctx.progress.applyProgress(state.progress);
    ctx.analysis.renderAnalysisResult();
    await loadEpisode(0);
    setStatus("Ready");
  }

  async function loadEpisode(index) {
    if (!state.dataset) return;
    ctx.video.stopPlayback();
    const total = state.dataset.total_episodes;
    state.episodeIndex = clamp(Number(index) || 0, 0, Math.max(0, total - 1));
    state.episode = await api(`/api/episode/${state.episodeIndex}`);
    state.currentFrame = 0;
    state.viewStart = 0;
    state.selectedSegment = null;
    state.curveCache = ctx.curves.buildCurveCache();
    ctx.video.renderVideoGrid();
    ctx.updateControlValues();
    ctx.timeline.updateTimelineContentSize();
    ctx.renderAll();
    renderEpisodeNav();
    ctx.progress.renderProgress();
  }

  async function moveEpisode(delta) {
    if (!state.dataset) return;
    await loadEpisode(state.episodeIndex + delta);
  }

  async function jumpToEpisode() {
    if (!state.dataset) return;
    await loadEpisode(Number(els.episodeJump.value));
  }

  return { defaultExportPath, jumpToEpisode, loadDatasets, loadEpisode, moveEpisode, openDataset, renderEpisodeNav };
}

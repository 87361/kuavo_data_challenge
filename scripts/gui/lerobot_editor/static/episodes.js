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
      const workspace = item.active_workspace_path || "";
      const suffix = workspace ? `  -> editing ${workspace}` : "";
      const mode = item.editable ? "" : (item.browseable ? "(view only)" : "(unsupported)");
      option.textContent = `${item.version} ${mode} ${item.path}${suffix}`;
      option.disabled = !item.browseable;
      els.datasetSelect.appendChild(option);
    }
    const firstOpenable = state.datasets.find((item) => item.editable) || state.datasets.find((item) => item.browseable);
    if (firstOpenable) els.datasetSelect.value = firstOpenable.path;
    setStatus(`${state.datasets.length} datasets`);
  }

  function defaultExportPath(datasetPath) {
    const parts = datasetPath.split("/").filter(Boolean);
    const base = parts[parts.length - 1] === "lerobot" ? parts.slice(-3, -1) : parts.slice(-2);
    const name = (base.join("_") || "lerobot").replaceAll(/[^A-Za-z0-9_.-]/g, "_");
    return `/mnt/data/kuavo_tianchi/lerobot_edits/${name}/lerobot`;
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

  async function openDataset(options = {}) {
    const path = els.datasetSelect.value;
    if (!path) return;
    const targetEpisode = options.keepEpisode ? state.episodeIndex : 0;
    ctx.video.stopPlayback();
    setStatus("Loading dataset");
    state.showDeletedSegments = Boolean(els.showDeletedSegments?.checked);
    state.dataset = await api("/api/open", {
      method: "POST",
      body: JSON.stringify({
        path,
        show_deleted_segments: state.showDeletedSegments,
      }),
    });
    state.showDeletedSegments = Boolean(state.dataset.show_deleted_segments);
    if (els.showDeletedSegments) els.showDeletedSegments.checked = state.showDeletedSegments;
    state.progress = state.dataset.progress || null;
    state.history = [];
    state.future = [];
    els.exportPath.value = state.progress?.last_export_path || state.dataset.default_export_path || defaultExportPath(path);
    if (!els.urdfPath.value.trim()) {
      els.urdfPath.value = storedString(STORAGE.urdfPath, "") || state.dataset.urdf_path || "";
    }
    ctx.progress.applyProgress(state.progress);
    await loadEpisode(targetEpisode);
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

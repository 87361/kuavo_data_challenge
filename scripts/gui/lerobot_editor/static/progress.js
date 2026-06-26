import { clamp, formatSavedAt } from "./utils.js";

export function createProgressController(ctx) {
  const { state, els, api, setStatus } = ctx;

  function completedSet() {
    return new Set(state.progress?.completed_episodes || []);
  }

  function isCurrentEpisodeComplete() {
    return state.episode ? completedSet().has(state.episodeIndex) : false;
  }

  function renderProgress() {
    const progress = state.progress || {};
    const total = Number(progress.total_episodes || state.dataset?.total_episodes || 0);
    const completed = Number(progress.completed_count || completedSet().size || 0);
    const edited = Number(progress.edited_count ?? Object.keys(progress.edits || {}).length);
    const pending = Number(progress.pending_export_count ?? 0);
    const pct = total > 0 ? clamp((completed / total) * 100, 0, 100) : 0;

    els.annotationProgressBar.style.width = `${pct.toFixed(1)}%`;
    els.completedMetric.textContent = `${completed} / ${total}`;
    els.editedMetric.textContent = String(edited);
    els.pendingExportMetric.textContent = String(pending);
    els.savedMetric.textContent = state.progressDirty ? "Unsaved" : formatSavedAt(progress.saved_at);
    els.progressStatus.textContent = state.progressDirty ? "Unsaved changes" : `Saved ${formatSavedAt(progress.saved_at)}`;

    const complete = isCurrentEpisodeComplete();
    els.markComplete.textContent = complete ? "Unmark Complete" : "Mark Complete";
    els.markComplete.disabled = !state.episode;
    els.saveProgress.disabled = !state.dataset;
  }

  function applyProgress(payload) {
    state.progress = payload || null;
    state.progressDirty = false;
    renderProgress();
    ctx.episodes.renderEpisodeNav();
  }

  async function loadProgress() {
    if (!state.dataset) return;
    applyProgress(await api("/api/progress"));
  }

  async function saveProgress() {
    if (!state.dataset) return;
    if (state.progressSaveTimer) {
      clearTimeout(state.progressSaveTimer);
      state.progressSaveTimer = null;
    }
    const payload = await api("/api/progress/save", {
      method: "POST",
      body: JSON.stringify({}),
    });
    applyProgress(payload);
    setStatus("Progress saved");
  }

  function scheduleSave() {
    if (state.progressSaveTimer) clearTimeout(state.progressSaveTimer);
    state.progressSaveTimer = setTimeout(() => {
      state.progressSaveTimer = null;
      saveProgress().catch((error) => setStatus(error.message));
    }, 900);
  }

  function markDirty({ autosave = true } = {}) {
    state.progressDirty = true;
    renderProgress();
    if (autosave) scheduleSave();
  }

  function noteEdits(allEdits) {
    if (!state.progress) return;
    state.progress.edits = allEdits || {};
    state.progress.edited_count = Object.keys(state.progress.edits).length;
    state.progress.pending_export_count = state.progress.edited_count;
    markDirty();
  }

  async function setEpisodeCompleted(episodeIndex, completed) {
    const payload = await api(`/api/progress/episode/${episodeIndex}`, {
      method: "POST",
      body: JSON.stringify({ completed }),
    });
    state.progress = payload;
    markDirty();
  }

  async function toggleCurrentComplete() {
    if (!state.episode) return;
    await setEpisodeCompleted(state.episodeIndex, !isCurrentEpisodeComplete());
  }

  return {
    applyProgress,
    isCurrentEpisodeComplete,
    loadProgress,
    markDirty,
    noteEdits,
    renderProgress,
    saveProgress,
    toggleCurrentComplete,
  };
}

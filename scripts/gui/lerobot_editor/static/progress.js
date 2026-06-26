import { clamp, formatSavedAt } from "./utils.js";

export function createProgressController(ctx) {
  const { state, els, api, setStatus } = ctx;

  function defaultAnnotation() {
    return { completed: false, rating: null, notes: [], updated_at: null };
  }

  function annotationMap() {
    return state.progress?.episode_annotations || {};
  }

  function annotationFor(index = state.episodeIndex) {
    return annotationMap()[String(index)]
      || (state.episode?.episode_index === index ? state.episode.annotation : null)
      || defaultAnnotation();
  }

  function completedSet() {
    const completed = new Set(state.progress?.completed_episodes || []);
    for (const [key, annotation] of Object.entries(annotationMap())) {
      if (annotation?.completed) completed.add(Number(key));
    }
    return completed;
  }

  function isCurrentEpisodeComplete() {
    return state.episode ? completedSet().has(state.episodeIndex) : false;
  }

  function renderChipList(container, values, { empty = "-", asButtons = false } = {}) {
    container.innerHTML = "";
    if (!values.length) {
      const chip = document.createElement("span");
      chip.className = "note-chip empty";
      chip.textContent = empty;
      container.appendChild(chip);
      return;
    }
    for (const value of values) {
      const el = document.createElement(asButtons ? "button" : "span");
      el.className = "note-chip";
      el.textContent = value;
      if (asButtons) el.dataset.note = value;
      container.appendChild(el);
    }
  }

  function renderAnnotation() {
    const annotation = annotationFor();
    const rating = annotation.rating ?? null;
    els.ratingValue.textContent = rating ? `${rating} / 10` : "-";
    for (const button of els.ratingButtons.querySelectorAll("button[data-rating]")) {
      button.classList.toggle("active", Number(button.dataset.rating) === rating);
      button.disabled = !state.episode;
    }
    els.noteInput.disabled = !state.episode;
    els.addNote.disabled = !state.episode;
    renderChipList(els.currentNotes, annotation.notes || [], { empty: "No notes" });
    renderChipList(els.noteLabels, state.progress?.note_labels || [], {
      empty: "No saved tags",
      asButtons: true,
    });
  }

  function renderTrajectoryPreview() {
    const job = state.trajectoryJob;
    if (job?.params?.episode_index === state.episodeIndex && job.status === "running") {
      els.trajectoryVideo.removeAttribute("src");
      els.trajectoryVideo.load();
      const pct = Math.round((Number(job.progress) || 0) * 100);
      els.trajectoryProgressBar.style.width = `${pct}%`;
      els.trajectoryStatus.textContent = `${pct}% ${job.message || "Generating preview"}`;
      return;
    }
    if (job?.params?.episode_index === state.episodeIndex && job.status === "failed") {
      els.trajectoryProgressBar.style.width = "100%";
      els.trajectoryVideo.removeAttribute("src");
      els.trajectoryVideo.load();
      els.trajectoryStatus.textContent = job.error || job.message || "Trajectory preview failed";
      return;
    }
    if (!state.episode) {
      els.trajectoryVideo.removeAttribute("src");
      els.trajectoryVideo.load();
      els.trajectoryProgressBar.style.width = "0%";
      els.trajectoryStatus.textContent = "No episode loaded";
      return;
    }
    const preview = state.trajectoryPreview;
    if (preview?.episodeIndex === state.episodeIndex && preview.url) {
      els.trajectoryVideo.src = preview.url;
      els.trajectoryProgressBar.style.width = "100%";
      els.trajectoryStatus.textContent = preview.cached ? "Cached preview" : "Preview ready";
      return;
    }
    els.trajectoryVideo.removeAttribute("src");
    els.trajectoryVideo.load();
    els.trajectoryProgressBar.style.width = "0%";
    els.trajectoryStatus.textContent = isCurrentEpisodeComplete()
      ? "Preview not generated yet"
      : "Mark an episode complete to generate a preview";
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
    renderAnnotation();
    renderTrajectoryPreview();
  }

  function applyProgress(payload) {
    state.progress = payload || null;
    state.progressDirty = false;
    if (state.episode) {
      state.episode.annotation = annotationFor(state.episodeIndex);
    }
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
    ctx.episodes.renderEpisodeNav();
    if (autosave) scheduleSave();
  }

  function noteEdits(allEdits) {
    if (!state.progress) return;
    state.progress.edits = allEdits || {};
    state.progress.edited_count = Object.keys(state.progress.edits).length;
    state.progress.pending_export_count = state.progress.edited_count;
    markDirty();
  }

  function currentFullAnnotation(patch = {}) {
    const current = annotationFor();
    return {
      completed: Boolean(current.completed),
      rating: current.rating ?? null,
      notes: [...(current.notes || [])],
      ...patch,
    };
  }

  async function updateCurrentAnnotation(patch, { autosave = true, preview = false } = {}) {
    if (!state.episode) return;
    const body = currentFullAnnotation(patch);
    const payload = await api(`/api/annotations/episode/${state.episodeIndex}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.progress = payload;
    state.episode.annotation = annotationFor(state.episodeIndex);
    markDirty({ autosave });
    if (preview && body.completed) {
      generateTrajectoryPreview().catch((error) => {
        els.trajectoryStatus.textContent = error.message;
      });
    }
  }

  async function setCurrentRating(rating) {
    if (!state.episode) return;
    await updateCurrentAnnotation({ rating: clamp(Math.round(rating), 1, 10) });
    setStatus(`Score ${rating}/10`);
  }

  async function submitNote() {
    const text = els.noteInput.value.trim();
    if (!text || !state.episode) return;
    await appendNote(text);
    els.noteInput.value = "";
  }

  async function appendNote(text) {
    if (!state.episode) return;
    const notes = annotationFor().notes || [];
    if (notes.includes(text)) return;
    await updateCurrentAnnotation({ notes: [...notes, text] });
    setStatus("Note saved");
  }

  async function setEpisodeCompleted(episodeIndex, completed) {
    if (!state.episode || episodeIndex !== state.episodeIndex) return;
    await updateCurrentAnnotation({ completed }, { preview: completed });
  }

  async function toggleCurrentComplete() {
    if (!state.episode) return;
    await setEpisodeCompleted(state.episodeIndex, !isCurrentEpisodeComplete());
  }

  async function completeCurrentAndSave() {
    if (!state.episode) return;
    if (!isCurrentEpisodeComplete()) {
      await setEpisodeCompleted(state.episodeIndex, true);
    }
    await saveProgress();
  }

  async function generateTrajectoryPreview() {
    if (!state.episode) return;
    stopTrajectoryPolling();
    state.trajectoryJob = {
      status: "running",
      message: "Queued",
      progress: 0,
      params: { episode_index: state.episodeIndex },
    };
    renderTrajectoryPreview();
    const videoKey = state.episode.video_keys.includes("observation.images.head_cam_h")
      ? "observation.images.head_cam_h"
      : state.episode.video_keys[0];
    const job = await api(`/api/trajectory/episode/${state.episodeIndex}`, {
      method: "POST",
      body: JSON.stringify({
        urdf_path: els.urdfPath.value.trim() || null,
        video_key: videoKey,
        source: "state",
        hand: "auto",
      }),
    });
    applyTrajectoryJob(job);
    if (job.status === "running") {
      state.trajectoryPoll = setInterval(() => {
        pollTrajectoryStatus().catch((error) => {
          stopTrajectoryPolling();
          state.trajectoryJob = { status: "failed", error: error.message, params: { episode_index: state.episodeIndex } };
          renderTrajectoryPreview();
        });
      }, 700);
      await pollTrajectoryStatus();
    }
  }

  function stopTrajectoryPolling() {
    if (state.trajectoryPoll) {
      clearInterval(state.trajectoryPoll);
      state.trajectoryPoll = null;
    }
  }

  function applyTrajectoryJob(job) {
    state.trajectoryJob = job || null;
    if (!job) {
      renderTrajectoryPreview();
      return;
    }
    if (job.status === "complete" && job.url) {
      state.trajectoryPreview = {
        episodeIndex: Number(job.params?.episode_index ?? state.episodeIndex),
        url: job.url,
        cached: Boolean(job.cached),
      };
      stopTrajectoryPolling();
    } else if (job.status === "failed") {
      stopTrajectoryPolling();
    }
    renderTrajectoryPreview();
  }

  async function pollTrajectoryStatus() {
    const job = await api("/api/trajectory/status");
    applyTrajectoryJob(job);
  }

  return {
    applyProgress,
    appendNote,
    completeCurrentAndSave,
    generateTrajectoryPreview,
    isCurrentEpisodeComplete,
    loadProgress,
    markDirty,
    noteEdits,
    renderProgress,
    saveProgress,
    setCurrentRating,
    submitNote,
    toggleCurrentComplete,
  };
}

import { STORAGE, clamp, formatSavedAt } from "./utils.js";

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

  function renderSummaryChips(container, items, emptyText, renderText) {
    container.innerHTML = "";
    if (!items.length) {
      const chip = document.createElement("span");
      chip.className = "summary-chip empty";
      chip.textContent = emptyText;
      container.appendChild(chip);
      return;
    }
    for (const item of items) {
      const chip = document.createElement("span");
      chip.className = "summary-chip";
      chip.textContent = renderText(item);
      container.appendChild(chip);
    }
  }

  function renderAnnotationStats() {
    const stats = state.progress?.annotation_stats || {};
    const total = Number(stats.total_episodes || state.dataset?.total_episodes || 0);
    const completed = Number(stats.completed_count || 0);
    const rated = Number(stats.rated_count || 0);
    const unlabeled = Number(stats.unlabeled_count || 0);
    const unrated = Number(stats.unrated_count || 0);
    els.annotationSummaryMain.textContent = `${completed} / ${total} complete · ${rated} scored`;
    els.annotationSummaryMain.title = `Unrated completed: ${unrated}; completed without labels: ${unlabeled}`;

    const ratings = Array.isArray(stats.ratings) ? stats.ratings : [];
    const labels = Array.isArray(stats.labels) ? stats.labels : [];
    const ratingItems = unrated > 0 ? [...ratings, { rating: "No score", count: unrated }] : ratings;
    const labelItems = unlabeled > 0 ? [...labels, { label: "No label", count: unlabeled }] : labels;

    renderSummaryChips(
      els.annotationScoreStats,
      ratingItems,
      "No scores",
      (item) => `${Number.isFinite(Number(item.rating)) ? `S${item.rating}` : item.rating}: ${item.count}`,
    );
    renderSummaryChips(els.annotationLabelStats, labelItems, "No labels", (item) => `${item.label}: ${item.count}`);
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

  function clearTrajectoryVideo() {
    if (!els.trajectoryVideo.hasAttribute("src")) return;
    els.trajectoryVideo.removeAttribute("src");
    els.trajectoryVideo.load();
  }

  function setTrajectoryVideo(url) {
    if (els.trajectoryVideo.getAttribute("src") === url) return;
    els.trajectoryVideo.src = url;
  }

  function trajectoryJobEpisode(job) {
    const index = Number(job?.params?.episode_index);
    return Number.isFinite(index) ? index : null;
  }

  function trajectoryJobResult(job) {
    return job?.result || (job?.url ? job : null);
  }

  function renderTrajectoryPreview() {
    const job = state.trajectoryJob;
    const jobEpisode = trajectoryJobEpisode(job);
    const running = job?.status === "running";
    const preview = state.trajectoryPreview?.episodeIndex === state.episodeIndex
      ? state.trajectoryPreview
      : null;

    els.generateTrajectory.disabled = !state.episode || !(state.episode.video_keys || []).length || running;
    els.generateTrajectory.textContent = running ? "Generating..." : "Generate";

    if (running) {
      if (jobEpisode === state.episodeIndex || !preview?.url) clearTrajectoryVideo();
      if (preview?.url && jobEpisode !== state.episodeIndex) setTrajectoryVideo(preview.url);
      const pct = Math.round((Number(job.progress) || 0) * 100);
      els.trajectoryProgressBar.style.width = `${pct}%`;
      const label = jobEpisode === state.episodeIndex || jobEpisode === null
        ? ""
        : ` for episode ${jobEpisode + 1}`;
      els.trajectoryStatus.textContent = `${pct}% ${job.message || "Generating preview"}${label}`;
      return;
    }
    if (jobEpisode === state.episodeIndex && job?.status === "failed") {
      els.trajectoryProgressBar.style.width = "100%";
      clearTrajectoryVideo();
      els.trajectoryStatus.textContent = job.error || job.message || "Trajectory preview failed";
      return;
    }
    if (!state.episode) {
      clearTrajectoryVideo();
      els.trajectoryProgressBar.style.width = "0%";
      els.trajectoryStatus.textContent = "No episode loaded";
      return;
    }
    if (!(state.episode.video_keys || []).length) {
      clearTrajectoryVideo();
      els.trajectoryProgressBar.style.width = "0%";
      els.trajectoryStatus.textContent = "No camera video available for this episode";
      return;
    }
    if (preview?.url) {
      setTrajectoryVideo(preview.url);
      els.trajectoryProgressBar.style.width = "100%";
      els.trajectoryStatus.textContent = preview.cached ? "Cached preview" : "Preview ready";
      return;
    }
    clearTrajectoryVideo();
    els.trajectoryProgressBar.style.width = "0%";
    els.trajectoryStatus.textContent = "Click Generate to render a preview";
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
    renderAnnotationStats();

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

  async function updateCurrentAnnotation(patch, { autosave = true } = {}) {
    if (!state.episode) return;
    const body = currentFullAnnotation(patch);
    const payload = await api(`/api/annotations/episode/${state.episodeIndex}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.progress = payload;
    state.episode.annotation = annotationFor(state.episodeIndex);
    markDirty({ autosave });
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
    await updateCurrentAnnotation({ completed });
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
    if (state.trajectoryJob?.status === "running") return;
    if (!(state.episode.video_keys || []).length) {
      throw new Error("No camera video available for trajectory preview");
    }
    const episodeIndex = state.episodeIndex;
    stopTrajectoryPolling();
    if (state.trajectoryPreview?.episodeIndex === episodeIndex) state.trajectoryPreview = null;
    state.trajectoryJob = {
      status: "running",
      message: "Queued",
      progress: 0,
      params: { episode_index: episodeIndex },
    };
    renderTrajectoryPreview();
    const videoKey = state.episode.video_keys.includes("observation.images.head_cam_h")
      ? "observation.images.head_cam_h"
      : state.episode.video_keys[0];
    const urdfPath = els.urdfPath.value.trim();
    localStorage.setItem(STORAGE.urdfPath, urdfPath);
    try {
      const job = await api(`/api/trajectory/episode/${episodeIndex}`, {
        method: "POST",
        body: JSON.stringify({
          urdf_path: urdfPath || null,
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
            state.trajectoryJob = {
              status: "failed",
              message: "trajectory preview failed",
              progress: 1,
              error: error.message,
              params: { episode_index: episodeIndex },
            };
            renderTrajectoryPreview();
          });
        }, 700);
        await pollTrajectoryStatus();
      }
    } catch (error) {
      stopTrajectoryPolling();
      state.trajectoryJob = {
        status: "failed",
        message: "trajectory preview failed",
        progress: 1,
        error: error.message,
        params: { episode_index: episodeIndex },
      };
      renderTrajectoryPreview();
      throw error;
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
    if (!els.urdfPath.value.trim() && job.params?.urdf_path) {
      els.urdfPath.value = job.params.urdf_path;
      localStorage.setItem(STORAGE.urdfPath, job.params.urdf_path);
    }
    const result = trajectoryJobResult(job);
    if (job.status === "complete" && result?.url) {
      state.trajectoryPreview = {
        episodeIndex: Number(job.params?.episode_index ?? state.episodeIndex),
        url: result.url,
        cached: Boolean(result.cached),
      };
      stopTrajectoryPolling();
    } else if (job.status === "complete" || job.status === "failed") {
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

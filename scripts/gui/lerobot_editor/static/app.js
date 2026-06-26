import { api } from "./api.js";
import { createAnalysisController } from "./analysis.js";
import { createCurveController } from "./curves.js";
import { collectElements, createStatus } from "./dom.js";
import { createEditController } from "./edits.js";
import { createEpisodeController } from "./episodes.js";
import { createExportController } from "./export.js";
import { createFrameController } from "./frames.js";
import { createLayoutController } from "./layout.js";
import { createMetricsController } from "./metrics.js";
import { createProgressController } from "./progress.js";
import { state } from "./state.js";
import { STORAGE, clamp } from "./utils.js";
import { createVideoController } from "./video.js";
import { createTimelineController } from "./timeline.js";

const els = collectElements();
const ctx = {
  api,
  els,
  state,
  setStatus: createStatus(els),
};

ctx.frames = createFrameController(ctx);
ctx.metrics = createMetricsController(ctx);
ctx.video = createVideoController(ctx);
ctx.timeline = createTimelineController(ctx);
ctx.curves = createCurveController(ctx);
ctx.analysis = createAnalysisController(ctx);
ctx.progress = createProgressController(ctx);
ctx.episodes = createEpisodeController(ctx);
ctx.edits = createEditController(ctx);
ctx.exporter = createExportController(ctx);
ctx.layout = createLayoutController(ctx);

ctx.updateControlValues = () => ctx.layout.updateControlValues();
ctx.renderAll = () => {
  ctx.video.syncVideosToFrame(state.currentFrame, true);
  ctx.video.renderFrameFallbacks();
  ctx.metrics.renderMetrics();
  ctx.timeline.renderTimeline();
  ctx.curves.renderCurves();
};

function bindEvents() {
  els.loadDataset.addEventListener("click", () => ctx.episodes.openDataset().catch((err) => ctx.setStatus(err.message)));
  els.prevEpisode.addEventListener("click", () => ctx.episodes.moveEpisode(-1).catch((err) => ctx.setStatus(err.message)));
  els.nextEpisode.addEventListener("click", () => ctx.episodes.moveEpisode(1).catch((err) => ctx.setStatus(err.message)));
  els.episodeJump.addEventListener("change", () => ctx.episodes.jumpToEpisode().catch((err) => ctx.setStatus(err.message)));
  els.episodeJump.addEventListener("keydown", (event) => {
    if (event.key === "Enter") ctx.episodes.jumpToEpisode().catch((err) => ctx.setStatus(err.message));
  });
  els.prevFrame.addEventListener("click", () => ctx.frames.moveFrame(-1));
  els.nextFrame.addEventListener("click", () => ctx.frames.moveFrame(1));
  els.playPause.addEventListener("click", ctx.video.playPause);
  els.cutFrame.addEventListener("click", () => ctx.edits.cutAtFrame().catch((err) => ctx.setStatus(err.message)));
  els.deleteSegment.addEventListener("click", () => ctx.edits.toggleDeleteSegment().catch((err) => ctx.setStatus(err.message)));
  els.undoEdit.addEventListener("click", () => ctx.edits.undo().catch((err) => ctx.setStatus(err.message)));
  els.redoEdit.addEventListener("click", () => ctx.edits.redo().catch((err) => ctx.setStatus(err.message)));
  els.markComplete.addEventListener("click", () => ctx.progress.toggleCurrentComplete().catch((err) => ctx.setStatus(err.message)));
  els.saveProgress.addEventListener("click", () => ctx.progress.saveProgress().catch((err) => ctx.setStatus(err.message)));
  els.exportDataset.addEventListener("click", () => ctx.exporter.exportDataset().catch((err) => {
    els.exportDataset.disabled = false;
    ctx.setStatus(err.message);
  }));
  els.runAnalysis.addEventListener("click", () => ctx.analysis.runCoverageAnalysis().catch((err) => {
    ctx.analysis.stopAnalysisPolling();
    els.analysisStatus.textContent = err.message;
  }));

  els.timeline.addEventListener("click", ctx.timeline.handleTimelineClick);
  els.timelineScroll.addEventListener("scroll", () => {
    ctx.timeline.syncHorizontalScroll(els.timelineScroll);
    ctx.timeline.renderTimeline();
    ctx.curves.renderCurves();
  });
  els.curveScroll.addEventListener("scroll", () => {
    ctx.timeline.syncHorizontalScroll(els.curveScroll);
    ctx.timeline.renderTimeline();
    ctx.curves.renderCurves();
  });
  els.zoomSlider.addEventListener("input", () => ctx.layout.setZoom(els.zoomSlider.value));
  els.zoomOut.addEventListener("click", () => ctx.layout.setZoom(state.zoom / 1.35));
  els.zoomIn.addEventListener("click", () => ctx.layout.setZoom(state.zoom * 1.35));
  els.zoomReset.addEventListener("click", () => ctx.layout.setZoom(1));
  els.curveGroup.addEventListener("change", () => {
    state.curveGroup = els.curveGroup.value;
    localStorage.setItem(STORAGE.curveGroup, state.curveGroup);
    ctx.curves.renderCurves();
  });
  els.curveScale.addEventListener("input", () => {
    state.curveScale = clamp(Number(els.curveScale.value) || 1, 0.5, 4);
    localStorage.setItem(STORAGE.curveScale, String(state.curveScale));
    ctx.curves.renderCurves();
  });

  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      ctx.frames.moveFrame(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      ctx.frames.moveFrame(1);
    } else if (event.key === "PageUp") {
      event.preventDefault();
      ctx.episodes.moveEpisode(-1).catch((err) => ctx.setStatus(err.message));
    } else if (event.key === "PageDown") {
      event.preventDefault();
      ctx.episodes.moveEpisode(1).catch((err) => ctx.setStatus(err.message));
    } else if (event.key === " ") {
      event.preventDefault();
      ctx.video.playPause();
    }
  });

  window.addEventListener("resize", () => {
    ctx.timeline.updateTimelineContentSize();
    ctx.timeline.ensureFrameInView(state.currentFrame);
    ctx.timeline.renderTimeline();
    ctx.curves.renderCurves();
  });
}

function init() {
  bindEvents();
  ctx.layout.initResizeHandle();
  ctx.analysis.initAnalysisControls();
  ctx.layout.updateControlValues();
  ctx.progress.renderProgress();
  ctx.episodes.renderEpisodeNav();
  ctx.episodes.loadDatasets().catch((err) => ctx.setStatus(err.message));
}

init();

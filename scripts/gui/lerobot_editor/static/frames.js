import { clamp } from "./utils.js";

export function createFrameController(ctx) {
  const { state, els } = ctx;

  function fps() {
    return Number(state.episode?.fps || state.dataset?.fps || 10) || 10;
  }

  function frameToTime(frame) {
    return Math.max(0, Number(frame) / fps());
  }

  function timeToFrame(time) {
    if (!state.episode) return 0;
    return clamp(Math.round(Number(time) * fps()), 0, state.episode.length - 1);
  }

  function setCurrentFrame(frame, options = {}) {
    if (!state.episode) return;
    const nextFrame = clamp(Math.round(frame), 0, state.episode.length - 1);
    const oldScrollLeft = els.curveScroll.scrollLeft;
    state.currentFrame = nextFrame;
    const viewportChanged = ctx.timeline.ensureFrameInView(nextFrame) || oldScrollLeft !== els.curveScroll.scrollLeft;
    if (options.seek !== false) ctx.video.syncVideosToFrame(nextFrame, Boolean(options.forceSeek));
    ctx.video.renderFrameFallbacks();
    ctx.metrics.renderMetrics();
    if (viewportChanged || options.forceTimeline) ctx.timeline.renderTimeline();
    else ctx.timeline.renderTimelinePlayhead();
    ctx.curves.renderCurves();
  }

  function moveFrame(delta) {
    if (!state.episode) return;
    if (state.playing) ctx.video.stopPlayback();
    setCurrentFrame(state.currentFrame + delta, { seek: true, forceTimeline: false });
  }

  return { fps, frameToTime, timeToFrame, setCurrentFrame, moveFrame };
}

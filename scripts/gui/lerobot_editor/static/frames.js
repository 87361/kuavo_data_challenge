import { clamp, getSegments } from "./utils.js";

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

  function visibleSegments() {
    if (!state.episode) return [];
    const segments = getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments);
    return state.showDeletedSegments ? segments : segments.filter((seg) => !seg.deleted);
  }

  function isVisibleFrame(frame) {
    if (!state.episode) return false;
    if (state.showDeletedSegments || !(state.episode.deleted_segments || []).length) return true;
    return visibleSegments().some((seg) => frame >= seg.start && frame < seg.end);
  }

  function nearestVisibleFrame(frame, direction = 1) {
    if (!state.episode) return 0;
    const clamped = clamp(Math.round(frame), 0, state.episode.length - 1);
    if (isVisibleFrame(clamped)) return clamped;
    const kept = visibleSegments();
    if (!kept.length) return clamped;
    if (direction < 0) {
      const previous = [...kept].reverse().find((seg) => seg.end - 1 <= clamped || seg.start < clamped);
      return previous ? clamp(previous.end - 1, 0, state.episode.length - 1) : kept[0].start;
    }
    const next = kept.find((seg) => seg.start >= clamped || seg.end > clamped);
    return next ? next.start : kept[kept.length - 1].end - 1;
  }

  function nextVisibleFrame(frame, delta) {
    if (!state.episode) return 0;
    const direction = delta < 0 ? -1 : 1;
    let next = clamp(Math.round(frame) + direction, 0, state.episode.length - 1);
    if (state.showDeletedSegments || !(state.episode.deleted_segments || []).length) return next;
    while (next >= 0 && next < state.episode.length && !isVisibleFrame(next)) {
      next += direction;
    }
    if (next < 0 || next >= state.episode.length) {
      return nearestVisibleFrame(frame, direction);
    }
    return next;
  }

  function setCurrentFrame(frame, options = {}) {
    if (!state.episode) return;
    const rawFrame = clamp(Math.round(frame), 0, state.episode.length - 1);
    const nextFrame = options.allowDeleted ? rawFrame : nearestVisibleFrame(rawFrame, options.direction || 1);
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
    setCurrentFrame(nextVisibleFrame(state.currentFrame, delta), {
      seek: true,
      forceTimeline: false,
      direction: delta,
    });
  }

  return { fps, frameToTime, isVisibleFrame, nearestVisibleFrame, nextVisibleFrame, timeToFrame, setCurrentFrame, moveFrame };
}

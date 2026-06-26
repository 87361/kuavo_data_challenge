import { getSegments } from "./utils.js";

export function createMetricsController(ctx) {
  const { state, els } = ctx;

  function currentSegment() {
    if (!state.episode) return null;
    return getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments).find(
      (seg) => state.currentFrame >= seg.start && state.currentFrame < seg.end,
    );
  }

  function renderMetrics() {
    if (!state.episode) return;
    const seg = currentSegment();
    els.episodeMetric.textContent = `${state.episodeIndex} / ${state.dataset.total_episodes - 1}`;
    els.frameMetric.textContent = `${state.currentFrame} / ${state.episode.length - 1}`;
    els.timeMetric.textContent = `${(state.currentFrame / ctx.frames.fps()).toFixed(2)}s`;
    els.segmentMetric.textContent = seg ? `${seg.index}  ${seg.start}-${seg.end - 1}` : "-";
  }

  return { currentSegment, renderMetrics };
}

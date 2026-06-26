import { getSegments } from "./utils.js";

export function createEditController(ctx) {
  const { state, api } = ctx;

  function snapshot() {
    if (!state.episode) return null;
    return {
      cuts: [...state.episode.cuts],
      deleted_segments: [...state.episode.deleted_segments],
    };
  }

  function pushHistory() {
    const snap = snapshot();
    if (!snap) return;
    state.history.push(snap);
    if (state.history.length > 100) state.history.shift();
    state.future = [];
  }

  function deletedRangesFromEpisode() {
    return getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments)
      .filter((seg) => seg.deleted)
      .map((seg) => [seg.start, seg.end]);
  }

  function deletedSegmentsForRanges(cuts, ranges) {
    return getSegments(state.episode.length, cuts, []).flatMap((seg) => {
      const isDeleted = ranges.some(([start, end]) => seg.start >= start && seg.end <= end);
      return isDeleted ? [seg.index] : [];
    });
  }

  async function syncCuts() {
    const payload = await api("/api/cuts", {
      method: "POST",
      body: JSON.stringify({
        episode_index: state.episodeIndex,
        cuts: state.episode.cuts,
        deleted_segments: state.episode.deleted_segments,
      }),
    });
    state.episode.cuts = payload.cuts;
    state.episode.deleted_segments = payload.deleted_segments;
    state.episode.segments = payload.segments;
    if (!state.showDeletedSegments) {
      state.currentFrame = ctx.frames.nearestVisibleFrame(state.currentFrame);
    }
    state.selectedSegment = ctx.metrics.currentSegment()?.index ?? null;
    ctx.renderAll();
    ctx.progress.noteEdits(payload.all_edits);
  }

  async function cutAtFrame() {
    if (!state.episode || state.currentFrame <= 0 || state.currentFrame >= state.episode.length) return;
    if (state.playing) ctx.video.stopPlayback();
    pushHistory();
    const oldDeletedRanges = deletedRangesFromEpisode();
    const cuts = Array.from(new Set([...state.episode.cuts, state.currentFrame])).sort((a, b) => a - b);
    state.episode.cuts = cuts;
    state.episode.deleted_segments = deletedSegmentsForRanges(cuts, oldDeletedRanges);
    await syncCuts();
  }

  async function toggleDeleteSegment() {
    if (!state.episode) return;
    if (state.playing) ctx.video.stopPlayback();
    const seg = state.selectedSegment ?? ctx.metrics.currentSegment()?.index;
    if (seg === null || seg === undefined) return;
    pushHistory();
    const deleted = new Set(state.episode.deleted_segments);
    if (deleted.has(seg)) deleted.delete(seg);
    else deleted.add(seg);
    state.episode.deleted_segments = [...deleted].sort((a, b) => a - b);
    await syncCuts();
  }

  async function applySnapshot(snap) {
    if (!state.episode || !snap) return;
    if (state.playing) ctx.video.stopPlayback();
    state.episode.cuts = [...snap.cuts];
    state.episode.deleted_segments = [...snap.deleted_segments];
    await syncCuts();
  }

  async function undo() {
    if (!state.history.length) return;
    const current = snapshot();
    const previous = state.history.pop();
    if (current) state.future.push(current);
    await applySnapshot(previous);
  }

  async function redo() {
    if (!state.future.length) return;
    const current = snapshot();
    const next = state.future.pop();
    if (current) state.history.push(current);
    await applySnapshot(next);
  }

  return { cutAtFrame, redo, toggleDeleteSegment, undo };
}

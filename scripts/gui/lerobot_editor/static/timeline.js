import { clamp, getSegments } from "./utils.js";

export function createTimelineController(ctx) {
  const { state, els } = ctx;

  function timelineViewportWidth() {
    return Math.max(320, Math.floor(els.curveScroll?.clientWidth || els.timelineScroll?.clientWidth || 320));
  }

  function updateTimelineContentSize() {
    if (!state.episode) return;
    const viewport = timelineViewportWidth();
    const nextWidth = Math.max(viewport, Math.round(viewport * state.zoom));
    state.timelineContentWidth = nextWidth;
    els.timeline.style.width = `${nextWidth}px`;
    els.curveCanvas.style.width = `${nextWidth}px`;
  }

  function pixelsPerFrame() {
    if (!state.episode) return 1;
    return state.timelineContentWidth / Math.max(1, state.episode.length - 1);
  }

  function visibleRange() {
    if (!state.episode) return { start: 0, end: 1, span: 1 };
    updateTimelineContentSize();
    const scrollLeft = els.curveScroll?.scrollLeft || els.timelineScroll?.scrollLeft || 0;
    const width = timelineViewportWidth();
    const ppf = pixelsPerFrame();
    const start = clamp(Math.floor(scrollLeft / ppf) - 1, 0, Math.max(0, state.episode.length - 1));
    const end = clamp(Math.ceil((scrollLeft + width) / ppf) + 2, start + 1, state.episode.length);
    return { start, end, span: Math.max(1, end - start), scrollLeft, width };
  }

  function frameToX(frame) {
    if (!state.episode) return 0;
    return (clamp(frame, 0, state.episode.length - 1) / Math.max(1, state.episode.length - 1)) * state.timelineContentWidth;
  }

  function boundaryToX(frame) {
    if (!state.episode) return 0;
    return (clamp(frame, 0, state.episode.length) / Math.max(1, state.episode.length)) * state.timelineContentWidth;
  }

  function syncHorizontalScroll(source) {
    if (!source || state.scrollSync) return;
    const target = source === els.timelineScroll ? els.curveScroll : els.timelineScroll;
    if (!target) return;
    state.scrollSync = true;
    target.scrollLeft = source.scrollLeft;
    state.scrollSync = false;
  }

  function ensureFrameInView(frame) {
    if (!state.episode) return false;
    updateTimelineContentSize();
    const scroller = els.curveScroll;
    const before = scroller.scrollLeft;
    const x = frameToX(frame);
    const margin = Math.max(40, scroller.clientWidth * 0.12);
    if (x < scroller.scrollLeft + margin) {
      scroller.scrollLeft = clamp(x - scroller.clientWidth * 0.35, 0, Math.max(0, state.timelineContentWidth - scroller.clientWidth));
    } else if (x > scroller.scrollLeft + scroller.clientWidth - margin) {
      scroller.scrollLeft = clamp(x - scroller.clientWidth * 0.65, 0, Math.max(0, state.timelineContentWidth - scroller.clientWidth));
    }
    syncHorizontalScroll(els.curveScroll);
    return before !== scroller.scrollLeft;
  }

  function niceTickStep(span) {
    const raw = Math.max(1, span / 6);
    const pow = 10 ** Math.floor(Math.log10(raw));
    for (const mul of [1, 2, 5, 10]) {
      const step = mul * pow;
      if (raw <= step) return Math.max(1, Math.round(step));
    }
    return Math.max(1, Math.round(10 * pow));
  }

  function renderTimeline() {
    if (!state.episode) return;
    updateTimelineContentSize();
    els.timeline.innerHTML = "";
    const { start, end } = visibleRange();
    const step = niceTickStep(end - start);

    for (let tick = Math.ceil(start / step) * step; tick < end; tick += step) {
      const node = document.createElement("div");
      node.className = "timeline-tick";
      node.style.left = `${frameToX(tick)}px`;
      const label = document.createElement("span");
      label.textContent = String(tick);
      node.appendChild(label);
      els.timeline.appendChild(node);
    }

    const segments = getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments);
    for (const seg of segments) {
      const segStart = Math.max(seg.start, start);
      const segEnd = Math.min(seg.end, end);
      if (segStart >= segEnd) continue;
      const node = document.createElement("div");
      node.className = `segment${seg.deleted ? " deleted" : ""}${state.selectedSegment === seg.index ? " selected" : ""}`;
      node.style.left = `${boundaryToX(segStart)}px`;
      node.style.width = `${Math.max(5, boundaryToX(segEnd) - boundaryToX(segStart))}px`;
      node.dataset.index = seg.index;
      const label = document.createElement("span");
      label.className = "segment-label";
      label.textContent = `${seg.start}-${seg.end - 1}`;
      node.appendChild(label);
      els.timeline.appendChild(node);
    }

    const playhead = document.createElement("div");
    playhead.className = "playhead";
    els.timeline.appendChild(playhead);
    renderTimelinePlayhead();
  }

  function renderTimelinePlayhead() {
    const playhead = els.timeline.querySelector(".playhead");
    if (!playhead || !state.episode) return;
    playhead.style.display = "block";
    playhead.style.left = `${frameToX(state.currentFrame)}px`;
  }

  function frameFromTimelineEvent(event) {
    const rect = els.timeline.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, state.timelineContentWidth);
    const pct = x / Math.max(1, state.timelineContentWidth);
    return clamp(Math.round(pct * Math.max(0, state.episode.length - 1)), 0, state.episode.length - 1);
  }

  function handleTimelineClick(event) {
    if (!state.episode) return;
    if (state.playing) ctx.video.stopPlayback();
    const frame = frameFromTimelineEvent(event);
    const seg = getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments).find(
      (item) => frame >= item.start && frame < item.end,
    );
    state.selectedSegment = seg?.index ?? null;
    ctx.frames.setCurrentFrame(frame, { seek: true, forceTimeline: true });
  }

  return {
    boundaryToX,
    ensureFrameInView,
    frameToX,
    handleTimelineClick,
    renderTimeline,
    renderTimelinePlayhead,
    syncHorizontalScroll,
    timelineViewportWidth,
    updateTimelineContentSize,
    visibleRange,
  };
}

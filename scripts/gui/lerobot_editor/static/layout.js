import { STORAGE, clamp, storedNumber } from "./utils.js";

export function createLayoutController(ctx) {
  const { state, els } = ctx;

  function updateControlValues() {
    els.zoomSlider.value = String(state.zoom);
    els.curveScale.value = String(state.curveScale);
    if (els.playbackRate) els.playbackRate.value = String(state.playbackRate);
    if ([...els.curveGroup.options].some((option) => option.value === state.curveGroup)) {
      els.curveGroup.value = state.curveGroup;
    } else {
      els.curveGroup.value = "left";
      state.curveGroup = "left";
    }
  }

  function setZoom(value) {
    if (!state.episode) return;
    const beforeX = ctx.timeline.frameToX(state.currentFrame);
    const nextZoom = clamp(Number(value) || 1, 1, 24);
    state.zoom = nextZoom;
    localStorage.setItem(STORAGE.zoom, String(nextZoom));
    ctx.timeline.updateTimelineContentSize();
    const afterX = ctx.timeline.frameToX(state.currentFrame);
    const delta = afterX - beforeX;
    const maxScroll = Math.max(0, state.timelineContentWidth - els.curveScroll.clientWidth);
    els.curveScroll.scrollLeft = clamp(els.curveScroll.scrollLeft + delta, 0, maxScroll);
    ctx.timeline.syncHorizontalScroll(els.curveScroll);
    ctx.timeline.ensureFrameInView(state.currentFrame);
    updateControlValues();
    ctx.timeline.renderTimeline();
    ctx.curves.renderCurves();
  }

  function applyBottomHeight(height) {
    const shellHeight = els.appShell.getBoundingClientRect().height;
    const maxHeight = Math.max(240, shellHeight - 360);
    const next = clamp(Math.round(height), 220, maxHeight);
    els.appShell.style.setProperty("--bottom-height", `${next}px`);
    localStorage.setItem(STORAGE.bottomHeight, String(next));
    ctx.timeline.updateTimelineContentSize();
    ctx.timeline.renderTimeline();
    ctx.curves.renderCurves();
  }

  function applySideWidth(width) {
    const viewerWidth = els.viewer.getBoundingClientRect().width;
    const maxWidth = Math.max(260, viewerWidth - 460);
    const next = clamp(Math.round(width), 260, Math.min(620, maxWidth));
    els.appShell.style.setProperty("--side-width", `${next}px`);
    localStorage.setItem(STORAGE.sideWidth, String(next));
    ctx.timeline.updateTimelineContentSize();
    ctx.timeline.renderTimeline();
    ctx.curves.renderCurves();
  }

  function applyTimelineHeight(height) {
    const wrapRect = document.querySelector(".timeline-wrap").getBoundingClientRect();
    const maxHeight = Math.max(54, wrapRect.height - 180);
    const next = clamp(Math.round(height), 54, maxHeight);
    els.appShell.style.setProperty("--timeline-height", `${next}px`);
    localStorage.setItem(STORAGE.timelineHeight, String(next));
    ctx.timeline.renderTimeline();
    ctx.curves.renderCurves();
  }

  function initResizeHandle() {
    const savedBottomValue = localStorage.getItem(STORAGE.bottomHeight);
    const savedBottom = savedBottomValue === null ? null : storedNumber(STORAGE.bottomHeight, null);
    if (savedBottom) els.appShell.style.setProperty("--bottom-height", `${savedBottom}px`);
    const savedSide = storedNumber(STORAGE.sideWidth, 320);
    if (savedSide) els.appShell.style.setProperty("--side-width", `${savedSide}px`);
    const savedTimeline = storedNumber(STORAGE.timelineHeight, 76);
    if (savedTimeline) els.appShell.style.setProperty("--timeline-height", `${savedTimeline}px`);

    els.mainResizeHandle.addEventListener("pointerdown", (event) => {
      const bottomRect = document.querySelector(".timeline-wrap").getBoundingClientRect();
      state.resizeDrag = {
        type: "bottom",
        pointerId: event.pointerId,
        startY: event.clientY,
        startHeight: bottomRect.height,
      };
      els.mainResizeHandle.setPointerCapture(event.pointerId);
      els.mainResizeHandle.classList.add("dragging");
      event.preventDefault();
    });

    els.mainResizeHandle.addEventListener("pointermove", (event) => {
      if (state.resizeDrag?.type !== "bottom") return;
      const delta = state.resizeDrag.startY - event.clientY;
      applyBottomHeight(state.resizeDrag.startHeight + delta);
    });

    function finishDrag(event) {
      if (state.resizeDrag?.type !== "bottom") return;
      if (els.mainResizeHandle.hasPointerCapture(state.resizeDrag.pointerId)) {
        els.mainResizeHandle.releasePointerCapture(state.resizeDrag.pointerId);
      }
      state.resizeDrag = null;
      els.mainResizeHandle.classList.remove("dragging");
      event.preventDefault();
    }

    els.mainResizeHandle.addEventListener("pointerup", finishDrag);
    els.mainResizeHandle.addEventListener("pointercancel", finishDrag);

    els.viewerResizeHandle.addEventListener("pointerdown", (event) => {
      const sideRect = document.querySelector(".side-panel").getBoundingClientRect();
      state.resizeDrag = {
        type: "side",
        pointerId: event.pointerId,
        startX: event.clientX,
        startWidth: sideRect.width,
      };
      els.viewerResizeHandle.setPointerCapture(event.pointerId);
      els.viewerResizeHandle.classList.add("dragging");
      event.preventDefault();
    });

    els.viewerResizeHandle.addEventListener("pointermove", (event) => {
      if (state.resizeDrag?.type !== "side") return;
      const delta = state.resizeDrag.startX - event.clientX;
      applySideWidth(state.resizeDrag.startWidth + delta);
    });

    function finishSideDrag(event) {
      if (state.resizeDrag?.type !== "side") return;
      if (els.viewerResizeHandle.hasPointerCapture(state.resizeDrag.pointerId)) {
        els.viewerResizeHandle.releasePointerCapture(state.resizeDrag.pointerId);
      }
      state.resizeDrag = null;
      els.viewerResizeHandle.classList.remove("dragging");
      event.preventDefault();
    }

    els.viewerResizeHandle.addEventListener("pointerup", finishSideDrag);
    els.viewerResizeHandle.addEventListener("pointercancel", finishSideDrag);

    els.curveResizeHandle.addEventListener("pointerdown", (event) => {
      const timelineRect = els.timelineScroll.getBoundingClientRect();
      state.resizeDrag = {
        type: "timeline",
        pointerId: event.pointerId,
        startY: event.clientY,
        startHeight: timelineRect.height,
      };
      els.curveResizeHandle.setPointerCapture(event.pointerId);
      els.curveResizeHandle.classList.add("dragging");
      event.preventDefault();
    });

    els.curveResizeHandle.addEventListener("pointermove", (event) => {
      if (state.resizeDrag?.type !== "timeline") return;
      const delta = event.clientY - state.resizeDrag.startY;
      applyTimelineHeight(state.resizeDrag.startHeight + delta);
    });

    function finishTimelineDrag(event) {
      if (state.resizeDrag?.type !== "timeline") return;
      if (els.curveResizeHandle.hasPointerCapture(state.resizeDrag.pointerId)) {
        els.curveResizeHandle.releasePointerCapture(state.resizeDrag.pointerId);
      }
      state.resizeDrag = null;
      els.curveResizeHandle.classList.remove("dragging");
      event.preventDefault();
    }

    els.curveResizeHandle.addEventListener("pointerup", finishTimelineDrag);
    els.curveResizeHandle.addEventListener("pointercancel", finishTimelineDrag);
  }

  return { initResizeHandle, setZoom, updateControlValues };
}

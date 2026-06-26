import { clamp, getSegments } from "./utils.js";

export function createCurveController(ctx) {
  const { state, els } = ctx;

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(300, Math.floor(rect.width * ratio));
    const height = Math.max(160, Math.floor(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { width, height, ratio };
  }

  function hexToRgba(hex, alpha) {
    const value = hex.replace("#", "");
    const full = value.length === 3 ? value.split("").map((part) => part + part).join("") : value;
    const num = Number.parseInt(full, 16);
    if (!Number.isFinite(num)) return `rgba(87,166,255,${alpha})`;
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function buildCurveCache() {
    const rows = state.episode?.curves?.["observation.state"] || [];
    const dims = rows[0]?.length || 0;
    const values = Array.from({ length: dims }, () => new Float32Array(rows.length));
    const mins = Array(dims).fill(Infinity);
    const maxs = Array(dims).fill(-Infinity);

    for (let frame = 0; frame < rows.length; frame += 1) {
      const row = rows[frame] || [];
      for (let dim = 0; dim < dims; dim += 1) {
        const value = Number(row[dim] || 0);
        values[dim][frame] = value;
        if (value < mins[dim]) mins[dim] = value;
        if (value > maxs[dim]) maxs[dim] = value;
      }
    }

    for (let dim = 0; dim < dims; dim += 1) {
      if (!Number.isFinite(mins[dim]) || Math.abs(maxs[dim] - mins[dim]) < 1e-6) {
        mins[dim] = -1;
        maxs[dim] = 1;
      }
    }
    return { dims, length: rows.length, values, mins, maxs };
  }

  function rangeIndexes(start, count, limit) {
    const out = [];
    for (let dim = start; dim < Math.min(limit, start + count); dim += 1) out.push(dim);
    return out;
  }

  function curveDimensionIndexes() {
    const dims = state.curveCache?.dims || 0;
    if (!dims) return [];
    const rightStart = dims >= 15 ? 8 : Math.min(7, dims);
    if (state.curveGroup === "left") return rangeIndexes(0, 7, dims);
    if (state.curveGroup === "right") return rangeIndexes(rightStart, 7, dims);
    if (state.curveGroup === "arms") {
      return [...rangeIndexes(0, 7, dims), ...rangeIndexes(rightStart, 7, dims)];
    }
    return rangeIndexes(0, dims, dims);
  }

  function curveLabel(dim) {
    return state.episode?.state_names?.[dim] || `j${dim}`;
  }

  function renderCurves() {
    const canvas = els.curveCanvas;
    const ctx2d = canvas.getContext("2d");
    ctx.timeline.updateTimelineContentSize();
    const dims = curveDimensionIndexes();
    const laneCssHeight = 48;
    const targetHeight = Math.max(els.curveScroll.clientHeight - 2, 28 + Math.max(1, dims.length) * laneCssHeight);
    canvas.style.height = `${targetHeight}px`;
    const { width, height, ratio } = resizeCanvas(canvas);
    ctx2d.clearRect(0, 0, width, height);
    ctx2d.fillStyle = "#0c0f12";
    ctx2d.fillRect(0, 0, width, height);
    if (!state.episode || !state.curveCache || !dims.length) return;

    const scrollLeft = (els.curveScroll?.scrollLeft || 0) * ratio;
    const visibleWidth = ctx.timeline.timelineViewportWidth() * ratio;
    const labelX = scrollLeft + 10 * ratio;
    const labelW = 124 * ratio;
    const padL = 0;
    const padR = 14 * ratio;
    const padT = 12 * ratio;
    const padB = 16 * ratio;
    const plotW = Math.max(1, width - padL - padR);
    const plotH = Math.max(1, height - padT - padB);
    const laneH = Math.max(1, plotH / dims.length);
    const { start, end } = ctx.timeline.visibleRange();
    const xForFrame = (frame) => ctx.timeline.frameToX(frame) * ratio;
    const xForBoundary = (frame) => ctx.timeline.boundaryToX(frame) * ratio;
    const colors = [
      "#57a6ff",
      "#68d391",
      "#f4bf63",
      "#ef6b73",
      "#b58cff",
      "#67d7e5",
      "#d9e368",
      "#f5987a",
    ];

    ctx2d.lineWidth = ratio;
    ctx2d.strokeStyle = "rgba(255,255,255,0.08)";
    ctx2d.fillStyle = "rgba(255,255,255,0.5)";
    ctx2d.font = `${11 * ratio}px Inter, system-ui, sans-serif`;
    ctx2d.textBaseline = "middle";

    const analysis = state.analysisResult;
    if (analysis?.windows?.length && analysis?.clusters?.length) {
      const colorByCluster = new Map(analysis.clusters.map((cluster) => [cluster.id, cluster.color]));
      for (const window of analysis.windows) {
        if (Number(window.episode_index) !== state.episodeIndex) continue;
        if (window.end_frame < start || window.start_frame > end) continue;
        const color = colorByCluster.get(window.dominant_cluster) || "#57a6ff";
        const x = xForBoundary(window.start_frame);
        const w = Math.max(1, xForBoundary(window.end_frame) - x);
        ctx2d.fillStyle = hexToRgba(color, 0.15);
        ctx2d.fillRect(x, padT, w, plotH);
      }
    }

    for (let idx = 0; idx < dims.length; idx += 1) {
      const top = padT + idx * laneH;
      const center = top + laneH / 2;
      ctx2d.strokeStyle = "rgba(255,255,255,0.08)";
      ctx2d.beginPath();
      ctx2d.moveTo(padL, center);
      ctx2d.lineTo(width - padR, center);
      ctx2d.stroke();
      if (laneH > 13 * ratio) {
        ctx2d.fillStyle = "rgba(12,15,18,0.82)";
        ctx2d.fillRect(labelX - 6 * ratio, top + 3 * ratio, labelW, Math.max(16 * ratio, laneH - 6 * ratio));
        ctx2d.fillStyle = "rgba(255,255,255,0.56)";
        ctx2d.fillText(curveLabel(dims[idx]), labelX, center);
      }
    }

    const segments = getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments);
    for (const seg of segments) {
      if (!seg.deleted) continue;
      const segStart = Math.max(seg.start, start);
      const segEnd = Math.min(seg.end, end);
      if (segStart >= segEnd) continue;
      ctx2d.fillStyle = "rgba(239,107,115,0.18)";
      ctx2d.fillRect(xForBoundary(segStart), padT, Math.max(1, xForBoundary(segEnd) - xForBoundary(segStart)), plotH);
    }

    const step = Math.max(1, Math.floor((end - start) / Math.max(1, visibleWidth / (2 * ratio))));
    for (let idx = 0; idx < dims.length; idx += 1) {
      const dim = dims[idx];
      const values = state.curveCache.values[dim];
      const min = state.curveCache.mins[dim];
      const max = state.curveCache.maxs[dim];
      const mid = (min + max) / 2;
      const range = Math.max(1e-6, max - min);
      const top = padT + idx * laneH + 2 * ratio;
      const bottom = padT + (idx + 1) * laneH - 2 * ratio;
      const center = (top + bottom) / 2;
      const amp = ((bottom - top) / 2) * 0.86 * state.curveScale;

      ctx2d.strokeStyle = colors[idx % colors.length];
      ctx2d.globalAlpha = 0.95;
      ctx2d.lineWidth = Math.max(1, 1.25 * ratio);
      ctx2d.beginPath();

      let first = true;
      for (let frame = start; frame < end; frame += step) {
        const value = values[frame] ?? 0;
        const normalized = ((value - mid) / range) * 2;
        const x = xForFrame(frame);
        const y = clamp(center - normalized * amp, top, bottom);
        if (first) {
          ctx2d.moveTo(x, y);
          first = false;
        } else {
          ctx2d.lineTo(x, y);
        }
      }

      if (end - 1 > start) {
        const value = values[end - 1] ?? 0;
        const normalized = ((value - mid) / range) * 2;
        ctx2d.lineTo(xForFrame(end - 1), clamp(center - normalized * amp, top, bottom));
      }
      ctx2d.stroke();
    }
    ctx2d.globalAlpha = 1;

    if (state.currentFrame >= start && state.currentFrame < end) {
      const x = xForFrame(state.currentFrame);
      ctx2d.strokeStyle = "#f4bf63";
      ctx2d.lineWidth = 2 * ratio;
      ctx2d.beginPath();
      ctx2d.moveTo(x, padT);
      ctx2d.lineTo(x, height - padB);
      ctx2d.stroke();
    }
  }

  return { buildCurveCache, renderCurves };
}

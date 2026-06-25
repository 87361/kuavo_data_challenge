const STORAGE = {
  bottomHeight: "lerobotEditor.bottomHeight",
  cameraFractions: "lerobotEditor.cameraFractions",
  curveGroup: "lerobotEditor.curveGroup",
  curveScale: "lerobotEditor.curveScale",
  sideWidth: "lerobotEditor.sideWidth",
  timelineHeight: "lerobotEditor.timelineHeight",
  zoom: "lerobotEditor.zoom",
  analysisWindow: "lerobotEditor.analysisWindow",
  analysisK: "lerobotEditor.analysisK",
  urdfPath: "lerobotEditor.urdfPath",
};

function storedNumber(key, fallback) {
  const value = Number(localStorage.getItem(key));
  return Number.isFinite(value) ? value : fallback;
}

function storedString(key, fallback) {
  return localStorage.getItem(key) || fallback;
}

function storedJson(key, fallback) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "");
    return value ?? fallback;
  } catch (_) {
    return fallback;
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

const state = {
  datasets: [],
  dataset: null,
  episode: null,
  episodeIndex: 0,
  currentFrame: 0,
  selectedSegment: null,
  playing: false,
  timer: null,
  playbackRaf: null,
  lastFollowerSync: 0,
  history: [],
  future: [],
  curveCache: null,
  analysisResult: null,
  analysisPoll: null,
  zoom: clamp(storedNumber(STORAGE.zoom, 1), 1, 24),
  viewStart: 0,
  curveGroup: storedString(STORAGE.curveGroup, "left"),
  curveScale: clamp(storedNumber(STORAGE.curveScale, 1), 0.5, 4),
  timelineContentWidth: 0,
  scrollSync: false,
  cameraFractions: [],
  resizeDrag: null,
};

const els = {
  appShell: document.getElementById("appShell"),
  datasetSelect: document.getElementById("datasetSelect"),
  loadDataset: document.getElementById("loadDataset"),
  episodeSelect: document.getElementById("episodeSelect"),
  statusText: document.getElementById("statusText"),
  viewer: document.getElementById("viewer"),
  videoGrid: document.getElementById("videoGrid"),
  viewerResizeHandle: document.getElementById("viewerResizeHandle"),
  prevFrame: document.getElementById("prevFrame"),
  playPause: document.getElementById("playPause"),
  nextFrame: document.getElementById("nextFrame"),
  cutFrame: document.getElementById("cutFrame"),
  deleteSegment: document.getElementById("deleteSegment"),
  undoEdit: document.getElementById("undoEdit"),
  redoEdit: document.getElementById("redoEdit"),
  urdfPath: document.getElementById("urdfPath"),
  exportPath: document.getElementById("exportPath"),
  exportDataset: document.getElementById("exportDataset"),
  mainResizeHandle: document.getElementById("mainResizeHandle"),
  timelineScroll: document.getElementById("timelineScroll"),
  timeline: document.getElementById("timeline"),
  curveResizeHandle: document.getElementById("curveResizeHandle"),
  curveScroll: document.getElementById("curveScroll"),
  curveCanvas: document.getElementById("curveCanvas"),
  zoomOut: document.getElementById("zoomOut"),
  zoomSlider: document.getElementById("zoomSlider"),
  zoomIn: document.getElementById("zoomIn"),
  zoomReset: document.getElementById("zoomReset"),
  curveGroup: document.getElementById("curveGroup"),
  curveScale: document.getElementById("curveScale"),
  datasetName: document.getElementById("datasetName"),
  episodeMetric: document.getElementById("episodeMetric"),
  frameMetric: document.getElementById("frameMetric"),
  timeMetric: document.getElementById("timeMetric"),
  segmentMetric: document.getElementById("segmentMetric"),
  analysisWindow: document.getElementById("analysisWindow"),
  analysisK: document.getElementById("analysisK"),
  runAnalysis: document.getElementById("runAnalysis"),
  analysisStatus: document.getElementById("analysisStatus"),
  clusterLegend: document.getElementById("clusterLegend"),
};

function setStatus(text) {
  els.statusText.textContent = text;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      // keep fallback
    }
    throw new Error(detail);
  }
  return response.json();
}

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

function getSegments(length, cuts, deletedSegments) {
  const boundaries = [0, ...Array.from(new Set(cuts)).sort((a, b) => a - b), length];
  const deleted = new Set(deletedSegments);
  const segments = [];
  for (let i = 0; i < boundaries.length - 1; i += 1) {
    if (boundaries[i] >= boundaries[i + 1]) continue;
    segments.push({
      index: i,
      start: boundaries[i],
      end: boundaries[i + 1],
      length: boundaries[i + 1] - boundaries[i],
      deleted: deleted.has(i),
    });
  }
  return segments;
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

function videoUrl(key) {
  const params = new URLSearchParams({
    episode_index: state.episodeIndex,
    video_key: key,
  });
  return `/api/video?${params.toString()}`;
}

function frameUrl(key, frame) {
  const params = new URLSearchParams({
    episode_index: state.episodeIndex,
    frame_index: frame,
    video_key: key,
    max_width: 720,
  });
  return `/api/frame?${params.toString()}`;
}

function videoElements() {
  return Array.from(els.videoGrid.querySelectorAll("video"));
}

function playableVideoElements() {
  return videoElements().filter((video) => video.dataset.failed !== "1");
}

function renderFrameFallbacks() {
  for (const image of els.videoGrid.querySelectorAll("img.fallback-frame")) {
    const tile = image.closest(".camera");
    if (!tile?.classList.contains("video-failed")) continue;
    image.src = frameUrl(image.dataset.key, state.currentFrame);
  }
}

function syncVideosToFrame(frame, force = false) {
  const target = frameToTime(frame);
  const tolerance = 0.5 / fps();
  for (const video of playableVideoElements()) {
    if (video.readyState === 0) {
      video.dataset.pendingFrame = String(frame);
      continue;
    }
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const targetTime = duration > 0 ? Math.min(target, Math.max(0, duration - 0.001)) : target;
    if (force || Math.abs(video.currentTime - targetTime) > tolerance) {
      video.currentTime = targetTime;
    }
  }
}

async function loadDatasets() {
  setStatus("Scanning");
  const payload = await api("/api/datasets");
  state.datasets = payload.datasets;
  els.datasetSelect.innerHTML = "";
  for (const item of state.datasets) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = `${item.version} ${item.editable ? "" : "(view only)"} ${item.path}`;
    option.disabled = !item.editable;
    els.datasetSelect.appendChild(option);
  }
  const firstEditable = state.datasets.find((item) => item.editable);
  if (firstEditable) els.datasetSelect.value = firstEditable.path;
  setStatus(`${state.datasets.length} datasets`);
}

function defaultExportPath(datasetPath) {
  const stamp = new Date().toISOString().replaceAll(":", "").replace(/\..+/, "").replace("T", "_");
  const name = datasetPath.split("/").filter(Boolean).slice(-3, -1).join("_") || "lerobot";
  return `/mnt/data/kuavo_tianchi/lerobot_edits/${name}_edited_${stamp}/lerobot`;
}

async function openDataset() {
  const path = els.datasetSelect.value;
  if (!path) return;
  stopPlayback();
  stopAnalysisPolling();
  setStatus("Loading dataset");
  state.dataset = await api("/api/open", { method: "POST", body: JSON.stringify({ path }) });
  state.analysisResult = null;
  state.history = [];
  state.future = [];
  els.episodeSelect.innerHTML = "";
  for (const episode of state.dataset.episodes) {
    const option = document.createElement("option");
    option.value = episode.episode_index;
    option.textContent = `${episode.episode_index.toString().padStart(4, "0")}  ${episode.length}f`;
    els.episodeSelect.appendChild(option);
  }
  els.exportPath.value = defaultExportPath(path);
  if (!els.urdfPath.value.trim()) {
    els.urdfPath.value = storedString(STORAGE.urdfPath, "") || state.dataset.urdf_path || "";
  }
  els.datasetName.textContent = state.dataset.path.split("/").slice(-3).join("/");
  renderAnalysisResult();
  await loadEpisode(0);
  setStatus("Ready");
}

async function loadEpisode(index) {
  if (!state.dataset) return;
  stopPlayback();
  state.episodeIndex = Number(index);
  state.episode = await api(`/api/episode/${state.episodeIndex}`);
  state.currentFrame = 0;
  state.viewStart = 0;
  state.selectedSegment = null;
  state.curveCache = buildCurveCache();
  renderVideoGrid();
  updateControlValues();
  updateTimelineContentSize();
  renderAll();
}

function getCameraFractions(count) {
  if (count <= 0) return [];
  const saved = storedJson(STORAGE.cameraFractions, null);
  if (
    Array.isArray(saved) &&
    saved.length === count &&
    saved.every((value) => Number.isFinite(Number(value)) && Number(value) > 0)
  ) {
    const total = saved.reduce((sum, value) => sum + Number(value), 0);
    return saved.map((value) => Number(value) / total);
  }
  return Array.from({ length: count }, () => 1 / count);
}

function applyCameraFractions(save = false) {
  const cameras = Array.from(els.videoGrid.querySelectorAll(".camera"));
  if (!cameras.length) return;
  if (state.cameraFractions.length !== cameras.length) {
    state.cameraFractions = getCameraFractions(cameras.length);
  }
  cameras.forEach((camera, index) => {
    const fraction = state.cameraFractions[index] || 1 / cameras.length;
    camera.style.flexGrow = String(fraction);
    camera.style.flexShrink = "1";
    camera.style.flexBasis = "0";
  });
  if (save) localStorage.setItem(STORAGE.cameraFractions, JSON.stringify(state.cameraFractions));
}

function normalizeFractions(fractions) {
  const total = fractions.reduce((sum, value) => sum + value, 0) || 1;
  return fractions.map((value) => value / total);
}

function renderVideoGrid() {
  els.videoGrid.innerHTML = "";
  if (!state.episode) return;
  state.cameraFractions = getCameraFractions(state.episode.video_keys.length);
  state.episode.video_keys.forEach((key, index) => {
    const tile = document.createElement("div");
    tile.className = "camera";

    const title = document.createElement("div");
    title.className = "camera-title";
    title.textContent = key;

    const stage = document.createElement("div");
    stage.className = "camera-stage";

    const video = document.createElement("video");
    video.dataset.key = key;
    video.src = videoUrl(key);
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.disablePictureInPicture = true;

    const fallback = document.createElement("img");
    fallback.className = "fallback-frame";
    fallback.dataset.key = key;
    fallback.alt = key;

    video.addEventListener("loadedmetadata", () => {
      const pending = Number(video.dataset.pendingFrame || state.currentFrame);
      delete video.dataset.pendingFrame;
      syncVideosToFrame(pending, true);
    });
    video.addEventListener("error", () => {
      video.dataset.failed = "1";
      tile.classList.add("video-failed");
      renderFrameFallbacks();
      setStatus(`Browser video decode failed for ${key}; using frame fallback`);
    });

    stage.appendChild(video);
    stage.appendChild(fallback);
    tile.appendChild(title);
    tile.appendChild(stage);
    els.videoGrid.appendChild(tile);
    if (index < state.episode.video_keys.length - 1) {
      const handle = document.createElement("div");
      handle.className = "camera-resize-handle";
      handle.dataset.index = String(index);
      handle.title = "Resize camera panels";
      els.videoGrid.appendChild(handle);
    }
  });
  applyCameraFractions(false);
  initCameraResizeHandles();
}

function renderMetrics() {
  if (!state.episode) return;
  const seg = currentSegment();
  els.episodeMetric.textContent = `${state.episodeIndex} / ${state.dataset.total_episodes - 1}`;
  els.frameMetric.textContent = `${state.currentFrame} / ${state.episode.length - 1}`;
  els.timeMetric.textContent = `${(state.currentFrame / fps()).toFixed(2)}s`;
  els.segmentMetric.textContent = seg ? `${seg.index}  ${seg.start}-${seg.end - 1}` : "-";
}

function currentSegment() {
  if (!state.episode) return null;
  return getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments).find(
    (seg) => state.currentFrame >= seg.start && state.currentFrame < seg.end,
  );
}

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

function ensureFrameInView(frame) {
  if (!state.episode) {
    return false;
  }
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

function syncHorizontalScroll(source) {
  if (!source || state.scrollSync) return;
  const target = source === els.timelineScroll ? els.curveScroll : els.timelineScroll;
  if (!target) return;
  state.scrollSync = true;
  target.scrollLeft = source.scrollLeft;
  state.scrollSync = false;
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
  const ctx = canvas.getContext("2d");
  updateTimelineContentSize();
  const dims = curveDimensionIndexes();
  const laneCssHeight = 48;
  const targetHeight = Math.max(els.curveScroll.clientHeight - 2, 28 + Math.max(1, dims.length) * laneCssHeight);
  canvas.style.height = `${targetHeight}px`;
  const { width, height, ratio } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0c0f12";
  ctx.fillRect(0, 0, width, height);
  if (!state.episode || !state.curveCache) return;

  if (!dims.length) return;

  const scrollLeft = (els.curveScroll?.scrollLeft || 0) * ratio;
  const visibleWidth = timelineViewportWidth() * ratio;
  const labelX = scrollLeft + 10 * ratio;
  const labelW = 124 * ratio;
  const padL = 0;
  const padR = 14 * ratio;
  const padT = 12 * ratio;
  const padB = 16 * ratio;
  const plotW = Math.max(1, width - padL - padR);
  const plotH = Math.max(1, height - padT - padB);
  const laneH = Math.max(1, plotH / dims.length);
  const { start, end } = visibleRange();
  const xForFrame = (frame) => frameToX(frame) * ratio;
  const xForBoundary = (frame) => boundaryToX(frame) * ratio;
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

  ctx.lineWidth = ratio;
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.font = `${11 * ratio}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = "middle";

  const analysis = state.analysisResult;
  if (analysis?.windows?.length && analysis?.clusters?.length) {
    const colorByCluster = new Map(analysis.clusters.map((cluster) => [cluster.id, cluster.color]));
    for (const window of analysis.windows) {
      if (Number(window.episode_index) !== state.episodeIndex) continue;
      if (window.end_frame < start || window.start_frame > end) continue;
      const color = colorByCluster.get(window.dominant_cluster) || "#57a6ff";
      const x = xForBoundary(window.start_frame);
      const w = Math.max(1, xForBoundary(window.end_frame) - x);
      ctx.fillStyle = hexToRgba(color, 0.15);
      ctx.fillRect(x, padT, w, plotH);
    }
  }

  for (let idx = 0; idx < dims.length; idx += 1) {
    const top = padT + idx * laneH;
    const center = top + laneH / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(padL, center);
    ctx.lineTo(width - padR, center);
    ctx.stroke();
    if (laneH > 13 * ratio) {
      ctx.fillStyle = "rgba(12,15,18,0.82)";
      ctx.fillRect(labelX - 6 * ratio, top + 3 * ratio, labelW, Math.max(16 * ratio, laneH - 6 * ratio));
      ctx.fillStyle = "rgba(255,255,255,0.56)";
      ctx.fillText(curveLabel(dims[idx]), labelX, center);
    }
  }

  const segments = getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments);
  for (const seg of segments) {
    if (!seg.deleted) continue;
    const segStart = Math.max(seg.start, start);
    const segEnd = Math.min(seg.end, end);
    if (segStart >= segEnd) continue;
    ctx.fillStyle = "rgba(239,107,115,0.18)";
    ctx.fillRect(xForBoundary(segStart), padT, Math.max(1, xForBoundary(segEnd) - xForBoundary(segStart)), plotH);
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

    ctx.strokeStyle = colors[idx % colors.length];
    ctx.globalAlpha = 0.95;
    ctx.lineWidth = Math.max(1, 1.25 * ratio);
    ctx.beginPath();

    let first = true;
    for (let frame = start; frame < end; frame += step) {
      const value = values[frame] ?? 0;
      const normalized = ((value - mid) / range) * 2;
      const x = xForFrame(frame);
      const y = clamp(center - normalized * amp, top, bottom);
      if (first) {
        ctx.moveTo(x, y);
        first = false;
      } else {
        ctx.lineTo(x, y);
      }
    }

    if (end - 1 > start) {
      const value = values[end - 1] ?? 0;
      const normalized = ((value - mid) / range) * 2;
      ctx.lineTo(xForFrame(end - 1), clamp(center - normalized * amp, top, bottom));
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  if (state.currentFrame >= start && state.currentFrame < end) {
    const x = xForFrame(state.currentFrame);
    ctx.strokeStyle = "#f4bf63";
    ctx.lineWidth = 2 * ratio;
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, height - padB);
    ctx.stroke();
  }
}

function renderAll() {
  syncVideosToFrame(state.currentFrame, true);
  renderFrameFallbacks();
  renderMetrics();
  renderTimeline();
  renderCurves();
}

function setCurrentFrame(frame, options = {}) {
  if (!state.episode) return;
  const nextFrame = clamp(Math.round(frame), 0, state.episode.length - 1);
  const oldScrollLeft = els.curveScroll.scrollLeft;
  state.currentFrame = nextFrame;
  const viewportChanged = ensureFrameInView(nextFrame) || oldScrollLeft !== els.curveScroll.scrollLeft;
  if (options.seek !== false) syncVideosToFrame(nextFrame, Boolean(options.forceSeek));
  renderFrameFallbacks();
  renderMetrics();
  if (viewportChanged || options.forceTimeline) renderTimeline();
  else renderTimelinePlayhead();
  renderCurves();
}

function moveFrame(delta) {
  if (!state.episode) return;
  if (state.playing) stopPlayback();
  setCurrentFrame(state.currentFrame + delta, { seek: true, forceTimeline: false });
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
  state.selectedSegment = currentSegment()?.index ?? null;
  renderAll();
}

async function cutAtFrame() {
  if (!state.episode || state.currentFrame <= 0 || state.currentFrame >= state.episode.length) return;
  if (state.playing) stopPlayback();
  pushHistory();
  const oldDeletedRanges = deletedRangesFromEpisode();
  const cuts = Array.from(new Set([...state.episode.cuts, state.currentFrame])).sort((a, b) => a - b);
  state.episode.cuts = cuts;
  state.episode.deleted_segments = deletedSegmentsForRanges(cuts, oldDeletedRanges);
  await syncCuts();
}

async function toggleDeleteSegment() {
  if (!state.episode) return;
  if (state.playing) stopPlayback();
  const seg = state.selectedSegment ?? currentSegment()?.index;
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
  if (state.playing) stopPlayback();
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

function stopPlayback() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  if (state.playbackRaf) {
    cancelAnimationFrame(state.playbackRaf);
    state.playbackRaf = null;
  }
  for (const video of playableVideoElements()) video.pause();
  state.playing = false;
  els.playPause.textContent = "Play";
}

function syncFollowerVideos(leader) {
  const now = performance.now();
  if (now - state.lastFollowerSync < 450) return;
  state.lastFollowerSync = now;
  for (const video of playableVideoElements()) {
    if (video === leader || video.readyState === 0) continue;
    if (Math.abs(video.currentTime - leader.currentTime) > 0.08) {
      video.currentTime = leader.currentTime;
    }
  }
}

function updatePlaybackFromVideo() {
  if (!state.playing || !state.episode) return;
  const leader = playableVideoElements()[0];
  if (!leader) return;

  const nextFrame = timeToFrame(leader.currentTime);
  if (nextFrame !== state.currentFrame) {
    setCurrentFrame(nextFrame, { seek: false });
  }
  syncFollowerVideos(leader);

  if (leader.ended || state.currentFrame >= state.episode.length - 1) {
    setCurrentFrame(state.episode.length - 1, { seek: false });
    stopPlayback();
    return;
  }
  state.playbackRaf = requestAnimationFrame(updatePlaybackFromVideo);
}

function startFrameFallbackPlayback() {
  const intervalMs = 1000 / fps();
  state.timer = setInterval(() => {
    if (!state.episode) return;
    if (state.currentFrame >= state.episode.length - 1) {
      stopPlayback();
      return;
    }
    setCurrentFrame(state.currentFrame + 1, { seek: false });
  }, intervalMs);
}

async function startNativePlayback() {
  syncVideosToFrame(state.currentFrame, true);
  const videos = playableVideoElements();
  if (!videos.length) {
    startFrameFallbackPlayback();
    return;
  }
  const attempts = await Promise.allSettled(videos.map((video) => video.play()));
  const ok = attempts.some((result) => result.status === "fulfilled");
  if (!ok) {
    throw new Error(attempts[0]?.reason?.message || "video playback failed");
  }
  state.lastFollowerSync = 0;
  state.playbackRaf = requestAnimationFrame(updatePlaybackFromVideo);
}

function playPause() {
  if (!state.episode) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (state.currentFrame >= state.episode.length - 1) {
    setCurrentFrame(0, { seek: true, forceTimeline: true });
  }
  state.playing = true;
  els.playPause.textContent = "Pause";
  startNativePlayback().catch((error) => {
    stopPlayback();
    setStatus(error.message);
  });
}

function setZoom(value) {
  if (!state.episode) return;
  const beforeX = frameToX(state.currentFrame);
  const nextZoom = clamp(Number(value) || 1, 1, 24);
  state.zoom = nextZoom;
  localStorage.setItem(STORAGE.zoom, String(nextZoom));
  updateTimelineContentSize();
  const afterX = frameToX(state.currentFrame);
  const delta = afterX - beforeX;
  const maxScroll = Math.max(0, state.timelineContentWidth - els.curveScroll.clientWidth);
  els.curveScroll.scrollLeft = clamp(els.curveScroll.scrollLeft + delta, 0, maxScroll);
  syncHorizontalScroll(els.curveScroll);
  ensureFrameInView(state.currentFrame);
  updateControlValues();
  renderTimeline();
  renderCurves();
}

function updateControlValues() {
  els.zoomSlider.value = String(state.zoom);
  els.curveScale.value = String(state.curveScale);
  if ([...els.curveGroup.options].some((option) => option.value === state.curveGroup)) {
    els.curveGroup.value = state.curveGroup;
  } else {
    els.curveGroup.value = "left";
    state.curveGroup = "left";
  }
}

function initAnalysisControls() {
  els.analysisWindow.value = String(clamp(storedNumber(STORAGE.analysisWindow, 1), 0.1, 60));
  els.analysisK.value = storedString(STORAGE.analysisK, "auto");
  els.urdfPath.value = storedString(STORAGE.urdfPath, "");
  renderAnalysisResult();
}

async function exportDataset() {
  if (!state.dataset) return;
  const outputPath = els.exportPath.value.trim();
  if (!outputPath) return;
  const urdfPath = els.urdfPath.value.trim();
  localStorage.setItem(STORAGE.urdfPath, urdfPath);
  els.exportDataset.disabled = true;
  setStatus(urdfPath ? "Exporting with URDF smoothing" : "Exporting with fixed transitions");
  await api("/api/export", {
    method: "POST",
    body: JSON.stringify({ output_path: outputPath, urdf_path: urdfPath || null }),
  });
  const poll = setInterval(async () => {
    try {
      const job = await api("/api/export/status");
      const pct = Math.round((job.progress || 0) * 100);
      setStatus(`${job.status} ${pct}% ${job.message || ""}`);
      if (job.status === "complete" || job.status === "failed") {
        clearInterval(poll);
        els.exportDataset.disabled = false;
        if (job.status === "failed") setStatus(`failed: ${job.error}`);
      }
    } catch (error) {
      clearInterval(poll);
      els.exportDataset.disabled = false;
      setStatus(error.message);
    }
  }, 1200);
}

function stopAnalysisPolling() {
  if (state.analysisPoll) {
    clearInterval(state.analysisPoll);
    state.analysisPoll = null;
  }
  els.runAnalysis.disabled = false;
}

function analysisClusterCountValue() {
  const raw = els.analysisK.value.trim();
  if (!raw || raw.toLowerCase() === "auto") return "auto";
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) throw new Error("K must be auto or a positive integer");
  return value;
}

function renderAnalysisResult(job = null) {
  const status = job?.status || (state.analysisResult ? "complete" : "idle");
  if (status === "running") {
    const pct = Math.round((job.progress || 0) * 100);
    els.analysisStatus.textContent = `${pct}% ${job.message || "Running"}`;
    els.runAnalysis.disabled = true;
  } else if (status === "failed") {
    els.analysisStatus.textContent = job.error || "Analysis failed";
    els.runAnalysis.disabled = false;
  } else if (state.analysisResult) {
    const result = state.analysisResult;
    els.analysisStatus.textContent = `${result.windows.length} windows, ${result.cluster_count} clusters`;
    els.runAnalysis.disabled = false;
  } else {
    els.analysisStatus.textContent = "Not run";
    els.runAnalysis.disabled = false;
  }

  els.clusterLegend.innerHTML = "";
  const clusters = state.analysisResult?.clusters || [];
  for (const cluster of clusters) {
    const chip = document.createElement("div");
    chip.className = "cluster-chip";
    const swatch = document.createElement("span");
    swatch.className = "cluster-swatch";
    swatch.style.background = cluster.color;
    const label = document.createElement("span");
    label.textContent = `C${cluster.id} ${cluster.window_count}`;
    chip.appendChild(swatch);
    chip.appendChild(label);
    els.clusterLegend.appendChild(chip);
  }
}

async function pollAnalysisStatus() {
  const job = await api("/api/analysis/coverage/status");
  if (job.status === "complete") {
    state.analysisResult = job.result;
    stopAnalysisPolling();
    renderAnalysisResult(job);
    renderCurves();
    renderTimeline();
  } else if (job.status === "failed") {
    stopAnalysisPolling();
    renderAnalysisResult(job);
  } else {
    renderAnalysisResult(job);
  }
}

async function runCoverageAnalysis() {
  if (!state.dataset) return;
  const windowSeconds = Number(els.analysisWindow.value || 1);
  if (!Number.isFinite(windowSeconds) || windowSeconds <= 0) throw new Error("Window must be positive");
  const clusterCount = analysisClusterCountValue();
  localStorage.setItem(STORAGE.analysisWindow, String(windowSeconds));
  localStorage.setItem(STORAGE.analysisK, String(clusterCount));
  stopAnalysisPolling();
  state.analysisResult = null;
  renderAnalysisResult({ status: "running", progress: 0, message: "Queued" });
  await api("/api/analysis/coverage", {
    method: "POST",
    body: JSON.stringify({ window_seconds: windowSeconds, cluster_count: clusterCount }),
  });
  state.analysisPoll = setInterval(() => {
    pollAnalysisStatus().catch((error) => {
      stopAnalysisPolling();
      els.analysisStatus.textContent = error.message;
    });
  }, 700);
  await pollAnalysisStatus();
}

function frameFromTimelineEvent(event) {
  const rect = els.timeline.getBoundingClientRect();
  const x = clamp(event.clientX - rect.left, 0, state.timelineContentWidth);
  const pct = x / Math.max(1, state.timelineContentWidth);
  return clamp(Math.round(pct * Math.max(0, state.episode.length - 1)), 0, state.episode.length - 1);
}

function handleTimelineClick(event) {
  if (!state.episode) return;
  if (state.playing) stopPlayback();
  const frame = frameFromTimelineEvent(event);
  const seg = getSegments(state.episode.length, state.episode.cuts, state.episode.deleted_segments).find(
    (item) => frame >= item.start && frame < item.end,
  );
  state.selectedSegment = seg?.index ?? null;
  setCurrentFrame(frame, { seek: true, forceTimeline: true });
}

function applyBottomHeight(height) {
  const shellHeight = els.appShell.getBoundingClientRect().height;
  const maxHeight = Math.max(240, shellHeight - 360);
  const next = clamp(Math.round(height), 220, maxHeight);
  els.appShell.style.setProperty("--bottom-height", `${next}px`);
  localStorage.setItem(STORAGE.bottomHeight, String(next));
  updateTimelineContentSize();
  renderTimeline();
  renderCurves();
}

function applySideWidth(width) {
  const viewerWidth = els.viewer.getBoundingClientRect().width;
  const maxWidth = Math.max(260, viewerWidth - 460);
  const next = clamp(Math.round(width), 260, Math.min(620, maxWidth));
  els.appShell.style.setProperty("--side-width", `${next}px`);
  localStorage.setItem(STORAGE.sideWidth, String(next));
  updateTimelineContentSize();
  renderTimeline();
  renderCurves();
}

function applyTimelineHeight(height) {
  const wrapRect = document.querySelector(".timeline-wrap").getBoundingClientRect();
  const maxHeight = Math.max(54, wrapRect.height - 180);
  const next = clamp(Math.round(height), 54, maxHeight);
  els.appShell.style.setProperty("--timeline-height", `${next}px`);
  localStorage.setItem(STORAGE.timelineHeight, String(next));
  renderTimeline();
  renderCurves();
}

function initCameraResizeHandles() {
  for (const handle of els.videoGrid.querySelectorAll(".camera-resize-handle")) {
    handle.onpointerdown = (event) => {
      const index = Number(handle.dataset.index);
      const rect = els.videoGrid.getBoundingClientRect();
      state.resizeDrag = {
        type: "camera",
        pointerId: event.pointerId,
        index,
        startX: event.clientX,
        width: Math.max(1, rect.width),
        fractions: [...state.cameraFractions],
      };
      handle.setPointerCapture(event.pointerId);
      handle.classList.add("dragging");
      event.preventDefault();
    };
    handle.onpointermove = (event) => {
      if (state.resizeDrag?.type !== "camera") return;
      const drag = state.resizeDrag;
      const delta = (event.clientX - drag.startX) / drag.width;
      const next = [...drag.fractions];
      const left = drag.index;
      const right = drag.index + 1;
      const min = 0.12;
      const pairTotal = drag.fractions[left] + drag.fractions[right];
      next[left] = clamp(drag.fractions[left] + delta, min, pairTotal - min);
      next[right] = pairTotal - next[left];
      state.cameraFractions = normalizeFractions(next);
      applyCameraFractions(true);
    };
    const finish = (event) => {
      if (state.resizeDrag?.type !== "camera") return;
      if (handle.hasPointerCapture(state.resizeDrag.pointerId)) {
        handle.releasePointerCapture(state.resizeDrag.pointerId);
      }
      state.resizeDrag = null;
      handle.classList.remove("dragging");
      event.preventDefault();
    };
    handle.onpointerup = finish;
    handle.onpointercancel = finish;
  }
}

function initResizeHandle() {
  const savedBottom = storedNumber(STORAGE.bottomHeight, 340);
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

els.loadDataset.addEventListener("click", () => openDataset().catch((err) => setStatus(err.message)));
els.episodeSelect.addEventListener("change", () => loadEpisode(els.episodeSelect.value).catch((err) => setStatus(err.message)));
els.prevFrame.addEventListener("click", () => moveFrame(-1));
els.nextFrame.addEventListener("click", () => moveFrame(1));
els.playPause.addEventListener("click", playPause);
els.cutFrame.addEventListener("click", () => cutAtFrame().catch((err) => setStatus(err.message)));
els.deleteSegment.addEventListener("click", () => toggleDeleteSegment().catch((err) => setStatus(err.message)));
els.undoEdit.addEventListener("click", () => undo().catch((err) => setStatus(err.message)));
els.redoEdit.addEventListener("click", () => redo().catch((err) => setStatus(err.message)));
els.exportDataset.addEventListener("click", () => exportDataset().catch((err) => {
  els.exportDataset.disabled = false;
  setStatus(err.message);
}));
els.runAnalysis.addEventListener("click", () => runCoverageAnalysis().catch((err) => {
  stopAnalysisPolling();
  els.analysisStatus.textContent = err.message;
}));

els.timeline.addEventListener("click", handleTimelineClick);
els.timelineScroll.addEventListener("scroll", () => {
  syncHorizontalScroll(els.timelineScroll);
  renderTimeline();
  renderCurves();
});
els.curveScroll.addEventListener("scroll", () => {
  syncHorizontalScroll(els.curveScroll);
  renderTimeline();
  renderCurves();
});
els.zoomSlider.addEventListener("input", () => setZoom(els.zoomSlider.value));
els.zoomOut.addEventListener("click", () => setZoom(state.zoom / 1.35));
els.zoomIn.addEventListener("click", () => setZoom(state.zoom * 1.35));
els.zoomReset.addEventListener("click", () => setZoom(1));
els.curveGroup.addEventListener("change", () => {
  state.curveGroup = els.curveGroup.value;
  localStorage.setItem(STORAGE.curveGroup, state.curveGroup);
  renderCurves();
});
els.curveScale.addEventListener("input", () => {
  state.curveScale = clamp(Number(els.curveScale.value) || 1, 0.5, 4);
  localStorage.setItem(STORAGE.curveScale, String(state.curveScale));
  renderCurves();
});

document.addEventListener("keydown", (event) => {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    moveFrame(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    moveFrame(1);
  } else if (event.key === " ") {
    event.preventDefault();
    playPause();
  }
});

window.addEventListener("resize", () => {
  updateTimelineContentSize();
  ensureFrameInView(state.currentFrame);
  renderTimeline();
  renderCurves();
});

initResizeHandle();
initAnalysisControls();
updateControlValues();
loadDatasets().catch((err) => setStatus(err.message));

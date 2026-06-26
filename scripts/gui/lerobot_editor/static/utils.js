export const STORAGE = {
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

export function storedNumber(key, fallback) {
  const value = Number(localStorage.getItem(key));
  return Number.isFinite(value) ? value : fallback;
}

export function storedString(key, fallback) {
  return localStorage.getItem(key) || fallback;
}

export function storedJson(key, fallback) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "");
    return value ?? fallback;
  } catch (_) {
    return fallback;
  }
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function getSegments(length, cuts, deletedSegments) {
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

export function normalizeFractions(fractions) {
  const total = fractions.reduce((sum, value) => sum + value, 0) || 1;
  return fractions.map((value) => value / total);
}

export function formatSavedAt(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString();
}

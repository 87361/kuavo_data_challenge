import { STORAGE, clamp, storedNumber, storedString } from "./utils.js";

export function createAnalysisController(ctx) {
  const { state, els, api } = ctx;

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
      ctx.curves.renderCurves();
      ctx.timeline.renderTimeline();
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

  function initAnalysisControls() {
    els.analysisWindow.value = String(clamp(storedNumber(STORAGE.analysisWindow, 1), 0.1, 60));
    els.analysisK.value = storedString(STORAGE.analysisK, "auto");
    els.urdfPath.value = storedString(STORAGE.urdfPath, "");
    renderAnalysisResult();
  }

  return { initAnalysisControls, renderAnalysisResult, runCoverageAnalysis, stopAnalysisPolling };
}

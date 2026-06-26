import { STORAGE } from "./utils.js";

export function createExportController(ctx) {
  const { state, els, api, setStatus } = ctx;

  async function exportDataset() {
    if (!state.dataset) return;
    const outputPath = els.exportPath.value.trim();
    if (!outputPath) return;
    const urdfPath = els.urdfPath.value.trim();
    localStorage.setItem(STORAGE.urdfPath, urdfPath);
    els.exportDataset.disabled = true;
    setStatus(urdfPath ? "Exporting incrementally with URDF smoothing" : "Exporting incrementally with fixed transitions");
    await api("/api/export", {
      method: "POST",
      body: JSON.stringify({ output_path: outputPath, urdf_path: urdfPath || null }),
    });
    if (state.exportPoll) clearInterval(state.exportPoll);
    state.exportPoll = setInterval(async () => {
      try {
        const job = await api("/api/export/status");
        const pct = Math.round((job.progress || 0) * 100);
        setStatus(`${job.status} ${pct}% ${job.message || ""}`);
        if (job.status === "complete" || job.status === "failed") {
          clearInterval(state.exportPoll);
          state.exportPoll = null;
          els.exportDataset.disabled = false;
          if (job.status === "failed") {
            setStatus(`failed: ${job.error}`);
          } else {
            await ctx.progress.loadProgress();
            setStatus(`export complete: ${outputPath}`);
          }
        }
      } catch (error) {
        clearInterval(state.exportPoll);
        state.exportPoll = null;
        els.exportDataset.disabled = false;
        setStatus(error.message);
      }
    }, 1200);
  }

  return { exportDataset };
}

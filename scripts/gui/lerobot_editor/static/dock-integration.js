import DockLayout, { ROOT_TARGET } from "./dock-layout.js";

const STORAGE_KEY = "lerobotEditor.dockLayout.v1";

const defaultLayout = {
  kind: "split",
  direction: "horizontal",
  ratio: 0.72,
  first: {
    kind: "split",
    direction: "vertical",
    ratio: 0.58,
    first: { kind: "panel", id: "viewport-main", type: "viewport", title: "Viewport" },
    second: {
      kind: "split",
      direction: "vertical",
      ratio: 0.45,
      first: { kind: "panel", id: "timeline-main", type: "timeline", title: "Timeline" },
      second: { kind: "panel", id: "curves-main", type: "curves", title: "Curves" },
    },
  },
  second: {
    kind: "split",
    direction: "vertical",
    ratio: 0.48,
    first: { kind: "panel", id: "annotation-main", type: "annotation", title: "Annotation" },
    second: {
      kind: "split",
      direction: "vertical",
      ratio: 0.52,
      first: { kind: "panel", id: "trajectory-main", type: "trajectory", title: "Trajectory Preview" },
      second: { kind: "panel", id: "terminal-main", type: "terminal", title: "Terminal" },
    },
  },
};

const panelTitles = {
  viewport: "Viewport",
  timeline: "Timeline",
  curves: "Curves",
  annotation: "Annotation",
  trajectory: "Trajectory Preview",
  terminal: "Terminal",
};

function hasPanelType(node, type) {
  if (!node) return false;
  if (node.kind === "panel") return node.type === type;
  return hasPanelType(node.first, type) || hasPanelType(node.second, type);
}

export function createDockController(ctx) {
  const { els } = ctx;
  let refreshFrame = null;

  const sources = {
    viewport: els.viewportPanelSource,
    timeline: els.timelinePanelSource,
    curves: els.curvesPanelSource,
    annotation: els.annotationPanelSource,
    trajectory: els.trajectoryPanelSource,
    terminal: els.terminalPanelSource,
  };

  function queueLayoutRefresh() {
    if (refreshFrame !== null) cancelAnimationFrame(refreshFrame);
    refreshFrame = requestAnimationFrame(() => {
      refreshFrame = null;
      ctx.video?.updateCameraLayout();
      ctx.timeline?.updateTimelineContentSize();
      if (ctx.state.episode) {
        ctx.timeline.ensureFrameInView(ctx.state.currentFrame);
        ctx.timeline.renderTimeline();
        ctx.curves.renderCurves();
      }
    });
  }

  const panelTypes = Object.fromEntries(
    Object.entries(sources).map(([type, source]) => [
      type,
      {
        title: panelTitles[type],
        render(container) {
          container.innerHTML = "";
          if (source) container.appendChild(source);
          queueLayoutRefresh();
        },
      },
    ]),
  );

  const dock = new DockLayout(els.dockRoot, {
    storageKey: STORAGE_KEY,
    layout: defaultLayout,
    emptyMessage: "Add a panel from the toolbar",
    panelTypes,
    onLayoutChange: queueLayoutRefresh,
  });

  function showPanel(type) {
    if (!panelTitles[type]) return;
    if (hasPanelType(dock.getLayout(), type)) return;
    dock.createPanel(
      {
        type,
        title: panelTitles[type],
      },
      { targetId: ROOT_TARGET, side: type === "terminal" || type === "trajectory" ? "bottom" : "right" },
    );
  }

  function resetLayout() {
    localStorage.removeItem(STORAGE_KEY);
    dock.setLayout(defaultLayout);
  }

  els.dockPanelButtons.forEach((button) => {
    button.addEventListener("click", () => showPanel(button.dataset.dockPanel));
  });
  els.resetDockLayout.addEventListener("click", resetLayout);

  queueLayoutRefresh();
  return { dock, resetLayout, showPanel };
}

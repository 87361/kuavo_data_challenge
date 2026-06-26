import DockLayout, { ROOT_TARGET } from "./dock-layout.js";

const STORAGE_KEY = "lerobot-editor-dock-demo-layout-v1";

const defaultLayout = {
  kind: "split",
  direction: "horizontal",
  ratio: 0.68,
  first: {
    kind: "split",
    direction: "vertical",
    ratio: 0.66,
    first: {
      kind: "panel",
      id: "viewport-main",
      type: "viewport",
      title: "Main Viewport",
      state: { camera: "head_cam_h", fps: 30 },
    },
    second: {
      kind: "panel",
      id: "timeline-main",
      type: "timeline",
      title: "Episode Timeline",
      state: {},
    },
  },
  second: {
    kind: "split",
    direction: "vertical",
    ratio: 0.54,
    first: {
      kind: "panel",
      id: "inspector-main",
      type: "inspector",
      title: "Inspector",
      state: { dataset: "kuavo_task1", episode: 12, playback: "1x" },
    },
    second: {
      kind: "panel",
      id: "console-main",
      type: "console",
      title: "Console",
      state: {},
    },
  },
};

const panelSpecs = {
  viewport: { title: "Viewport", initialState: () => ({ camera: "wrist_cam", fps: 30 }) },
  inspector: { title: "Inspector", initialState: () => ({ dataset: "kuavo_task1", episode: 0, playback: "1x" }) },
  timeline: { title: "Timeline", initialState: () => ({}) },
  curves: { title: "Curves", initialState: () => ({}) },
  console: { title: "Console", initialState: () => ({}) },
};

const counters = Object.fromEntries(Object.keys(panelSpecs).map((type) => [type, 1]));

const panelCount = document.querySelector("#panelCount");
const layoutStatus = document.querySelector("#layoutStatus");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const dock = new DockLayout("#dockRoot", {
  storageKey: STORAGE_KEY,
  layout: defaultLayout,
  emptyMessage: "Add a panel",
  onLayoutChange: updateStatus,
});

dock.registerPanelType("viewport", {
  title: "Viewport",
  render(container, panel) {
    const camera = escapeHtml(panel.state.camera || "camera");
    const fps = escapeHtml(panel.state.fps || 30);
    container.innerHTML = `
      <div class="viewport">
        <div class="viewport-scene">
          <div class="armature"></div>
          <div class="viewport-target"></div>
        </div>
        <div class="viewport-strip">
          <span>${camera}</span>
          <span>${fps} fps</span>
          <span>RGB</span>
          <span>Live</span>
        </div>
      </div>
    `;
  },
});

dock.registerPanelType("inspector", {
  title: "Inspector",
  render(container, panel, api) {
    const dataset = escapeHtml(panel.state.dataset || "");
    const episode = escapeHtml(panel.state.episode ?? 0);
    container.innerHTML = `
      <div class="panel-pad">
        <div class="metric-grid">
          <div class="metric"><span>Frames</span><strong>1,248</strong></div>
          <div class="metric"><span>Cuts</span><strong>7</strong></div>
          <div class="metric"><span>Edited</span><strong>82%</strong></div>
          <div class="metric"><span>Export</span><strong>Ready</strong></div>
        </div>
        <div class="inspector-list">
          <label class="inspector-row">
            <span>Dataset</span>
            <input data-field="dataset" value="${dataset}" />
          </label>
          <label class="inspector-row">
            <span>Episode</span>
            <input data-field="episode" type="number" min="0" value="${episode}" />
          </label>
          <label class="inspector-row">
            <span>Playback</span>
            <select data-field="playback">
              <option value="0.5x">0.5x</option>
              <option value="1x">1x</option>
              <option value="2x">2x</option>
              <option value="4x">4x</option>
            </select>
          </label>
        </div>
      </div>
    `;

    const playback = container.querySelector('[data-field="playback"]');
    playback.value = panel.state.playback || "1x";
    container.querySelectorAll("[data-field]").forEach((field) => {
      field.addEventListener("change", () => {
        const key = field.dataset.field;
        const value = field.type === "number" ? Number(field.value) : field.value;
        api.updateState({ [key]: value });
      });
    });
  },
});

dock.registerPanelType("timeline", {
  title: "Timeline",
  render(container) {
    const tracks = [
      ["--start: 4%; --width: 28%; --color: var(--demo-teal);"],
      ["--start: 18%; --width: 36%; --color: var(--demo-amber);"],
      ["--start: 46%; --width: 21%; --color: var(--demo-green);"],
      ["--start: 64%; --width: 27%; --color: var(--demo-red);"],
    ];
    container.innerHTML = `<div class="timeline">${tracks.map(([style]) => `<div class="track" style="${style}"></div>`).join("")}</div>`;
  },
});

dock.registerPanelType("curves", {
  title: "Curves",
  render(container) {
    const colors = ["#4db6ac", "#d39b43", "#7bc17e", "#d76d77", "#8da2f8", "#d6d0bd"];
    container.innerHTML = `
      <div class="curve-board">
        ${colors.map((color) => `<div class="curve-line" style="--line-color: ${color}"></div>`).join("")}
      </div>
    `;
  },
});

dock.registerPanelType("console", {
  title: "Console",
  render(container) {
    const rows = [
      ["12:04:21", "Loaded episode 12"],
      ["12:04:23", "Generated transition preview"],
      ["12:04:29", "Marked 3 segments for review"],
      ["12:04:33", "Layout snapshot saved"],
    ];
    container.innerHTML = `
      <div class="log-list">
        ${rows.map(([time, text]) => `<div class="log-line"><span>${time}</span><strong>${text}</strong></div>`).join("")}
      </div>
    `;
  },
});

dock.registerPanelType("default", {
  title: "Panel",
  render(container, panel) {
    container.innerHTML = `<div class="panel-pad"><div class="metric"><span>Type</span><strong>${panel.type}</strong></div></div>`;
  },
});

document.querySelectorAll("[data-add-panel]").forEach((button) => {
  button.addEventListener("click", () => {
    const type = button.dataset.addPanel;
    const spec = panelSpecs[type] || panelSpecs.console;
    counters[type] = (counters[type] || 0) + 1;
    dock.createPanel(
      {
        type,
        title: `${spec.title} ${counters[type]}`,
        state: spec.initialState(),
      },
      { targetId: ROOT_TARGET, side: "right" },
    );
  });
});

document.querySelector("#resetLayout").addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  dock.setLayout(defaultLayout);
});

function updateStatus(_layout, instance = dock) {
  panelCount.textContent = `Panels: ${instance.getPanelCount()}`;
  layoutStatus.textContent = `Saved ${new Date().toLocaleTimeString()}`;
}

updateStatus(null, dock);

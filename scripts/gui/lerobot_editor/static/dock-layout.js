export const ROOT_TARGET = "__dock_root__";

const DEFAULT_OPTIONS = {
  minPaneSize: 120,
  gutterSize: 8,
  snapSize: 72,
  storageKey: null,
  emptyMessage: "No panels",
  layout: null,
  panelTypes: {},
  onLayoutChange: null,
};

let idCounter = 0;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function cloneJson(value) {
  if (value === null || value === undefined) return value;
  return JSON.parse(JSON.stringify(value));
}

function createId(prefix = "panel") {
  if (window.crypto?.randomUUID) return `${prefix}-${window.crypto.randomUUID()}`;
  idCounter += 1;
  return `${prefix}-${Date.now().toString(36)}-${idCounter}`;
}

function isHorizontal(direction) {
  return direction === "horizontal";
}

function sideToDirection(side) {
  return side === "left" || side === "right" ? "horizontal" : "vertical";
}

function sidePlacesNewFirst(side) {
  return side === "left" || side === "top";
}

function normalizeSide(side) {
  return ["left", "right", "top", "bottom"].includes(side) ? side : "right";
}

function pointInRect(x, y, rect) {
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

function panelCount(node) {
  if (!node) return 0;
  if (node.kind === "panel") return 1;
  return panelCount(node.first) + panelCount(node.second);
}

function findPanelNode(node, panelId) {
  if (!node) return null;
  if (node.kind === "panel") return node.id === panelId ? node : null;
  return findPanelNode(node.first, panelId) || findPanelNode(node.second, panelId);
}

function makeSplit(side, newPanel, targetNode) {
  const first = sidePlacesNewFirst(side) ? newPanel : targetNode;
  const second = sidePlacesNewFirst(side) ? targetNode : newPanel;
  return {
    kind: "split",
    direction: sideToDirection(side),
    ratio: 0.5,
    first,
    second,
  };
}

function insertAtPanel(node, targetId, side, panelNode) {
  if (!node) return { node, inserted: false };
  if (node.kind === "panel") {
    if (node.id !== targetId) return { node, inserted: false };
    return { node: makeSplit(side, panelNode, node), inserted: true };
  }

  const first = insertAtPanel(node.first, targetId, side, panelNode);
  if (first.inserted) return { node: { ...node, first: first.node }, inserted: true };

  const second = insertAtPanel(node.second, targetId, side, panelNode);
  if (second.inserted) return { node: { ...node, second: second.node }, inserted: true };

  return { node, inserted: false };
}

function removePanel(node, panelId) {
  if (!node) return { node: null, removed: null };
  if (node.kind === "panel") {
    return node.id === panelId ? { node: null, removed: node } : { node, removed: null };
  }

  const first = removePanel(node.first, panelId);
  if (first.removed) {
    if (!first.node) return { node: node.second, removed: first.removed };
    return { node: { ...node, first: first.node }, removed: first.removed };
  }

  const second = removePanel(node.second, panelId);
  if (second.removed) {
    if (!second.node) return { node: node.first, removed: second.removed };
    return { node: { ...node, second: second.node }, removed: second.removed };
  }

  return { node, removed: null };
}

export class DockLayout {
  constructor(root, options = {}) {
    this.root = typeof root === "string" ? document.querySelector(root) : root;
    if (!this.root) throw new Error("DockLayout root element was not found");

    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.panelTypes = new Map(Object.entries(this.options.panelTypes || {}));
    this.layout = this.normalizeLayout(this.loadSavedLayout() || this.options.layout);
    this.drag = null;
    this.resize = null;

    this.surface = document.createElement("div");
    this.surface.className = "dock-surface";
    this.dropIndicator = document.createElement("div");
    this.dropIndicator.className = "dock-drop-indicator";

    this.root.classList.add("dock-root");
    this.root.style.setProperty("--dock-gutter-size", `${this.options.gutterSize}px`);
    this.root.innerHTML = "";
    this.root.append(this.surface, this.dropIndicator);

    this.onDocumentPointerMove = this.onDocumentPointerMove.bind(this);
    this.onDocumentPointerUp = this.onDocumentPointerUp.bind(this);
    document.addEventListener("pointermove", this.onDocumentPointerMove);
    document.addEventListener("pointerup", this.onDocumentPointerUp);
    document.addEventListener("pointercancel", this.onDocumentPointerUp);

    this.render();
  }

  destroy() {
    document.removeEventListener("pointermove", this.onDocumentPointerMove);
    document.removeEventListener("pointerup", this.onDocumentPointerUp);
    document.removeEventListener("pointercancel", this.onDocumentPointerUp);
    this.cleanupDrag();
    this.cleanupResize();
    this.root.innerHTML = "";
  }

  registerPanelType(type, definition) {
    this.panelTypes.set(type, definition);
    this.render();
  }

  setLayout(layout) {
    this.layout = this.normalizeLayout(layout);
    this.render();
    this.notifyLayoutChange();
  }

  getLayout() {
    return cloneJson(this.layout);
  }

  getPanelCount() {
    return panelCount(this.layout);
  }

  createPanel(panel, placement = {}) {
    const panelNode = this.normalizePanel(panel);
    const side = normalizeSide(placement.side);

    if (!this.layout) {
      this.layout = panelNode;
    } else if (placement.targetId && placement.targetId !== ROOT_TARGET) {
      const inserted = insertAtPanel(this.layout, placement.targetId, side, panelNode);
      this.layout = inserted.inserted ? inserted.node : makeSplit(side, panelNode, this.layout);
    } else {
      this.layout = makeSplit(side, panelNode, this.layout);
    }

    this.render();
    this.notifyLayoutChange();
    return panelNode.id;
  }

  closePanel(panelId) {
    const result = removePanel(this.layout, panelId);
    if (!result.removed) return false;
    this.layout = result.node;
    this.render();
    this.notifyLayoutChange();
    return true;
  }

  movePanel(panelId, targetId, side) {
    const normalizedSide = normalizeSide(side);
    if (!findPanelNode(this.layout, panelId)) return false;
    if (targetId !== ROOT_TARGET && panelId === targetId) return false;
    if (targetId !== ROOT_TARGET && !findPanelNode(this.layout, targetId)) return false;

    const original = this.layout;
    const result = removePanel(this.layout, panelId);
    if (!result.removed) return false;

    if (!result.node) {
      this.layout = result.removed;
    } else if (targetId === ROOT_TARGET) {
      this.layout = makeSplit(normalizedSide, result.removed, result.node);
    } else {
      const inserted = insertAtPanel(result.node, targetId, normalizedSide, result.removed);
      if (!inserted.inserted) {
        this.layout = original;
        return false;
      }
      this.layout = inserted.node;
    }

    this.render();
    this.notifyLayoutChange();
    return true;
  }

  normalizeLayout(node) {
    if (!node) return null;

    if (node.kind === "split") {
      const first = this.normalizeLayout(node.first);
      const second = this.normalizeLayout(node.second);
      if (!first) return second;
      if (!second) return first;
      const direction = node.direction === "vertical" ? "vertical" : "horizontal";
      return {
        kind: "split",
        direction,
        ratio: clamp(Number(node.ratio) || 0.5, 0.08, 0.92),
        first,
        second,
      };
    }

    return this.normalizePanel(node);
  }

  normalizePanel(panel) {
    const type = panel.type || "default";
    return {
      kind: "panel",
      id: panel.id || createId(type),
      type,
      title: panel.title || this.panelTypes.get(type)?.title || type,
      state: cloneJson(panel.state || {}),
    };
  }

  loadSavedLayout() {
    if (!this.options.storageKey) return null;
    try {
      const raw = localStorage.getItem(this.options.storageKey);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  persistLayout() {
    if (!this.options.storageKey) return;
    try {
      localStorage.setItem(this.options.storageKey, JSON.stringify(this.layout));
    } catch {
      // Layout persistence is best-effort.
    }
  }

  notifyLayoutChange() {
    this.persistLayout();
    const layout = this.getLayout();
    this.options.onLayoutChange?.(layout, this);
    this.root.dispatchEvent(new CustomEvent("dock-layout-change", { detail: { layout } }));
  }

  render() {
    this.surface.innerHTML = "";
    if (!this.layout) {
      const empty = document.createElement("div");
      empty.className = "dock-empty";
      empty.textContent = this.options.emptyMessage;
      this.surface.append(empty);
      return;
    }
    this.surface.append(this.renderNode(this.layout));
  }

  renderNode(node) {
    if (node.kind === "panel") return this.renderPanel(node);
    return this.renderSplit(node);
  }

  renderSplit(node) {
    const splitEl = document.createElement("div");
    splitEl.className = `dock-split is-${node.direction}`;
    this.applySplitStyle(splitEl, node);

    const gutter = document.createElement("div");
    gutter.className = `dock-gutter is-${isHorizontal(node.direction) ? "horizontal" : "vertical"}`;
    gutter.setAttribute("role", "separator");
    gutter.setAttribute("aria-orientation", isHorizontal(node.direction) ? "vertical" : "horizontal");
    gutter.addEventListener("pointerdown", (event) => this.beginResize(event, node, splitEl, gutter));
    gutter.addEventListener("dblclick", () => {
      node.ratio = 0.5;
      this.applySplitStyle(splitEl, node);
      this.notifyLayoutChange();
    });

    splitEl.append(this.renderNode(node.first), gutter, this.renderNode(node.second));
    return splitEl;
  }

  applySplitStyle(splitEl, node) {
    const first = clamp(node.ratio, 0.05, 0.95);
    const second = 1 - first;
    const min = `${this.options.minPaneSize}px`;
    const gutter = `${this.options.gutterSize}px`;

    if (isHorizontal(node.direction)) {
      splitEl.style.gridTemplateColumns = `minmax(${min}, ${first}fr) ${gutter} minmax(${min}, ${second}fr)`;
      splitEl.style.gridTemplateRows = "minmax(0, 1fr)";
    } else {
      splitEl.style.gridTemplateColumns = "minmax(0, 1fr)";
      splitEl.style.gridTemplateRows = `minmax(${min}, ${first}fr) ${gutter} minmax(${min}, ${second}fr)`;
    }
  }

  renderPanel(panel) {
    const panelEl = document.createElement("section");
    panelEl.className = "dock-panel";
    panelEl.dataset.dockPanelId = panel.id;
    panelEl.dataset.dockPanelType = panel.type;

    const titlebar = document.createElement("header");
    titlebar.className = "dock-titlebar";
    titlebar.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest("button")) return;
      this.beginPanelDrag(event, panel.id, panelEl);
    });

    const title = document.createElement("div");
    title.className = "dock-title";
    title.textContent = panel.title;

    const type = document.createElement("div");
    type.className = "dock-type";
    type.textContent = panel.type;

    const actions = document.createElement("div");
    actions.className = "dock-panel-actions";

    const close = document.createElement("button");
    close.className = "dock-icon-button";
    close.type = "button";
    close.title = "Close";
    close.setAttribute("aria-label", "Close panel");
    close.textContent = "x";
    close.addEventListener("click", () => this.closePanel(panel.id));
    actions.append(close);

    const content = document.createElement("div");
    content.className = "dock-panel-content";

    titlebar.append(title, type, actions);
    panelEl.append(titlebar, content);
    this.renderPanelContent(content, panel);
    return panelEl;
  }

  renderPanelContent(contentEl, panel) {
    const definition = this.panelTypes.get(panel.type) || this.panelTypes.get("default");
    const api = this.createPanelApi(panel);

    try {
      if (definition?.render) {
        definition.render(contentEl, panel, api);
      } else {
        contentEl.textContent = panel.title;
      }
    } catch (error) {
      contentEl.innerHTML = "";
      const message = document.createElement("pre");
      message.textContent = error instanceof Error ? error.message : String(error);
      contentEl.append(message);
    }
  }

  createPanelApi(panel) {
    return {
      layout: this,
      close: () => this.closePanel(panel.id),
      createPanel: (definition, placement) => this.createPanel(definition, placement),
      requestRender: () => this.render(),
      setTitle: (title) => {
        panel.title = title;
        this.render();
        this.notifyLayoutChange();
      },
      updateState: (patch, options = {}) => {
        panel.state = { ...panel.state, ...cloneJson(patch) };
        if (options.render) this.render();
        this.notifyLayoutChange();
      },
    };
  }

  beginResize(event, node, splitEl, gutter) {
    const rect = splitEl.getBoundingClientRect();
    this.resize = {
      pointerId: event.pointerId,
      node,
      splitEl,
      gutter,
      rect,
      startRatio: node.ratio,
      startX: event.clientX,
      startY: event.clientY,
    };
    gutter.classList.add("dragging");
    document.body.classList.add("dock-resizing");
    document.body.style.cursor = isHorizontal(node.direction) ? "col-resize" : "row-resize";
    event.preventDefault();
  }

  updateResize(event) {
    const resize = this.resize;
    if (!resize || event.pointerId !== resize.pointerId) return;

    const horizontal = isHorizontal(resize.node.direction);
    const size = horizontal ? resize.rect.width : resize.rect.height;
    if (size <= 0) return;

    const delta = horizontal ? event.clientX - resize.startX : event.clientY - resize.startY;
    const startPx = resize.startRatio * size;
    const minRatio = clamp(this.options.minPaneSize / size, 0.05, 0.45);
    resize.node.ratio = clamp((startPx + delta) / size, minRatio, 1 - minRatio);
    this.applySplitStyle(resize.splitEl, resize.node);
    event.preventDefault();
  }

  cleanupResize() {
    if (!this.resize) return;
    this.resize.gutter.classList.remove("dragging");
    document.body.classList.remove("dock-resizing");
    document.body.style.cursor = "";
    this.resize = null;
  }

  beginPanelDrag(event, panelId, panelEl) {
    const rect = panelEl.getBoundingClientRect();
    this.drag = {
      pointerId: event.pointerId,
      panelId,
      panelEl,
      active: false,
      startX: event.clientX,
      startY: event.clientY,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      width: rect.width,
      height: rect.height,
      candidate: null,
      ghost: null,
    };
    event.preventDefault();
  }

  activateDrag() {
    const drag = this.drag;
    if (!drag || drag.active) return;

    const ghost = drag.panelEl.cloneNode(true);
    ghost.classList.add("dock-drag-ghost");
    ghost.style.width = `${drag.width}px`;
    ghost.style.height = `${drag.height}px`;
    document.body.append(ghost);

    drag.ghost = ghost;
    drag.active = true;
    drag.panelEl.classList.add("drag-source");
    document.body.classList.add("dock-dragging");
  }

  updateDrag(event) {
    const drag = this.drag;
    if (!drag || event.pointerId !== drag.pointerId) return;

    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    if (!drag.active && Math.hypot(dx, dy) < 4) return;
    this.activateDrag();

    drag.ghost.style.left = `${event.clientX - drag.offsetX}px`;
    drag.ghost.style.top = `${event.clientY - drag.offsetY}px`;
    drag.candidate = this.findDropCandidate(event.clientX, event.clientY, drag.panelId);
    this.showDropIndicator(drag.candidate);
    event.preventDefault();
  }

  findDropCandidate(x, y, panelId) {
    const rootRect = this.root.getBoundingClientRect();
    if (!pointInRect(x, y, rootRect)) return null;

    const hovered = document.elementFromPoint(x, y)?.closest(".dock-panel[data-dock-panel-id]");
    const hoveredId = hovered?.dataset.dockPanelId;
    if (hovered && this.root.contains(hovered) && hoveredId && hoveredId !== panelId) {
      const side = this.pickSide(hovered.getBoundingClientRect(), x, y);
      if (side) return { targetId: hoveredId, side, rect: hovered.getBoundingClientRect() };
    }

    for (const panel of this.root.querySelectorAll(".dock-panel[data-dock-panel-id]")) {
      const id = panel.dataset.dockPanelId;
      const rect = panel.getBoundingClientRect();
      if (id !== panelId && pointInRect(x, y, rect)) {
        const side = this.pickSide(rect, x, y);
        if (side) return { targetId: id, side, rect };
      }
    }

    const rootSide = this.pickSide(rootRect, x, y, 0.2);
    if (rootSide && this.getPanelCount() > 1) return { targetId: ROOT_TARGET, side: rootSide, rect: rootRect };
    return null;
  }

  pickSide(rect, x, y, maxEdgeRatio = 0.35) {
    const maxEdge = Math.min(this.options.snapSize, rect.width * maxEdgeRatio, rect.height * maxEdgeRatio);
    const distances = [
      ["left", x - rect.left],
      ["right", rect.right - x],
      ["top", y - rect.top],
      ["bottom", rect.bottom - y],
    ].sort((a, b) => a[1] - b[1]);

    return distances[0][1] <= maxEdge ? distances[0][0] : null;
  }

  showDropIndicator(candidate) {
    if (!candidate) {
      this.dropIndicator.style.display = "none";
      return;
    }

    const rootRect = this.root.getBoundingClientRect();
    const rect = candidate.rect;
    const isRoot = candidate.targetId === ROOT_TARGET;
    const fraction = isRoot ? 0.28 : 0.5;
    let left = rect.left;
    let top = rect.top;
    let width = rect.width;
    let height = rect.height;

    if (candidate.side === "left") width *= fraction;
    if (candidate.side === "right") {
      width *= fraction;
      left = rect.right - width;
    }
    if (candidate.side === "top") height *= fraction;
    if (candidate.side === "bottom") {
      height *= fraction;
      top = rect.bottom - height;
    }

    this.dropIndicator.style.display = "block";
    this.dropIndicator.style.left = `${left - rootRect.left}px`;
    this.dropIndicator.style.top = `${top - rootRect.top}px`;
    this.dropIndicator.style.width = `${width}px`;
    this.dropIndicator.style.height = `${height}px`;
  }

  finishDrag(event) {
    const drag = this.drag;
    if (!drag || event.pointerId !== drag.pointerId) return;

    if (drag.active && drag.candidate) {
      this.movePanel(drag.panelId, drag.candidate.targetId, drag.candidate.side);
    }

    this.cleanupDrag();
    event.preventDefault();
  }

  cleanupDrag() {
    if (!this.drag) return;
    this.drag.panelEl?.classList.remove("drag-source");
    this.drag.ghost?.remove();
    this.dropIndicator.style.display = "none";
    document.body.classList.remove("dock-dragging");
    this.drag = null;
  }

  onDocumentPointerMove(event) {
    if (this.resize) this.updateResize(event);
    if (this.drag) this.updateDrag(event);
  }

  onDocumentPointerUp(event) {
    if (this.resize && event.pointerId === this.resize.pointerId) {
      this.notifyLayoutChange();
      this.cleanupResize();
      event.preventDefault();
    }
    if (this.drag) this.finishDrag(event);
  }
}

export default DockLayout;

#!/usr/bin/env python3
# Usage:
#   python scripts/gui/lerobot_editor/urdf_viewer.py /path/to/biped_s45.urdf
#   python scripts/gui/lerobot_editor/urdf_viewer.py /path/to/biped_s45.urdf /path/to/drake/biped_v3_arm.urdf --port 8765
#   python scripts/gui/lerobot_editor/urdf_viewer.py /path/to/biped_s45.urdf --package-root kuavo_assets=/path/to/kuavo_assets
#   python scripts/gui/lerobot_editor/urdf_viewer.py /path/to/biped_s45.urdf --ros-root /path/to/ros_ws/src --no-browser
#
# The viewer serves local URDF/STL files over HTTP and opens a Three.js page.
# Mesh references are resolved from:
#   - package://name/... via --package-root or package.xml files found under --ros-root / URDF ancestors
#   - relative mesh paths against each URDF file directory
# Xacro files should be expanded to URDF first. The browser viewer currently renders STL visual meshes.

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import posixpath
import socket
import sys
import threading
import urllib.parse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>URDF Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f7;
      --panel: #ffffff;
      --ink: #1d242b;
      --muted: #66717d;
      --line: #d8dde3;
      --warn: #b85c38;
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body {
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(170px, 1fr) auto auto auto;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 1px 5px rgba(25, 32, 38, 0.06);
      z-index: 2;
    }

    .brand {
      min-width: 0;
      font-size: 14px;
      font-weight: 700;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    select,
    button {
      height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      font: inherit;
      font-size: 13px;
    }

    select {
      min-width: 260px;
      max-width: min(52vw, 620px);
      padding: 0 32px 0 10px;
    }

    button {
      min-width: 72px;
      padding: 0 12px;
      cursor: pointer;
    }

    button:hover,
    select:hover {
      border-color: #aeb8c2;
    }

    #status {
      justify-self: end;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    #stage {
      position: relative;
      min-height: 0;
    }

    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }

    .overlay {
      position: absolute;
      left: 14px;
      bottom: 14px;
      display: grid;
      gap: 5px;
      max-width: min(560px, calc(100vw - 28px));
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(8px);
      font-size: 12px;
      line-height: 1.35;
      color: var(--muted);
      pointer-events: none;
    }

    .overlay strong {
      color: var(--ink);
    }

    .overlay .warn {
      color: var(--warn);
    }

    @media (max-width: 760px) {
      .toolbar {
        grid-template-columns: 1fr auto;
      }

      .brand,
      #status {
        display: none;
      }

      select {
        min-width: 0;
        width: 100%;
        max-width: none;
      }
    }
  </style>
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }
    }
  </script>
</head>
<body>
  <div class="toolbar">
    <div class="brand" id="brand">URDF Viewer</div>
    <select id="modelSelect" aria-label="URDF model"></select>
    <button id="fitButton" type="button">Fit</button>
    <div id="status">Idle</div>
  </div>
  <main id="stage">
    <div class="overlay" id="meta">
      <div><strong>Model</strong> <span id="modelName">-</span></div>
      <div><strong>Links</strong> <span id="linkCount">0</span> · <strong>Joints</strong> <span id="jointCount">0</span> · <strong>Meshes</strong> <span id="meshCount">0</span></div>
      <div id="warning" class="warn"></div>
    </div>
  </main>

  <script id="viewer-config" type="application/json">__CONFIG_JSON__</script>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { STLLoader } from "three/addons/loaders/STLLoader.js";

    const CONFIG = JSON.parse(document.getElementById("viewer-config").textContent);
    const MODELS = CONFIG.models;

    const stage = document.getElementById("stage");
    const brand = document.getElementById("brand");
    const modelSelect = document.getElementById("modelSelect");
    const fitButton = document.getElementById("fitButton");
    const statusEl = document.getElementById("status");
    const modelNameEl = document.getElementById("modelName");
    const linkCountEl = document.getElementById("linkCount");
    const jointCountEl = document.getElementById("jointCount");
    const meshCountEl = document.getElementById("meshCount");
    const warningEl = document.getElementById("warning");

    brand.textContent = CONFIG.title || "URDF Viewer";
    for (const model of MODELS) {
      const option = document.createElement("option");
      option.value = String(model.index);
      option.textContent = model.name;
      modelSelect.appendChild(option);
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xf5f6f7, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    stage.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(2.2, -3.1, 1.5);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0.35);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x98a0a8, 1.5));

    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(3, -4, 5);
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xffffff, 0.75);
    fill.position.set(-3, 2, 2);
    scene.add(fill);

    const grid = new THREE.GridHelper(2.4, 24, 0xb9c2cb, 0xdce1e6);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(0.35));

    const stlLoader = new STLLoader();
    let robotRoot = null;
    let loadToken = 0;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function parseVec(value, fallback = [0, 0, 0]) {
      if (!value) return fallback.slice();
      const items = value.trim().split(/\s+/).map(Number);
      return items.length >= 3 && items.every(Number.isFinite) ? items.slice(0, 3) : fallback.slice();
    }

    function applyOrigin(object, originEl) {
      const xyz = parseVec(originEl?.getAttribute("xyz"));
      const rpy = parseVec(originEl?.getAttribute("rpy"));
      object.position.set(xyz[0], xyz[1], xyz[2]);
      object.rotation.set(rpy[0], rpy[1], rpy[2], "XYZ");
    }

    function materialFromVisual(visualEl) {
      const colorEl = visualEl.querySelector("material color");
      const rgba = parseVec(colorEl?.getAttribute("rgba"), [0.86, 0.88, 0.9]);
      return new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        opacity: rgba[3] ?? 1,
        transparent: (rgba[3] ?? 1) < 1,
        roughness: 0.58,
        metalness: 0.08,
      });
    }

    function resolveMeshUrl(filename, model) {
      if (filename.startsWith("package://")) {
        const rest = filename.slice("package://".length);
        const slash = rest.indexOf("/");
        if (slash < 0) return null;
        const packageName = rest.slice(0, slash);
        const packageRest = rest.slice(slash + 1);
        return `/asset/package/${encodeURIComponent(packageName)}?path=${encodeURIComponent(packageRest)}`;
      }
      return `/asset/relative/${model.index}?path=${encodeURIComponent(filename)}`;
    }

    function addVisuals(linkGroup, linkEl, model, loadState) {
      const visuals = [...linkEl.querySelectorAll(":scope > visual")];
      for (const visualEl of visuals) {
        const meshEl = visualEl.querySelector("geometry mesh");
        const filename = meshEl?.getAttribute("filename");
        if (!filename) continue;

        const url = resolveMeshUrl(filename, model);
        if (!url) {
          loadState.warnings.push(`Could not resolve mesh: ${filename}`);
          continue;
        }
        const ext = filename.split(".").pop().toLowerCase();
        const visualGroup = new THREE.Group();
        applyOrigin(visualGroup, visualEl.querySelector(":scope > origin"));
        linkGroup.add(visualGroup);
        loadState.meshes += 1;

        if (ext !== "stl") {
          loadState.warnings.push(`Skipped ${ext.toUpperCase()} mesh: ${filename}`);
          continue;
        }

        loadState.pending += 1;
        stlLoader.load(
          url,
          (geometry) => {
            if (loadToken !== loadState.token) return;
            geometry.computeVertexNormals();
            const scale = parseVec(meshEl.getAttribute("scale"), [1, 1, 1]);
            const mesh = new THREE.Mesh(geometry, materialFromVisual(visualEl));
            mesh.scale.set(scale[0], scale[1], scale[2]);
            visualGroup.add(mesh);
            markMeshDone(loadState);
          },
          undefined,
          () => {
            loadState.warnings.push(`Missing mesh: ${filename}`);
            markMeshDone(loadState);
          },
        );
      }
    }

    function markMeshDone(loadState) {
      loadState.pending -= 1;
      const loaded = loadState.meshes - loadState.pending;
      setStatus(`Meshes ${loaded}/${loadState.meshes}`);
      if (loadState.pending === 0) {
        warningEl.textContent = loadState.warnings.slice(0, 3).join("  ");
        fitCamera();
        setStatus(loadState.warnings.length ? `Ready, ${loadState.warnings.length} warning(s)` : "Ready");
      }
    }

    function buildRobot(doc, model, token) {
      const links = new Map();
      const joints = [];
      const childLinks = new Set();
      const loadState = { token, pending: 0, meshes: 0, warnings: [] };
      const root = new THREE.Group();

      for (const linkEl of doc.querySelectorAll("robot > link")) {
        const name = linkEl.getAttribute("name");
        const group = new THREE.Group();
        group.name = name;
        links.set(name, group);
        addVisuals(group, linkEl, model, loadState);
      }

      for (const jointEl of doc.querySelectorAll("robot > joint")) {
        const parent = jointEl.querySelector("parent")?.getAttribute("link");
        const child = jointEl.querySelector("child")?.getAttribute("link");
        if (!parent || !child || !links.has(parent) || !links.has(child)) continue;
        joints.push({ jointEl, parent, child });
        childLinks.add(child);
      }

      for (const { jointEl, parent, child } of joints) {
        const jointGroup = new THREE.Group();
        jointGroup.name = jointEl.getAttribute("name") || "";
        applyOrigin(jointGroup, jointEl.querySelector(":scope > origin"));
        jointGroup.add(links.get(child));
        links.get(parent).add(jointGroup);
      }

      for (const [name, group] of links) {
        if (!childLinks.has(name)) root.add(group);
      }

      linkCountEl.textContent = String(links.size);
      jointCountEl.textContent = String(joints.length);
      meshCountEl.textContent = String(loadState.meshes);
      if (loadState.meshes === 0) {
        loadState.warnings.push("No visual STL meshes found.");
      }
      return { root, loadState };
    }

    async function loadModel(model) {
      loadToken += 1;
      const token = loadToken;
      if (robotRoot) {
        scene.remove(robotRoot);
        robotRoot.traverse((obj) => {
          obj.geometry?.dispose?.();
          obj.material?.dispose?.();
        });
      }
      robotRoot = null;
      warningEl.textContent = "";
      modelNameEl.textContent = model.name;
      linkCountEl.textContent = "0";
      jointCountEl.textContent = "0";
      meshCountEl.textContent = "0";
      setStatus("Loading URDF");

      const response = await fetch(`/urdf/${model.index}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const xml = await response.text();
      const doc = new DOMParser().parseFromString(xml, "application/xml");
      const parseError = doc.querySelector("parsererror");
      if (parseError) throw new Error("URDF XML parse error");

      const built = buildRobot(doc, model, token);
      if (token !== loadToken) return;
      robotRoot = built.root;
      scene.add(robotRoot);
      setStatus(`Meshes 0/${built.loadState.meshes}`);
      if (built.loadState.pending === 0) {
        warningEl.textContent = built.loadState.warnings.slice(0, 3).join("  ");
        fitCamera();
        setStatus(built.loadState.warnings.length ? `Ready, ${built.loadState.warnings.length} warning(s)` : "Ready");
      }
    }

    function fitCamera() {
      if (!robotRoot) return;
      const box = new THREE.Box3().setFromObject(robotRoot);
      if (box.isEmpty()) return;
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      const maxDim = Math.max(size.x, size.y, size.z, 0.2);
      const fov = THREE.MathUtils.degToRad(camera.fov);
      const distance = maxDim / (2 * Math.tan(fov / 2)) * 1.45;
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(distance * 0.85, -distance * 1.25, distance * 0.68));
      camera.near = Math.max(distance / 100, 0.01);
      camera.far = Math.max(distance * 20, 20);
      camera.updateProjectionMatrix();
      controls.update();
    }

    function resize() {
      const rect = stage.getBoundingClientRect();
      renderer.setSize(rect.width, rect.height, false);
      camera.aspect = rect.width / Math.max(rect.height, 1);
      camera.updateProjectionMatrix();
    }

    function animate() {
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }

    modelSelect.addEventListener("change", () => {
      const model = MODELS.find((item) => String(item.index) === modelSelect.value) ?? MODELS[0];
      const url = new URL(window.location.href);
      url.searchParams.set("model", String(model.index));
      history.replaceState(null, "", url);
      loadModel(model).catch((error) => {
        setStatus("Error");
        warningEl.textContent = String(error.message || error);
      });
    });

    fitButton.addEventListener("click", fitCamera);
    window.addEventListener("resize", resize);

    const requested = new URLSearchParams(window.location.search).get("model");
    const fallback = MODELS[0]?.index ?? 0;
    modelSelect.value = MODELS.some((item) => String(item.index) === requested) ? requested : String(fallback);
    resize();
    animate();
    modelSelect.dispatchEvent(new Event("change"));
  </script>
</body>
</html>
"""


class ViewerState:
    def __init__(
        self,
        urdf_paths: list[Path],
        package_roots: dict[str, Path],
        relative_roots: list[Path],
        title: str,
    ) -> None:
        self.urdf_paths = urdf_paths
        self.package_roots = package_roots
        self.relative_roots = relative_roots
        self.title = title


class ViewerHandler(SimpleHTTPRequestHandler):
    server_version = "UrdfViewerHTTP/1.0"

    def __init__(self, *args: Any, state: ViewerState, **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[urdf-viewer] {self.address_string()} - {fmt % args}\n")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = posixpath.normpath(urllib.parse.unquote(parsed.path))
        if path in {"/", "/index.html"}:
            self._serve_index()
            return
        if path.startswith("/urdf/"):
            self._serve_urdf(path)
            return
        if path.startswith("/asset/package/"):
            self._serve_package_asset(path, parsed.query)
            return
        if path.startswith("/asset/relative/"):
            self._serve_relative_asset(path, parsed.query)
            return
        if path == "/favicon.ico":
            self.send_error(404, "not found")
            return
        self.send_error(404, "not found")

    def _serve_index(self) -> None:
        models = [
            {
                "index": idx,
                "name": path.name if len(self.state.urdf_paths) == 1 else str(path),
            }
            for idx, path in enumerate(self.state.urdf_paths)
        ]
        config = {
            "title": self.state.title,
            "models": models,
            "packages": {name: str(path) for name, path in self.state.package_roots.items()},
        }
        payload = json.dumps(config, ensure_ascii=True)
        body = VIEWER_HTML.replace("__CONFIG_JSON__", html.escape(payload, quote=False)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_urdf(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) != 3 or not parts[2].isdigit():
            self.send_error(404, "bad URDF id")
            return
        idx = int(parts[2])
        if idx < 0 or idx >= len(self.state.urdf_paths):
            self.send_error(404, "unknown URDF id")
            return
        self._serve_file(self.state.urdf_paths[idx])

    def _serve_package_asset(self, path: str, query: str) -> None:
        rest = path.removeprefix("/asset/package/")
        package_name = rest.strip("/")
        asset_path = query_path(query)
        if not package_name or asset_path is None:
            self.send_error(404, "bad package asset path")
            return
        package_root = self.state.package_roots.get(package_name)
        if package_root is None:
            self.send_error(404, f"unknown package: {package_name}")
            return
        target = safe_join(package_root, package_root, asset_path)
        if target is None:
            self.send_error(403, "invalid package asset path")
            return
        self._serve_file(target)

    def _serve_relative_asset(self, path: str, query: str) -> None:
        rest = path.removeprefix("/asset/relative/")
        asset_path = query_path(query)
        if not rest.isdigit() or asset_path is None:
            self.send_error(404, "bad relative asset path")
            return
        idx = int(rest)
        if idx < 0 or idx >= len(self.state.urdf_paths):
            self.send_error(404, "unknown URDF id")
            return
        target = safe_join(self.state.urdf_paths[idx].parent, self.state.relative_roots[idx], asset_path)
        if target is None:
            self.send_error(403, "invalid relative asset path")
            return
        self._serve_file(target)

    def _serve_file(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            self.send_error(404, f"not found: {resolved}")
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        try:
            with resolved.open("rb") as file_obj:
                data = file_obj.read()
        except OSError as exc:
            self.send_error(500, str(exc))
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def query_path(query: str) -> str | None:
    values = urllib.parse.parse_qs(query).get("path")
    if not values:
        return None
    return values[0]


def safe_join(base: Path, allowed_root: Path, rest: str) -> Path | None:
    decoded = urllib.parse.unquote(rest)
    decoded = decoded.replace("\\", "/")
    target = (base / decoded).resolve()
    try:
        target.relative_to(allowed_root.resolve())
    except ValueError:
        return None
    return target


def parse_package_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("package name is empty")
    package_path = Path(path).expanduser().resolve()
    if not package_path.is_dir():
        raise argparse.ArgumentTypeError(f"package path does not exist: {package_path}")
    return name, package_path


def find_free_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def discover_ros_roots(urdf_paths: list[Path], explicit_roots: list[Path]) -> list[Path]:
    roots: list[Path] = [root.expanduser().resolve() for root in explicit_roots]
    for urdf_path in urdf_paths:
        for parent in [urdf_path.parent, *urdf_path.parents]:
            if parent.name == "src":
                roots.append(parent.resolve())
                break
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        unique.append(root)
    return unique


def discover_package_roots(roots: list[Path]) -> dict[str, Path]:
    packages: dict[str, Path] = {}
    for root in roots:
        for package_xml in root.rglob("package.xml"):
            package_dir = package_xml.parent.resolve()
            packages.setdefault(package_dir.name, package_dir)
    return packages


def relative_root_for_urdf(urdf_path: Path, package_roots: dict[str, Path]) -> Path:
    resolved = urdf_path.resolve()
    containing = [
        package_root.resolve()
        for package_root in package_roots.values()
        if is_relative_to(resolved, package_root.resolve())
    ]
    if containing:
        return max(containing, key=lambda item: len(item.parts))
    return resolved.parent


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Three.js URDF/STL viewer.")
    parser.add_argument("urdf", nargs="+", help="URDF file(s) to visualize")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8765, help="preferred HTTP port; falls back to a free port")
    parser.add_argument("--title", default="URDF Viewer", help="title shown in the toolbar")
    parser.add_argument(
        "--package-root",
        action="append",
        default=[],
        type=parse_package_root,
        metavar="NAME=PATH",
        help="map package://NAME/... to PATH; can be repeated",
    )
    parser.add_argument(
        "--ros-root",
        action="append",
        default=[],
        type=Path,
        help="ROS workspace src directory to scan for package.xml; can be repeated",
    )
    parser.add_argument("--no-browser", action="store_true", help="serve only; do not open a browser tab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urdf_paths = [Path(item).expanduser().resolve() for item in args.urdf]
    missing = [path for path in urdf_paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"URDF not found: {path}", file=sys.stderr)
        raise SystemExit(2)

    ros_roots = discover_ros_roots(urdf_paths, args.ros_root)
    package_roots = discover_package_roots(ros_roots)
    package_roots.update(dict(args.package_root))

    port = find_free_port(args.host, args.port)
    relative_roots = [relative_root_for_urdf(path, package_roots) for path in urdf_paths]
    state = ViewerState(
        urdf_paths=urdf_paths,
        package_roots=package_roots,
        relative_roots=relative_roots,
        title=args.title,
    )
    handler = partial(ViewerHandler, state=state)
    server = ThreadingHTTPServer((args.host, port), handler)
    url = f"http://{args.host}:{port}/"

    print(f"Serving URDF viewer at {url}")
    print("URDFs:")
    for idx, path in enumerate(urdf_paths):
        print(f"  [{idx}] {path}")
    if package_roots:
        print("package:// mappings:")
        for name, path in sorted(package_roots.items()):
            print(f"  {name} -> {path}")
    else:
        print("No package:// mappings found. Use --package-root NAME=PATH if meshes are missing.")

    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping URDF viewer.")


if __name__ == "__main__":
    main()

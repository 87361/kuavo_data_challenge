export async function api(path, options = {}) {
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
    window.dispatchEvent(
      new CustomEvent("lerobot-log", {
        detail: { level: "error", source: "api", message: `${path}: ${detail}` },
      }),
    );
    throw new Error(detail);
  }
  return response.json();
}

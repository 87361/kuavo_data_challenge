import { STORAGE, clamp, normalizeFractions, storedJson } from "./utils.js";

export function createVideoController(ctx) {
  const { state, els, setStatus } = ctx;

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

  function applyPlaybackRate() {
    const rate = clamp(Number(state.playbackRate) || 1, 0.5, 4);
    state.playbackRate = rate;
    for (const video of playableVideoElements()) {
      video.playbackRate = rate;
    }
    if (els.playbackRate) els.playbackRate.value = String(rate);
  }

  function renderFrameFallbacks() {
    for (const image of els.videoGrid.querySelectorAll("img.fallback-frame")) {
      const tile = image.closest(".camera");
      if (!tile?.classList.contains("video-failed")) continue;
      image.src = frameUrl(image.dataset.key, state.currentFrame);
    }
  }

  function syncVideosToFrame(frame, force = false) {
    const target = ctx.frames.frameToTime(frame);
    const tolerance = 0.5 / ctx.frames.fps();
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
      video.playbackRate = state.playbackRate;

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
    applyPlaybackRate();
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

    const nextFrame = ctx.frames.timeToFrame(leader.currentTime);
    if (nextFrame !== state.currentFrame) {
      ctx.frames.setCurrentFrame(nextFrame, { seek: false });
    }
    syncFollowerVideos(leader);

    if (leader.ended || state.currentFrame >= state.episode.length - 1) {
      ctx.frames.setCurrentFrame(state.episode.length - 1, { seek: false });
      stopPlayback();
      return;
    }
    state.playbackRaf = requestAnimationFrame(updatePlaybackFromVideo);
  }

  function startFrameFallbackPlayback() {
    const intervalMs = 1000 / (ctx.frames.fps() * Math.max(0.1, Number(state.playbackRate) || 1));
    state.timer = setInterval(() => {
      if (!state.episode) return;
      if (state.currentFrame >= state.episode.length - 1) {
        stopPlayback();
        return;
      }
      ctx.frames.setCurrentFrame(state.currentFrame + 1, { seek: false });
    }, intervalMs);
  }

  async function startNativePlayback() {
    syncVideosToFrame(state.currentFrame, true);
    const videos = playableVideoElements();
    if (!videos.length) {
      startFrameFallbackPlayback();
      return;
    }
    applyPlaybackRate();
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
      ctx.frames.setCurrentFrame(0, { seek: true, forceTimeline: true });
    }
    state.playing = true;
    els.playPause.textContent = "Pause";
    startNativePlayback().catch((error) => {
      stopPlayback();
      setStatus(error.message);
    });
  }

  function setPlaybackRate(rate) {
    state.playbackRate = clamp(Number(rate) || 1, 0.5, 4);
    localStorage.setItem(STORAGE.playbackRate, String(state.playbackRate));
    applyPlaybackRate();
    if (state.playing && state.timer) {
      clearInterval(state.timer);
      state.timer = null;
      startFrameFallbackPlayback();
    }
  }

  return {
    playableVideoElements,
    renderFrameFallbacks,
    renderVideoGrid,
    setPlaybackRate,
    syncVideosToFrame,
    stopPlayback,
    playPause,
  };
}

from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from kuavo_deploy.config import ConfigInference
from kuavo_deploy.openpi.client import OpenPIWebsocketClient
from kuavo_deploy.utils.logging_utils import setup_logger

log_model = setup_logger("model")


class IdentityProcessor:
    def __call__(self, value):
        return value


class OpenPIChunkPolicyWrapper:
    """Expose OpenPI's official action-chunk semantics through select_action()."""

    def __init__(self, cfg: ConfigInference, repo_root: str | Path | None = None) -> None:
        self.config = cfg
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        self._process: subprocess.Popen | None = None
        self._client: OpenPIWebsocketClient | None = None
        self._actions: np.ndarray | None = None
        self._chunk_index = 0

        if cfg.openpi_auto_start:
            self._start_server()

        self._client = OpenPIWebsocketClient(
            cfg.openpi_server_host,
            cfg.openpi_server_port,
            timeout_s=cfg.openpi_connect_timeout_s,
        )
        log_model.info(f"OpenPI server metadata: {self._client.metadata}")

    def eval(self):
        return self

    def to(self, device):
        return self

    def reset(self):
        self._actions = None
        self._chunk_index = 0

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:
        if self._client is None:
            raise RuntimeError("OpenPI client is not initialized")

        if self._actions is None or self._chunk_index >= len(self._actions):
            request = self._build_openpi_observation(observation)
            result = self._client.infer(request)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(f"Expected OpenPI actions with shape [T, D], got {actions.shape}")
            if actions.shape[0] != self.config.openpi_action_horizon:
                log_model.warning(
                    f"OpenPI action horizon is {actions.shape[0]}, "
                    f"expected {self.config.openpi_action_horizon}"
                )
            self._actions = actions
            self._chunk_index = 0
            log_model.info(f"Fetched OpenPI action chunk: shape={actions.shape}")

        action = self._actions[self._chunk_index]
        log_model.debug(f"Using OpenPI action chunk index {self._chunk_index}/{len(self._actions)}")
        self._chunk_index += 1
        return torch.from_numpy(action).float().unsqueeze(0)

    def _start_server(self) -> None:
        python_bin = self._resolve_openpi_python()
        checkpoint_path = self._resolve_path(self.config.checkpoint_path)
        log_dir = self.repo_root / "log" / "kuavo_deploy"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "openpi_server.log"

        cmd = [
            python_bin,
            "-m",
            "kuavo_deploy.openpi.serve_openpi_policy",
            "--config-name",
            self.config.openpi_config_name,
            "--checkpoint-path",
            str(checkpoint_path),
            "--prompt",
            self.config.openpi_prompt,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.config.openpi_server_port),
            "--warmup-steps",
            str(self.config.openpi_warmup_steps),
            "--state-dim",
            str(self.config.openpi_state_dim),
            "--image-width",
            str(self.config.openpi_image_width),
            "--image-height",
            str(self.config.openpi_image_height),
        ]
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.repo_root) if not pythonpath else f"{self.repo_root}:{pythonpath}"
        if self.config.openpi_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.config.openpi_cuda_visible_devices

        log_model.info(f"Starting OpenPI server: {' '.join(cmd)}")
        log_file = log_path.open("a", buffering=1)
        self._process = subprocess.Popen(cmd, cwd=self.repo_root, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        atexit.register(self.close)

    def _resolve_openpi_python(self) -> str:
        candidates = [
            self.config.openpi_python,
            os.environ.get("OPENPI_PYTHON", ""),
            "/opt/conda/envs/openpi/bin/python",
            "/root/kuavo_data_challenge/third_party/openpi/.venv/bin/python",
            "/root/kuavo_data_challenge/openpi/.venv/bin/python",
            "/data/vepfs/users/intern/lingyue.yang/openpi/.venv/bin/python",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        found = shutil.which("python3") or shutil.which("python") or sys.executable
        log_model.warning(f"OpenPI python was not configured; falling back to {found}")
        return found

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        return candidate.resolve()

    def _build_openpi_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = self._to_numpy(observation["observation.state"])
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.shape[-1] != self.config.openpi_state_dim:
            raise ValueError(f"OpenPI state dim must be {self.config.openpi_state_dim}, got {state.shape}")

        return {
            "state": state.astype(np.float32),
            "images": {
                "cam_high": self._image_to_numpy(observation["observation.images.head_cam_h"]),
                "cam_left_wrist": self._image_to_numpy(observation["observation.images.wrist_cam_l"]),
                "cam_right_wrist": self._image_to_numpy(observation["observation.images.wrist_cam_r"]),
            },
            "prompt": self.config.openpi_prompt,
        }

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _image_to_numpy(self, value: Any) -> np.ndarray:
        image = self._to_numpy(value)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected image with shape [C,H,W] or [H,W,C], got {image.shape}")
        return image

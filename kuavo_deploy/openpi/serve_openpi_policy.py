from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a trained OpenPI policy for Kuavo deployment.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--image-width", type=int, default=848)
    parser.add_argument("--image-height", type=int, default=480)
    return parser.parse_args()


def make_dummy_obs(args: argparse.Namespace) -> dict:
    image = np.zeros((3, args.image_height, args.image_width), dtype=np.uint8)
    return {
        "state": np.zeros((args.state_dim,), dtype=np.float32),
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": args.prompt,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    checkpoint_path = Path(args.checkpoint_path).resolve()
    logging.info("Loading OpenPI config=%s checkpoint=%s", args.config_name, checkpoint_path)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config_name),
        checkpoint_path,
        default_prompt=args.prompt,
    )

    dummy_obs = make_dummy_obs(args)
    for i in range(max(args.warmup_steps, 0)):
        start = time.monotonic()
        result = policy.infer(dummy_obs)
        logging.info(
            "Warmup %d/%d: actions=%s %.3fs",
            i + 1,
            args.warmup_steps,
            np.asarray(result["actions"]).shape,
            time.monotonic() - start,
        )

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "config_name": args.config_name,
            "checkpoint_path": str(checkpoint_path),
            "prompt": args.prompt,
        }
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logging.info("Serving OpenPI policy on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()

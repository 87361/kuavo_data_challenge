#!/usr/bin/env bash
# Sync selected repository files from the laptop/workstation to the cloud dev box.
#
# Defaults match the current machine topology:
#   laptop -> ssh pi1022 -> /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge
#
# Examples:
#   bash scripts/dev/sync_to_cloud.sh scripts/gui_phone/routes_auto.py
#   bash scripts/dev/sync_to_cloud.sh --changed
#   CLOUD_HOST=pi1022 CLOUD_REPO=/data/.../kuavo_data_challenge bash scripts/dev/sync_to_cloud.sh --changed

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLOUD_HOST="${CLOUD_HOST:-pi1022}"
CLOUD_REPO="${CLOUD_REPO:-/data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge}"

usage() {
  cat <<'EOF'
Usage:
  scripts/dev/sync_to_cloud.sh [--changed] [--dry-run] [FILE ...]

Options:
  --changed  Sync modified/untracked files reported by git status --short.
  --dry-run  Print the rsync plan without writing remote files.

Environment:
  CLOUD_HOST  SSH host alias, default: pi1022
  CLOUD_REPO  Remote repo path, default: /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge
EOF
}

mode="explicit"
dry_run=0
files=()

while (($#)); do
  case "$1" in
    --changed)
      mode="changed"
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        files+=("$1")
        shift
      done
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      files+=("$1")
      shift
      ;;
  esac
done

cd "$ROOT"

if [[ "$mode" == "changed" ]]; then
  mapfile -t files < <(
    git status --short --untracked-files=all |
      awk '
        /^ D / || /^D  / { next }
        {
          path = substr($0, 4)
          if ($0 ~ /^R/ || $0 ~ /^C/) {
            n = split(path, parts, " -> ")
            path = parts[n]
          }
          print path
        }
      '
  )
fi

if ((${#files[@]} == 0)); then
  echo "no files selected; pass FILE ... or --changed" >&2
  exit 1
fi

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

for f in "${files[@]}"; do
  if [[ -e "$f" ]]; then
    printf '%s\n' "$f" >> "$tmp_list"
  else
    echo "skip missing/deleted path: $f" >&2
  fi
done

if [[ ! -s "$tmp_list" ]]; then
  echo "selected files are all missing/deleted; nothing to sync" >&2
  exit 1
fi

rsync_args=(
  -avh
  --relative
  --files-from="$tmp_list"
  --info=progress2
)

if ((dry_run)); then
  rsync_args+=(--dry-run)
fi

echo "[sync_to_cloud] host: $CLOUD_HOST"
echo "[sync_to_cloud] repo: $CLOUD_REPO"
echo "[sync_to_cloud] files:"
sed 's/^/  - /' "$tmp_list"

rsync "${rsync_args[@]}" ./ "${CLOUD_HOST}:${CLOUD_REPO}/"

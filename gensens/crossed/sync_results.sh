#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# sync_results.sh — continuously mirror the crossed-run results+logs from
# the GPU box down to this Mac, so an auto-pause / crash / disconnect never
# loses more than one sync interval of work.
#
# Why continuous (not a single "download after 1h"): a Jarvis auto-pause can
# wipe the box mid-run. rsync is incremental (transfers only what changed), so
# pulling every few minutes is cheap and means the Mac is always at most
# INTERVAL seconds behind whatever the box has generated.
#
# USAGE:
#   bash sync_results.sh user@host [interval_sec] [duration_min] [remote_dir]
#
# EXAMPLE:
#   bash sync_results.sh root@217.18.55.106 300 120
#     -> every 5 min for up to 2h, mirror the box's results/ + logs/ here.
#
# It stops early when the run reports completion (summary_report.md appears
# or "Done. See" shows up in the run log), after one final sync.
# ═══════════════════════════════════════════════════════════════════

set -uo pipefail

HOST="${1:?usage: bash sync_results.sh user@host [interval_sec] [duration_min] [remote_dir]}"
INTERVAL="${2:-300}"          # seconds between syncs (default 5 min)
DURATION_MIN="${3:-120}"      # give up after this many minutes (default 2h)
REMOTE_DIR="${4:-/home/prompt_sensitivity/gensens/crossed}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DEST="$SCRIPT_DIR/results_run30_live"   # local mirror target
mkdir -p "$LOCAL_DEST/results" "$LOCAL_DEST/logs"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15)
RSYNC=(rsync -az --partial --timeout=60 -e "ssh ${SSH_OPTS[*]}")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Mirroring $HOST:$REMOTE_DIR  ->  $LOCAL_DEST  (every ${INTERVAL}s, up to ${DURATION_MIN}min)"

deadline=$(( $(date +%s) + DURATION_MIN * 60 ))
sync_num=0

do_sync() {
    # Pull results/ and logs/. Failures (box paused/unreachable) are non-fatal:
    # we just log and retry next interval — the box may come back after resume.
    "${RSYNC[@]}" "$HOST:$REMOTE_DIR/results/" "$LOCAL_DEST/results/" 2>/dev/null
    local rc_r=$?
    "${RSYNC[@]}" "$HOST:$REMOTE_DIR/logs/" "$LOCAL_DEST/logs/" 2>/dev/null
    local rc_l=$?
    if [[ $rc_r -ne 0 && $rc_l -ne 0 ]]; then
        log "  sync #$sync_num: box unreachable (paused/offline?) — will retry"
        return 1
    fi
    local resp cells
    resp=$(wc -l < "$LOCAL_DEST/results/responses_crossed.jsonl" 2>/dev/null || echo 0)
    cells=$(wc -l < "$LOCAL_DEST/results/cell_metrics.jsonl" 2>/dev/null || echo 0)
    log "  sync #$sync_num OK — responses=$resp cells=$cells  (mirrored to results_run30_live/)"
    return 0
}

is_done() {
    # Done when the final report exists locally, or the run log says so.
    [[ -f "$LOCAL_DEST/results/summary_report.md" ]] && return 0
    grep -q "Done. See" "$LOCAL_DEST/logs/run30.log" 2>/dev/null && return 0
    return 1
}

while (( $(date +%s) < deadline )); do
    sync_num=$(( sync_num + 1 ))
    do_sync
    if is_done; then
        log "Run complete — doing one final sync and stopping."
        do_sync
        log "DONE. All results mirrored to: $LOCAL_DEST"
        exit 0
    fi
    sleep "$INTERVAL"
done

log "Reached ${DURATION_MIN}min duration limit. Final state mirrored to: $LOCAL_DEST"
log "(Run may still be going on the box; re-launch this script to keep mirroring.)"

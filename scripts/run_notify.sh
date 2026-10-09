#!/bin/bash
# run_notify.sh -- run a long job with desktop notification on completion.
# Usage: scripts/run_notify.sh LABEL LOGFILE cmd [args...]
# Runs cmd (output tee'd to LOGFILE), then notify-sends rc + duration +
# log tail, and appends one line to /tmp/opencode/runs.log (ledger).
# No more watching htop: start it, leave, get pinged.
LABEL="$1"; LOG="$2"; shift 2
if [ -z "$LABEL" ] || [ -z "$LOG" ] || [ $# -eq 0 ]; then
    echo "usage: $0 LABEL LOGFILE cmd [args...]" >&2
    exit 2
fi
export DISPLAY="${DISPLAY:-:0}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
T0=$(date +%s)
echo "=== [$LABEL] started $(date -Is) ===" | tee "$LOG"
if git -C "$(dirname "$0")/.." rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "--- code: $(git -C "$(dirname "$0")/.." rev-parse --short HEAD) dirty: $(git -C "$(dirname "$0")/.." status --short | tr '\n' ';')" | tee -a "$LOG"
fi
"$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s); DT=$((T1 - T0))
DH=$((DT / 3600)); DM=$(((DT % 3600) / 60)); DS=$((DT % 60))
DUR=$(printf "%dh%02dm%02ds" "$DH" "$DM" "$DS")
TAIL=$(tail -n 3 "$LOG" | tr '\n' '|' | cut -c1-220)
mkdir -p /tmp/opencode
echo "$(date -Is) label=$LABEL rc=$RC dur=$DUR" >> /tmp/opencode/runs.log
if [ "$RC" -eq 0 ]; then
    notify-send -u normal "done: $LABEL" "rc=0 in $DUR :: $TAIL" 2>/dev/null
else
    notify-send -u critical "FAILED: $LABEL" "rc=$RC in $DUR :: $TAIL" 2>/dev/null
fi
printf '\a'
echo "=== [$LABEL] rc=$RC in $DUR ===" | tee -a "$LOG"
exit "$RC"

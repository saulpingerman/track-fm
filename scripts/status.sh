#!/usr/bin/env bash
# One-shot status of everything TrackFM-related on this machine.
# Usage: scripts/status.sh        (or: watch -n 30 scripts/status.sh)

echo "════════ TRACKFM STATUS $(date -u '+%H:%M UTC') ════════"

echo "─── processes"
pgrep -af "run_pipeline|materialize_campaign|trackfm (materialize|pretrain|port-data|finetune)" \
  | sed 's/^[0-9]* /  /' | cut -c1-90 || echo "  (none)"

echo "─── materialization campaign"
if [ -f ~/data/trackfm/materialize_campaign.log ]; then
  grep "===" ~/data/trackfm/materialize_campaign.log | tail -2 | sed 's/^/  /'
  tail -c 200 ~/data/trackfm/materialize_campaign.log | tr '\r' '\n' | tail -1 | sed 's/^/  /' | cut -c1-90
fi
for d in golden69 v1; do
  n=$(ls ~/data/trackfm/materialized/$d/train/samples_*.parquet 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "  $d: $n train shards on disk"
done

echo "─── GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo "─── disk"
df -h / | tail -1 | awk '{print "  used "$3" of "$2"  ("$5")  free: "$4}'

echo "─── recent MLflow runs (http://localhost:5000 — 5001 on your Mac)"
python3 -c "
import sqlite3, datetime, os
db = os.path.expanduser('~/data/trackfm/mlflow/mlflow.db')
for name, status, t in sqlite3.connect(db).execute(
        'SELECT name, status, start_time FROM runs ORDER BY start_time DESC LIMIT 4'):
    print(f'  {name:28s} {status:10s} {datetime.datetime.utcfromtimestamp(t/1000):%m-%d %H:%M}')
" 2>/dev/null || echo "  (mlflow db unreadable — check the UI)"

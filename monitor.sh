#!/bin/bash
source /home/xitongzhang/miniconda3/etc/profile.d/conda.sh
conda activate base
while true; do
    echo "=== $(date '+%H:%M:%S') ===" >> /home/xitongzhang/Maylie/monitor.log
    python3 -c "
import json, pathlib
r = json.load(open('/home/xitongzhang/Maylie/reddit_data.json')) if pathlib.Path('/home/xitongzhang/Maylie/reddit_data.json').exists() else []
t = json.load(open('/home/xitongzhang/Maylie/twitter_data.json')) if pathlib.Path('/home/xitongzhang/Maylie/twitter_data.json').exists() else []
print(f'Reddit={len(r)}/3000 Twitter={len(t)}/3000')
" >> /home/xitongzhang/Maylie/monitor.log 2>&1
    sleep 900
done

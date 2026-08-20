#!/bin/bash
# Monitor LST download, run robustness analysis when done.
# Usage: nohup bash auto_lst_analysis.sh > /data/lab/csee_dml/auto_lst.log 2>&1 &

set -e

LST_PID=35305
LST_LOG="/data/lab/csee_dml/lst_download.log"
ROBUSTNESS_SCRIPT="/data/lab/csee_dml/lst_robustness.py"
RESULT_LOG="/data/lab/csee_dml/lst_robustness.log"

echo "========================================" 
echo "Auto LST Robustness Monitor"
echo "Waiting for LST download (PID $LST_PID) to complete..."
echo "Started at: $(date)"
echo "========================================"

# Wait for LST download process to finish
while kill -0 $LST_PID 2>/dev/null; do
    sleep 30
done

echo ""
echo "LST download process (PID $LST_PID) has exited."
echo "Time: $(date)"
echo ""

# Verify the LST file exists and is valid
LST_FILE="/data/lab/csee_dml/data/lst_panel.pkl"
if [ ! -f "$LST_FILE" ]; then
    echo "ERROR: LST file not found at $LST_FILE"
    exit 1
fi

FILE_SIZE=$(stat -c%s "$LST_FILE" 2>/dev/null || echo 0)
echo "LST file size: $FILE_SIZE bytes"

if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "WARNING: LST file seems too small ($FILE_SIZE bytes), may be incomplete"
fi

# Check last line of download log
echo "Last lines of LST download log:"
tail -5 "$LST_LOG"
echo ""

# Run the robustness analysis
echo "========================================"
echo "Starting LST Robustness Analysis..."
echo "Time: $(date)"
echo "========================================"

cd /data/lab/csee_dml
PYTHONUNBUFFERED=1 python3 -u lst_robustness.py 2>&1 | tee "$RESULT_LOG"

echo ""
echo "========================================"
echo "LST Robustness Analysis Complete"
echo "Time: $(date)"
echo "========================================"

#!/bin/bash
# Monitor V3 test progress

DATA_DIR="/Users/luisfelipesena/Development/Personal/projeto-final-ia/youbot_mcp/data/youbot"
STATUS_FILE="$DATA_DIR/status.json"
RESULTS_FILE="$DATA_DIR/test_results_v3.json"

echo "=== GRASP TEST V3 MONITOR ==="
echo "Monitoring: $STATUS_FILE"
echo "Results: $RESULTS_FILE"
echo ""

while true; do
    clear
    echo "=== GRASP TEST V3 MONITOR === $(date '+%H:%M:%S')"
    echo ""

    if [ -f "$STATUS_FILE" ]; then
        echo "--- STATUS ---"
        cat "$STATUS_FILE" | python3 -m json.tool 2>/dev/null || cat "$STATUS_FILE"
        echo ""
    else
        echo "Waiting for status file..."
    fi

    if [ -f "$RESULTS_FILE" ]; then
        echo "--- RESULTS ---"
        cat "$RESULTS_FILE" | python3 -m json.tool 2>/dev/null || cat "$RESULTS_FILE"
        echo ""
    fi

    echo "--- SCREENSHOTS ---"
    ls -la "$DATA_DIR"/v3_*.jpg 2>/dev/null | tail -5 || echo "No V3 screenshots yet"

    sleep 2
done

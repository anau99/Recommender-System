# #!/bin/bash

# # ============================================
# # AUTO START STREAMLIT + NGROK TUNNEL
# # ============================================

# STREAMLIT_FILE="streamlit_app.py"
# PORT=8501

# echo "Starting Streamlit on port $PORT ..."
# streamlit run "$STREAMLIT_FILE" --server.address=127.0.0.1 --server.port=$PORT &

# STREAMLIT_PID=$!
# echo "Streamlit PID = $STREAMLIT_PID"
# echo "Waiting for Streamlit to start..."
# sleep 3

# echo "Starting ngrok tunnel..."
# ngrok http $PORT > /dev/null &

# echo "Waiting for ngrok URL..."
# sleep 2

# NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o 'https://[a-zA-Z0-9.-]*')

# echo ""
# echo "======================================="
# echo "Public URL:"
# echo "$NGROK_URL"
# echo "======================================="
# echo ""
# echo "Press CTRL + C to stop."
# echo ""

# wait $STREAMLIT_PID



#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMLIT_FILE="$SCRIPT_DIR/streamlit_app.py"
PORT=8501

if [ ! -f "$STREAMLIT_FILE" ]; then
  echo "Error: streamlit_app.py not found: $STREAMLIT_FILE"
  exit 1
fi

# Start Streamlit
streamlit run "$STREAMLIT_FILE" --server.address=127.0.0.1 --server.port=$PORT >/dev/null 2>&1 &
ST_PID=$!

sleep 2

# Start ngrok
ngrok http $PORT >/dev/null 2>&1 &
sleep 5

# Fetch URL
RAW=$(curl -s http://127.0.0.1:4040/api/tunnels)

URL=$(echo "$RAW" | grep -oE 'https://[^"]+' | grep 'ngrok' | head -n 1)

echo ""
echo "======================================="
echo "Public URL:"
echo "$URL"
echo "======================================="
echo ""

wait $ST_PID

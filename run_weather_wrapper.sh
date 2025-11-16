#!/bin/bash
# run_weather_wrapper.sh - loads env and runs weather collector

# Load API keys
if [ -f "$HOME/.thesis_env" ]; then
  source "$HOME/.thesis_env"
else
  echo "ERROR: $HOME/.thesis_env not found" >&2
  exit 1
fi

# Run the weather collector using system Python (not venv)
# Output is appended to logs_weather.txt
/usr/bin/python3 "$HOME/thesis/scripts/data_collection/collect_weather.py" >> "$HOME/thesis/data/logs/logs_weather.txt" 2>&1

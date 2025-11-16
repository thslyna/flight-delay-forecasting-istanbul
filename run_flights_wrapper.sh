#!/bin/bash
# run_flights_wrapper.sh - loads env and runs flights collector

# Load API keys safely
if [ -f "$HOME/.thesis_env" ]; then
  source "$HOME/.thesis_env"
else
  echo "ERROR: $HOME/.thesis_env not found" >&2
  exit 1
fi

# Run the flights collector using system Python
/usr/bin/python3 "$HOME/thesis/scripts/data_collection/collect_flights.py" >> "$HOME/thesis/data/logs/logs_flights.txt" 2>&1

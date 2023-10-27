#!/bin/bash
while inotifywait -r -e modify /home/gui/INF6390/competition/ClimateNet/example; do
  echo "Detected changes, creating duplicates with timestamps..."
  timestamp=$(date +%Y%m%d%H%M%S)
  cp -r /home/gui/INF6390/competition/ClimateNet/example /home/gui/INF6390/competition/ClimateNet/example_tmp_$timestamp

  echo "Syncing to remote..."
  rclone sync /home/gui/INF6390/competition/ClimateNet/example_tmp_$timestamp sam:climateDoc/

  if [ $? -eq 0 ]; then
    echo "Sync successful. Changes pushed to sam:climateDoc/"
    # Optionally remove the temporary directory after successful sync
    rm -r /home/gui/INF6390/competition/ClimateNet/example_tmp_$timestamp
  else
    echo "Sync failed."
  fi
done

#!/bin/bash
while inotifywait -r -e modify /home/gui/INF6390/competition/ClimateNet/example; do
  echo "Detected changes, syncing to remote..."
  rclone sync /home/gui/INF6390/competition/ClimateNet/example sam:climateDoc/
  if [ $? -eq 0 ]; then
    echo "Sync successful. Changes pushed to sam:climateDoc/"
  else
    echo "Sync failed."
  fi
done

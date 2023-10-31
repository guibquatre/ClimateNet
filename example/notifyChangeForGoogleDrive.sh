#!/bin/bash
while inotifywait -r -e modify /home/gui/INF6390/competition/ClimateNet/example; do
  echo "\n\n\n\n\nDetected changes, syncing to remote...\n\n\n\n"

  rclone copy --ignore-existing /home/gui/INF6390/competition/ClimateNet/example sam:climateDoc/

  if [ $? -eq 0 ]; then
    echo "\n\n\n\nCopy successful. Changes pushed to sam:climateDoc/\n\n\n\n"
  else
    echo "\n\n\n\nCopy failed.\n\n\n\n"
  fi
done
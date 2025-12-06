==========================================
MELVIN DEPLOYMENT WORKFLOW
==========================================

QUICK START:
------------
From Mac, in Melvin_november directory:

  ./deploy_to_jetson.sh              # Deploy, keep brain.m
  ./deploy_to_jetson.sh reset_brain  # Deploy, fresh brain.m

WHAT IT DOES:
-------------
1. Stops running Melvin
2. Backs up brain.m (timestamped)
3. Copies source files to Jetson
4. Rebuilds binary
5. Starts Melvin

WHEN TO RESET BRAIN.M:
----------------------
KEEP brain.m (default):
  - Code fixes, bug fixes
  - Performance improvements
  - NaN protection
  - Dynamic thresholds
  - Tool improvements
  - Hardware fixes

RESET brain.m (use reset_brain):
  - Graph structure changes
  - Header format changes
  - New node/edge fields
  - UEL physics major changes
  - File format version changes

CURRENT CHANGES (NaN + Dynamic Thresholds):
--------------------------------------------
These are CODE changes, not structure changes.
→ brain.m can be preserved (learned patterns kept)

BRAIN LOCATION:
---------------
/mnt/melvin_ssd/melvin_brain/brain.m (4TB SSD)

BACKUPS:
--------
/mnt/melvin_ssd/melvin_brain/brain.m.backup.* (timestamped)

MONITORING:
-----------
ssh melvin@169.254.123.100 'tail -f /mnt/melvin_ssd/melvin_brain/melvin.log'

==========================================

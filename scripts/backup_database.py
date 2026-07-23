from __future__ import annotations
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ruyitutor.storage import Storage
source=Storage(ROOT/"data"/"ruyitutor.db")
target=ROOT/"data"/"backups"/f'ruyitutor-{datetime.now():%Y%m%d-%H%M%S}.db'
source.backup(target);print(target)

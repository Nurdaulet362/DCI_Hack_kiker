import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    path = ROOT / '.env'
    if path.exists():
        if os.name == 'posix': path.chmod(0o600)
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def protect_local_data():
    directory = ROOT / 'data'
    directory.mkdir(mode=0o700, exist_ok=True)
    if os.name == 'posix':
        directory.chmod(0o700)
        for name in ['dev.json', 'cqa.sqlite3', 'cqa.sqlite3-journal', 'cqa.sqlite3-wal', 'cqa.sqlite3-shm']:
            path = directory / name
            if path.exists(): path.chmod(0o600)


load_env()
protect_local_data()

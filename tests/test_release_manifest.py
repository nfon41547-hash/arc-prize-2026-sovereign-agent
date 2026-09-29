"""Release manifest currency gate."""
import subprocess
import sys
import os


def test_release_manifest_is_current():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "release_manifest.py")],
        capture_output=True, text=True, timeout=300, cwd=root)
    assert proc.returncode == 0, proc.stdout + proc.stderr

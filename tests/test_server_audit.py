"""Server audit gate."""
import subprocess
import sys
import os


def test_server_audit_all_pass():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "server_audit.py")],
        capture_output=True, text=True, timeout=300, cwd=root)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "ALL_PASS" in proc.stdout

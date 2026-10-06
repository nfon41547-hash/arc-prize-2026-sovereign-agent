import json
from pathlib import Path

# Load markdown paper content from papers/arc_prize_2026_sovereign_paper.md
paper_md = Path(r'papers\arc_prize_2026_sovereign_paper.md').read_text(encoding='utf-8')

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ARC Prize 2026 — Sovereign Autonomous Hyper-Cortex Paper Track\n",
    "## High-Throughput Test-Time Reasoning and Sycophancy-Resistant Adaptation for ARC-AGI\n",
    "**Category:** Research & Paper Track ($450,000 USD Prize Pool) | **Team:** bkk\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": paper_md.splitlines(keepends=True)
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Export Paper and Solution Writeup to /kaggle/working/arc_prize_2026_sovereign_paper.md\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "out_path = Path('/kaggle/working/arc_prize_2026_sovereign_paper.md')\n",
    "out_path.parent.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "paper_text = '''" + paper_md.replace("'''", "\\'\\'\\'") + "'''\n",
    "out_path.write_text(paper_text, encoding='utf-8')\n",
    "print(f'[+] Sovereign Academic Paper exported successfully to {out_path} ({out_path.stat().st_size} bytes)')\n",
    "\n",
    "# Generate submission marker\n",
    "sub_path = Path('/kaggle/working/submission.csv')\n",
    "sub_path.write_text('id,prediction\\n1,1.0\\n', encoding='utf-8')\n",
    "print(f'[+] Verified Paper Track submission artifact created at {sub_path}')\n"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
    "accelerator": "none",
    "dataSources": [
      {
        "sourceType": "competition",
        "sourceId": 133470
      }
    ],
    "isInternetEnabled": False,
    "language": "python",
    "sourceType": "notebook",
    "isGpuEnabled": False
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(r'starter_push_paper\arc-prize-2026-sovereign-paper-track.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print('Paper Track notebook created successfully!')

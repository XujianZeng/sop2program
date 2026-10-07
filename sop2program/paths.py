"""Data and knowledge-base locations, overridable per cross-validation fold.

SOP_DATA holds {split}.jsonl and {split}_workflows.json; SOP_KNOWLEDGE holds the rule
library and lexicons mined from that data's training split. Gold identity chains are
always read from the full corpus in data/processed, since they are annotation, not training.
"""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/os.environ.get('SOP_DATA','data/processed')
KNOWLEDGE=ROOT/os.environ.get('SOP_KNOWLEDGE','results/symbolic')
CORPUS=ROOT/'data/processed'

import os
import sys
from pathlib import Path

import pytest

os.environ["GR_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_rag.config import Settings
from grounded_rag.service import Service

CORPUS = Path(__file__).resolve().parents[1] / "evals" / "corpus"


@pytest.fixture()
def svc() -> Service:
    s = Service(Settings(offline=True, log_level="WARNING"), db_path=":memory:")
    s.ingest_dir(CORPUS)
    return s

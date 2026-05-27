"""
Runtime configuration for the production agent demo.

Environment variables override the defaults so the submitted code can keep
the required base_model/base_url/api_key locations without hard-coding secrets.
"""

import os
from pathlib import Path

from scene_config import DT, INSPECT_STEPS, RENDER_DELAY, SHUTTLE_SPEED


PROJECT_ROOT = Path(__file__).resolve().parent

# Required LLM configuration locations.
BASE_MODEL = os.getenv("BASE_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "")

COPPELIASIM_EXE = os.getenv(
    "COPPELIASIM_EXE",
    r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe",
)
SCENE_PATH = os.getenv(
    "SCENE_PATH",
    str(PROJECT_ROOT / "scenes" / "assembly_line.ttt"),
)

SIM_CONNECT_HOST = os.getenv("SIM_CONNECT_HOST", "localhost")
SIM_CONNECT_PORT = int(os.getenv("SIM_CONNECT_PORT", "23000"))
SIM_CONNECT_TIMEOUT = float(os.getenv("SIM_CONNECT_TIMEOUT", "30"))

SIM_DT = float(os.getenv("SIM_DT", str(DT)))
SIM_RENDER_DELAY = float(os.getenv("SIM_RENDER_DELAY", str(RENDER_DELAY)))
DEFAULT_SHUTTLE_SPEED = float(os.getenv("SHUTTLE_SPEED", str(SHUTTLE_SPEED)))
DEFAULT_INSPECT_STEPS = int(os.getenv("INSPECT_STEPS", str(INSPECT_STEPS)))

# Product stack heights relative to a shuttle frame.
SHUTTLE_PART_Z = float(os.getenv("SHUTTLE_PART_Z", "0.018"))
ASSEMBLY_LAYER_Z = float(os.getenv("ASSEMBLY_LAYER_Z", "0.014"))

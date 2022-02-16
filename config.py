import os
from dotenv import load_dotenv
load_dotenv()

IMAGE_EXT = '.png'


# ACTIVE-> ACTIVELY PLAYING GAME
# INACTIVE-> IN OTHER STATE
ACTIVE="ACTIVE"
INACTIVE="INACTIVE"

GAME_STATUSES = {
    "HOME_SCREEN": INACTIVE,
    "MID_ROUND": ACTIVE,
    "BUY_PHASE": INACTIVE,
    "AGENT_SELECT": INACTIVE,
    "SPECTATING": INACTIVE
}
FRAME_LABELS = GAME_STATUSES.keys()

CLASSES = {
    INACTIVE: 0,
    ACTIVE: 1,
}

MEDIA_ACTIONS = [
    "MUTE",
    "PAUSE",
    "QUIET",
    "PLAY"
]

MEDIA_ACTION_MAP = {
    INACTIVE: "PLAY",
    ACTIVE: "QUIET"
}

# PATHS
DATA_DIR = os.path.join(".","data")
VIDEOS_DIR = os.path.join(".","data","videos")

for cls in CLASSES.keys():
    if not os.path.exists(os.path.join(DATA_DIR, cls)):
        os.mkdir(os.path.join(DATA_DIR, cls))

# SPOTIFY
SPOTIFY_USERNAME = os.getenv("SPOTIFY_USERNAME", "")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("REDIRECT_URI", "")
LOWER_VOL_PCTG = 30
SCOPE = "user-modify-playback-state"
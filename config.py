import os
from dotenv import load_dotenv
load_dotenv()

IMAGE_EXT = '.png'

# ACTIVE-> ACTIVELY PLAYING GAME
# INACTIVE-> IN OTHER STATE
ACTIVE="ACTIVE"
INACTIVE="INACTIVE"

GAME_STATUS_MAP = {
    "HOME_SCREEN": INACTIVE,
    "MID_ROUND": ACTIVE,
    "BUY_PHASE": INACTIVE,
    "AGENT_SELECT": INACTIVE,
    "SPECTATING": INACTIVE
}
FRAME_LABELS = GAME_STATUS_MAP.keys()

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
    ACTIVE: ["MUTE","PAUSE","QUIET"]
}

# PATHS
DATA_DIR = os.path.join(".","data")
LABELLED_DATA_ROOT_DIR = os.path.join(DATA_DIR, 'labelled_data')
VIDEOS_DIR = os.path.join(".","data","videos")

# for cls in CLASSES.keys():
#     if not os.path.exists(os.path.join(DATA_DIR, cls)):
#         os.mkdir(os.path.join(DATA_DIR, cls))

# SPOTIFY
SPOTIFY_USERNAME = os.getenv("SPOTIFY_USERNAME", "")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("REDIRECT_URI", "")
SCOPE = "user-modify-playback-state"
LOWER_VOL_PCTG = 30

# App Settings
GAME = os.getenv("GAME", "Valorant").title()

# Training params -> Will be moved to a better config settings
BATCH_SIZE = 4
BUILD_TR_VA_TE = True
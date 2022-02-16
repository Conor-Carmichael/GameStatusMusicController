import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.util import prompt_for_user_token
from config import (
    SPOTIFY_USERNAME,
    CLIENT_ID,
    CLIENT_SECRET,
    REDIRECT_URI,
    LOWER_VOL_PCTG,
    SCOPE
)
from logger import logger

class SpotifyController:
    '''
        Light wrapper on Spotipy's client to control the users listening
        based on the config settings.
    '''

    def __init__(
        self, 
        spotify_username:str,
        scope:str,
        client_id:str, 
        client_secret:str,
        redirect:str,
        lower_vol_pctg:int,
    ) -> None:
        tkn = prompt_for_user_token(
            username=spotify_username,
            scope=scope, 
            client_id=client_id, 
            client_secret=client_secret, 
            redirect_uri=redirect
        )
        # tkn = oauth.get_access_token()
        self.sp = spotipy.Spotify(
            auth=tkn
        )        
        self.lower_vol_pctg = lower_vol_pctg

    def mute(self):
        try: 
            self.sp.volume(0)
        except SpotifyException as se:
            logger.warning("SpotifyException caught on mute()")

    def decrease_vol(self):
        try: 
            self.sp.volume(self.lower_vol_pctg)
        except SpotifyException as se:
            logger.warning("SpotifyException caught on decrease_vol()")

    def increase_vol(self):
        try: 
            self.sp.volume(100)
        except SpotifyException as se:
            logger.warning("SpotifyException caught on increase_vol()")

    def pause(self):
        try: 
            self.sp.pause_playback()
        except SpotifyException as se:
            logger.warning("SpotifyException caught on pause()")
        
spotify_controller = SpotifyController(
    SPOTIFY_USERNAME,
    SCOPE, 
    CLIENT_ID, 
    CLIENT_SECRET, 
    REDIRECT_URI, 
    LOWER_VOL_PCTG
)

from typing import List
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
        actions:List[str]
    ) -> None:
        # Authenticate user
        tkn = prompt_for_user_token(
            username=spotify_username,
            scope=scope, 
            client_id=client_id, 
            client_secret=client_secret, 
            redirect_uri=redirect
        )
        self.sp = spotipy.Spotify(
            auth=tkn
        )       
        self.actions = [a.lower() for a in actions] 
        self.lower_vol_pctg = lower_vol_pctg

    def handle_action_request(self, action:str) :
        action = action.lower()
        assert action in self.actions , "Unexpected action request."
        fn = getattr(self, action)
        if action == 'play' and self.last_action == 'quiet':
            self.increase_vol()
        else:
            fn()

        self.last_action = action
        
    def mute(self):
        try: 
            self.sp.volume(0)
        except SpotifyException as se:
            logger.warning("SpotifyException caught on mute()")

    def quiet(self):
        try: 
            self.sp.volume(self.lower_vol_pctg)
        except SpotifyException as se:
            logger.warning("SpotifyException caught on quiet()")

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

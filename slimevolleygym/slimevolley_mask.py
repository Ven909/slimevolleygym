
import numpy as np
import gymnasium as gym
from .slimevolley import SlimeVolleyEnv

class SlimeVolleyNoOpponentEnv(SlimeVolleyEnv):
    """
    Validation Environment: Opponent info is completely zeroed out.
    The agent must learn to play the ball without knowing where the opponent is.
    """
    def __init__(self):
        super(SlimeVolleyNoOpponentEnv, self).__init__()

    def getObs(self):
        obs = super(SlimeVolleyNoOpponentEnv, self).getObs()
        # obs structure:
        # [x, y, vx, vy, bx, by, bvx, bvy, ox, oy, ovx, ovy]
        # Indices 8, 9, 10, 11 are opponent state.
        
        # Mask opponent
        obs[8] = 0.0
        obs[9] = 0.0
        obs[10] = 0.0
        obs[11] = 0.0
        return obs

class SlimeVolleyMaskedEnv(SlimeVolleyEnv):
    """
    "Fog of War" Environment:
    - Opponent is ALWAYS invisible.
    - Ball is invisible when it is on the opponent's side (x > 0).
    """
    def __init__(self):
        super(SlimeVolleyMaskedEnv, self).__init__()

    def getObs(self):
        obs = super(SlimeVolleyMaskedEnv, self).getObs()
        # obs index 4 is ball_x.
        # However, the observations are scaled/relative.
        # Let's check getObservation in slimevolley.py:
        # result = [self.x, self.y, self.vx, self.vy, self.bx, self.by, self.bvx, self.bvy, self.ox, self.oy, self.ovx, self.ovy]
        # agent_left is at x = -REF_W/4.
        
        # Mask Opponent (Always)
        obs[8:12] = 0.0
        
        # Mask Ball if on opponent side (Left side, x < 0)
        # The Agent is on the Right side (x > 0).
        # We want the agent to see the ball only when it crosses the net to the Right.
        
        if self.game.ball.x < 0: # Ball is on Left side (Opponent side)
            obs[4:8] = 0.0 # Mask ball x, y, vx, vy
            
        return obs

# Register the environments so they can be gym.make'd
gym.register(
    id='SlimeVolleyNoOpponent-v0',
    entry_point='slimevolleygym.slimevolley_mask:SlimeVolleyNoOpponentEnv'
)

gym.register(
    id='SlimeVolleyMasked-v0',
    entry_point='slimevolleygym.slimevolley_mask:SlimeVolleyMaskedEnv'
)

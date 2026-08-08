"""
Atari environment wrapper — replicates the preprocessing from the Nature DQN paper.

Mnih et al., "Human-level control through deep reinforcement learning"
Nature 2015, doi:10.1038/nature14236

Preprocessing pipeline:
    1. Convert to grayscale, resize to 84×84
    2. Stack last 4 frames → (4, 84, 84) input
    3. Action repeat (frameskip = 4)
    4. No-op reset with random steps
    5. Reward clipping to [-1, 0, 1]
    6. Episode termination on life loss (optional)
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym  # fallback to legacy gym

import cv2


class AtariWrapper(gym.Wrapper):
    """Nature DQN-style Atari preprocessing wrapper.

    Args:
        env: Gym Atari environment
        frame_size: Output frame dimensions (square)
        frame_stack: Number of frames to stack
        frameskip: Number of frames per action repeat
        noop_max: Max no-op steps at episode start
        clip_reward: Whether to clip rewards to [-1, 0, 1]
        terminal_on_life_loss: End episode on life loss (for evaluation)
    """

    def __init__(
        self,
        env: gym.Env,
        frame_size: int = 84,
        frame_stack: int = 4,
        frameskip: int = 4,
        noop_max: int = 30,
        clip_reward: bool = True,
        terminal_on_life_loss: bool = False,
    ):
        super().__init__(env)
        self.frame_size = frame_size
        self.frame_stack = frame_stack
        self.frameskip = frameskip
        self.noop_max = noop_max
        self.clip_reward = clip_reward
        self.terminal_on_life_loss = terminal_on_life_loss

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            self._screen_dims = obs_shape[:2]
        else:
            self._screen_dims = (210, 160)  # Default Atari

        # Frame buffer
        self._frames: collections.deque = collections.deque(maxlen=frame_stack)

        # Life tracking
        self._lives: int = 0

        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, frame_size, frame_size),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with random no-ops."""
        obs, info = self.env.reset(**kwargs)
        self._lives = self._get_lives()

        # Random no-op steps
        if self.noop_max > 0:
            noops = np.random.randint(0, self.noop_max + 1)
            for _ in range(noops):
                obs, _, term, trunc, _ = self.env.step(0)
                if term or trunc:
                    obs, info = self.env.reset(**kwargs)
            self._lives = self._get_lives()

        # Initialize frame buffer
        frame = self._preprocess_frame(obs)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame)

        return self._get_state(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with frameskip and reward clipping."""
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frameskip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc

            if self.terminal_on_life_loss:
                lives = self._get_lives()
                if lives < self._lives:
                    terminated = True
                self._lives = lives

            if terminated or truncated:
                break

        # Clip reward
        if self.clip_reward:
            total_reward = np.sign(total_reward)

        # Add frame to buffer
        self._frames.append(self._preprocess_frame(obs))

        return self._get_state(), total_reward, terminated, truncated, info

    def _preprocess_frame(self, obs: np.ndarray) -> np.ndarray:
        """Convert to grayscale, resize to target size."""
        # Convert to grayscale (luminance)
        if obs.ndim == 3:
            # Standard Atari: (210, 160, 3) RGB
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs

        # Resize
        resized = cv2.resize(
            gray,
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.uint8)

    def _get_state(self) -> np.ndarray:
        """Stack buffered frames."""
        return np.stack(list(self._frames), axis=0)

    def _get_lives(self) -> int:
        """Get remaining lives (ALE-specific)."""
        try:
            return self.env.unwrapped.ale.lives()
        except AttributeError:
            return 1


def make_atari_env(
    game: str = "PongNoFrameskip-v4",
    frame_size: int = 84,
    frame_stack: int = 4,
    frameskip: int = 4,
    clip_reward: bool = True,
    render_mode: Optional[str] = None,
) -> AtariWrapper:
    """Create a Nature-DQN-style Atari environment.

    Args:
        game: Gym Atari environment ID
        frame_size: Frame width/height after resize
        frame_stack: Number of frames to stack (4 = paper default)
        frameskip: Frames per action step (4 = paper default)
        clip_reward: Clip rewards to [-1, 0, 1]
        render_mode: 'human', 'rgb_array', or None

    Returns:
        Wrapped environment with 84×84×4 grayscale output
    """
    if render_mode:
        env = gym.make(game, render_mode=render_mode)
    else:
        env = gym.make(game)

    wrapped = AtariWrapper(
        env,
        frame_size=frame_size,
        frame_stack=frame_stack,
        frameskip=frameskip,
        clip_reward=clip_reward,
    )
    return wrapped

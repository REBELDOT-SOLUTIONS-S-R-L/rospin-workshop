"""SO-101 MuJoCo workshop environment and LeRobot integration."""

from gymnasium.envs.registration import register, registry

from rospin_workshop.env import SO101WorkshopEnv

ENV_ID = "ROSpin/SO101Workshop-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="rospin_workshop.env:SO101WorkshopEnv",
        nondeterministic=True,
    )

__all__ = ["ENV_ID", "SO101WorkshopEnv"]
__version__ = "0.1.0"

import random

import ray
from ray import tune
from ray.rllib.agents import ppo
import os
from gym_ww.envs import ComMaWw

ray.init(local_mode=True)



def on_episode_end(info):
    episode=info['episode']
    infos=episode._agent_to_last_info[0]
    for k,v in infos.items():
        episode.custom_metrics[k]=v

configs={
        "env": ComMaWw,
        "env_config": {'num_players': 5},  # config to pass to env class

        "callbacks": {

            "on_episode_end": on_episode_end,
        },
    }

analysis = tune.run(
    "PG",
    local_dir="/ray_results",
    config=configs,
)

#
# while True:
#     print(trainer.train())
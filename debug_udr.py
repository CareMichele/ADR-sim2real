import gymnasium as gym
import numpy as np

import src.envs.custom_hopper_base 

def main():
    dr_params = {
        "ranges": {
            "thigh": (0.5, 1.5), 
            "leg": (0.5, 1.5),
            "foot": (0.5, 1.5),
        },
        "ablate":{
            "randomize_thigh": True,
            "randomize_leg": True,
            "randomize_foot": True,
        }
    }

    # 2. env created
    print("Creating env...")
    env = gym.make("CustomHopper-source-udr-v0", dr_params=dr_params)

    # 3. first reset
    env.reset()
    print(f"Initial masses: {env.unwrapped.get_parameters()}")

    # 4. test loop
    for i in range(5):
        env.reset()
        masses = env.unwrapped.get_parameters()
        print(f"Reset #{i+1} -> Actual masses: {masses}")

    env.close()

if __name__ == "__main__":
    main()
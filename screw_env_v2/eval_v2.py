from pathlib import Path
import yaml
import cv2
from stable_baselines3 import PPO
from screw_env_v2 import ScrewDetectorEnv


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env(config: dict):
    eval_config = config["eval"]
    env_config = eval_config["env"]
    targets = config["targets"]

    workspace = Path(config["workspace_path"])
    experiment_name = config["experiment_name"]
    log_dir = workspace / config["log_dir"] / experiment_name
    yolo_path = workspace / config["yolo_model"]
    mode = env_config["mode"]

    random_luminosity = env_config['random_luminosity']
    set_luminosity = env_config['set_luminosity']

    env = ScrewDetectorEnv(
        yolo_path=str(yolo_path),
        visualize_detections=env_config["visualize_detections"],
        max_steps=env_config["max_steps"],
        control_mode=env_config["control_mode"],
        targets_dict=targets,
        mode=mode,

    )
    return env, log_dir, experiment_name


def evaluate_model(config: dict):
    env, log_dir, experiment_name = build_env(config)
    checkpoint = log_dir / f"{experiment_name}_10000000_steps"
    #checkpoint = log_dir / f"{experiment_name}_2500000_steps"
    model = PPO.load(str(checkpoint), env=env)

    initial_screws = config["eval"]["env"].get("initial_confirmed_screws", [])
    custom_options = None

    if initial_screws:
        custom_options = {"initial_confirmed_screws": initial_screws}
    else:
        custom_options=None

    obs, _ = env.reset(options=custom_options)
    total_reward = 0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1

        if terminated or truncated:
            break

    print(f"Evaluation completed. Total Reward: {total_reward:.2f}, Steps: {step}")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = "/config_v2.yaml"
    config = load_config(config_path=config_path)
    evaluate_model(config)

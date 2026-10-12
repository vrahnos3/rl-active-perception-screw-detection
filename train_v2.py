from pathlib import Path
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from screw_env_v2 import ScrewDetectorEnv


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env(config: dict):
    train_config = config["train"]
    env_config = train_config["env"]
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
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, str(log_dir))
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return env, log_dir, experiment_name


def train(config: dict):
    env, log_dir, experiment_name = build_env(config)
    ppo_config = config["train"]["ppo"]

    model = PPO(
        ppo_config["policy"],
        env,
        verbose=2,
        _init_setup_model=True,
        n_steps=ppo_config["n_steps"],
        learning_rate=ppo_config["learning_rate"],
        gamma=ppo_config["gamma"],
        tensorboard_log=str(log_dir),
        batch_size=ppo_config["batch_size"],
        target_kl=ppo_config["target_kl"],
        clip_range=ppo_config["clip_range"],
        gae_lambda=ppo_config["gae_lambda"],
        vf_coef=ppo_config["vf_coef"],
        ent_coef=ppo_config["ent_coef"],
        device=ppo_config["device"],
    )

    callback_prefix = experiment_name
    checkpoint_callback = CheckpointCallback(
        save_freq=2500000,
        save_path=str(log_dir),
        name_prefix=callback_prefix,
    )
    model.learn(total_timesteps=config["train"]["total_timesteps"],
                callback=checkpoint_callback,
                progress_bar=True)
    last_step = model.num_timesteps
    model.save(str(log_dir / f"{callback_prefix}_{last_step}_steps"))


def train_from_checkpoint(config: dict):
    env, log_dir, experiment_name = build_env(config)
    ppo_config = config["train"]["ppo"]
    # resume_path = log_dir / f"final_{experiment_name}"
    resume_path = log_dir / f"{experiment_name}_2500000_steps"
    if resume_path is None:
        raise ValueError("resume_path is empty; set it in config_v2.yaml to resume training.")

    model = PPO.load(path=resume_path, env=env, tensorboard_log=str(log_dir), device=ppo_config["device"])

    callback_prefix = experiment_name
    checkpoint_callback = CheckpointCallback(
        save_freq=2500000,
        save_path=str(log_dir),
        name_prefix=callback_prefix,
    )
    model.learn(total_timesteps=config["train"]["total_timesteps"], callback=checkpoint_callback, reset_num_timesteps=False)
    last_step = model.num_timesteps
    model.save(str(log_dir / f"{callback_prefix}_{last_step}_steps"))


if __name__ == "__main__":
    config_path = "/config_v2.yaml"
    config = load_config(config_path=config_path)
    if config["train"]["train_from_scratch"]:
        train(config)
    else:
        train_from_checkpoint(config)

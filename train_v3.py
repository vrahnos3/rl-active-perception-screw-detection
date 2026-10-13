from pathlib import Path
import argparse
import sys

# Direct execution starts in screw_env_v3; shared test/utils packages live
# in its parent, controllers. Add that parent before importing dependencies.
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import os

def add_webots_python_controller_path() -> None:
    webots_home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")

    controller_python_path = Path(webots_home) / "lib" / "controller" / "python"
    controller_dll_path = Path(webots_home) / "lib" / "controller"

    if controller_python_path.exists():
        sys.path.insert(0, str(controller_python_path))
    if controller_dll_path.exists():
        os.environ["PATH"] = f"{controller_dll_path}{os.pathsep}{os.environ.get('PATH', '')}"

add_webots_python_controller_path()
from screw_env_v3 import ScrewDetectorEnv

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env(config: dict):
    train_config = config["train"]
    env_config = train_config["env"]
    
    # targets = config["targets"]  <-- You can delete this line, it is no longer needed

    workspace = Path(config["workspace_path"])
    experiment_name = config["experiment_name"]
    log_dir = workspace / config["log_dir"] / experiment_name
    yolo_path = workspace / config["yolo_model"]
    mode = env_config["mode"]
    finetuning = train_config.get("finetuning", False)

    random_luminosity = env_config['random_luminosity']
    set_luminosity = env_config['set_luminosity']

    env = ScrewDetectorEnv(
        yolo_path=str(yolo_path),
        visualize_detections=env_config["visualize_detections"],
        max_steps=env_config["max_steps"],
        control_mode=env_config["control_mode"],
        mode=mode,
        # ADD THIS LINE INSTEAD OF targets_dict
        laptop_metadata_path=str(workspace / config.get("laptop_metadata_path")), 
        finetuning=finetuning
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, str(log_dir))
    env = DummyVecEnv([lambda: env])
    normalization_checkpoint = train_config.get("resume_vecnormalize")
    if normalization_checkpoint and not train_config["train_from_scratch"]:
        env = VecNormalize.load(str(workspace / normalization_checkpoint), env)
        if env.norm_obs:
            env.close()
            raise ValueError("This v3 trainer expects norm_obs=False.")
        env.training = True
        env.norm_reward = True
    else:
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
    workspace_path = Path(config.get("workspace_path"))
    train_config = config.get("train")
    ppo_config = train_config.get("ppo")

    checkpoint_value = train_config.get("resume_checkpoint")
    if not checkpoint_value:
        raise ValueError("Set train.resume_checkpoint in config.yaml")

    resume_path = (workspace_path / checkpoint_value).resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    output_dir = (
        workspace_path / config["log_dir"] / config["experiment_name"]
    ).resolve()
    if output_dir == resume_path.parent:
        raise ValueError("Use different folder from original")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )
    normalization_checkpoint = train_config.get("resume_vecnormalize")
    if normalization_checkpoint and not (workspace_path / normalization_checkpoint).is_file():
        raise FileNotFoundError(workspace_path / normalization_checkpoint)
    # Check output paths before Monitor can create or overwrite any log files.
    env, log_dir, experiment_name = build_env(config)
    try:
        model = PPO.load(path=resume_path, 
                        env=env, 
                        tensorboard_log=str(log_dir), 
                        device=ppo_config["device"], 
                        n_steps=int(ppo_config["n_steps"]),
                        batch_size=int(ppo_config["batch_size"]),
                        learning_rate=float(ppo_config["learning_rate"]),
        )
        additional_steps = int(train_config["total_timesteps"])
        print(f"Fine-tuning: {model.num_timesteps:,} steps + "
              f"{additional_steps:,} requested steps (complete PPO rollouts)")
        print(f"New checkpoints: {log_dir}")

        checkpoint_callback = CheckpointCallback(
            save_freq=int(train_config.get("checkpoint_freq")),
            save_path=str(log_dir),
            name_prefix=experiment_name,
            save_vecnormalize=True,
        )
        model.learn(
            total_timesteps=additional_steps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        last_step = model.num_timesteps
        model.save(str(log_dir / f"{experiment_name}_{last_step}_steps"))
        env.save(str(log_dir / f"{experiment_name}_vecnormalize_{last_step}_steps.pkl"))
    finally:
        env.close()
    


if __name__ == "__main__":
    add_webots_python_controller_path()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_v3_finetune_resume.yaml")
    args = parser.parse_args()
    config = load_config(config_path=args.config)
    if config["train"]["train_from_scratch"]:
        train(config)
    else:
        train_from_checkpoint(config)

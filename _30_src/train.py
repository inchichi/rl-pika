# Import Required External Libraries
import copy
import inspect
import json
import os
import random
import signal
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque

import torch

# Import Required Internal Modules
import _00_environment
import _20_model


def get_primary_policy_path(model):
    if hasattr(model, "policy_path"):
        return str(model.policy_path)
    if hasattr(model, "actor_path"):
        return str(model.actor_path)
    raise AttributeError(
        f"{model.__class__.__name__} does not expose policy or actor path"
    )


def get_secondary_policy_path(model, primary_path=None):
    if not hasattr(model, "critic_path"):
        return None

    critic_path = str(model.critic_path)
    if primary_path is None:
        return critic_path

    current_primary_name, _ = os.path.splitext(
        os.path.basename(get_primary_policy_path(model))
    )
    current_critic_name, current_critic_ext = os.path.splitext(
        os.path.basename(critic_path)
    )
    new_primary_root, new_primary_ext = os.path.splitext(
        os.path.basename(str(primary_path))
    )

    if current_primary_name and current_critic_name.startswith(current_primary_name):
        suffix = current_critic_name[len(current_primary_name):]
        new_critic_name = f"{new_primary_root}{suffix}"
    else:
        new_critic_name = f"{new_primary_root}_critic"

    return os.path.join(
        os.path.dirname(str(primary_path)),
        f"{new_critic_name}{new_primary_ext or current_critic_ext}",
    )


def save_policy_snapshot(model, primary_path):
    original_policy_path = getattr(model, "policy_path", None)
    original_actor_path = getattr(model, "actor_path", None)
    original_critic_path = getattr(model, "critic_path", None)
    snapshot_critic_path = get_secondary_policy_path(model, primary_path)

    if hasattr(model, "policy_path"):
        model.policy_path = primary_path
    if hasattr(model, "actor_path"):
        model.actor_path = primary_path
    if hasattr(model, "critic_path") and snapshot_critic_path is not None:
        model.critic_path = snapshot_critic_path

    try:
        model.save()
    finally:
        if original_policy_path is not None:
            model.policy_path = original_policy_path
        if original_actor_path is not None:
            model.actor_path = original_actor_path
        if original_critic_path is not None:
            model.critic_path = original_critic_path

    return primary_path


def save_checkpoint(model, episode):
    primary_policy_path = get_primary_policy_path(model)
    policy_root, policy_ext = os.path.splitext(primary_policy_path)
    checkpoint_path = f"{policy_root}_ep{int(episode)}{policy_ext}"
    return save_policy_snapshot(model, checkpoint_path)


def supports_kwarg(callable_obj, kwarg_name):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return str(kwarg_name) in signature.parameters


def create_training_env(target_score, seed):
    env_kwargs = {
        "render_mode": "log",
        "target_score": int(target_score),
        "seed": seed,
    }
    if supports_kwarg(_00_environment.Env.__init__, "alternate_serve_after_point"):
        env_kwargs["alternate_serve_after_point"] = True
    return _00_environment.Env(**env_kwargs)


def set_training_match(env, train_side, model_train, opponent, *, force_player2_serve=False):
    set_kwargs = {
        "random_serve": False,
        "return_state": False,
    }
    if supports_kwarg(env.set, "force_player2_serve"):
        set_kwargs["force_player2_serve"] = bool(force_player2_serve)

    if str(train_side) == "1p":
        env.set(player1=model_train, player2=opponent, **set_kwargs)
    else:
        env.set(player1=opponent, player2=model_train, **set_kwargs)


def get_training_state_path(model):
    policy_root, _ = os.path.splitext(get_primary_policy_path(model))
    return f"{policy_root}.train_state.json"


def load_training_state(path):
    try:
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def save_training_state(
    path,
    *,
    policy_path,
    episodes_completed,
    win_count,
    loss_count,
    draw_count,
    epsilon,
):
    payload = {
        "policy_path": str(policy_path),
        "episodes_completed": int(episodes_completed),
        "win_count": int(win_count),
        "loss_count": int(loss_count),
        "draw_count": int(draw_count),
        "epsilon": None if epsilon is None else float(epsilon),
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
    os.replace(tmp_path, path)


def load_policy_snapshot(model, policy_path):
    snapshot_path = str(policy_path)
    if not snapshot_path or not os.path.exists(snapshot_path):
        return False

    module_name = str(model.__class__.__module__).lower()
    if ".qlearning." in module_name:
        model.policy = _20_model.qlearning._02_qtable.load_qtable(snapshot_path)
        setattr(model, "_loaded_policy_path", snapshot_path)
        return True
    if ".sarsa." in module_name:
        model.policy = _20_model.sarsa._02_qtable.load_qtable(snapshot_path)
        setattr(model, "_loaded_policy_path", snapshot_path)
        return True
    if ".dqn." in module_name:
        _20_model.dqn._02_network.load_nn(model.policy, snapshot_path)
        model.target_policy.load_state_dict(model.policy.state_dict())
        setattr(model, "_loaded_policy_path", snapshot_path)
        return True
    if ".ppo." in module_name or ".a2c." in module_name or ".ddpg." in module_name:
        critic_snapshot_path = get_secondary_policy_path(model, snapshot_path)
        model.actor.load_state_dict(
            torch.load(snapshot_path, map_location=model.device, weights_only=True)
        )
        if critic_snapshot_path and os.path.exists(critic_snapshot_path):
            model.critic.load_state_dict(
                torch.load(
                    critic_snapshot_path,
                    map_location=model.device,
                    weights_only=True,
                )
            )
        if hasattr(model, "actor_old"):
            model.actor_old.load_state_dict(model.actor.state_dict())
        setattr(model, "_loaded_policy_path", snapshot_path)
        return True
    return False


def sync_self_play_snapshot(model_train, model_opponent):
    train_module = str(model_train.__class__.__module__).lower()
    opponent_module = str(model_opponent.__class__.__module__).lower()
    if train_module != opponent_module:
        return False

    if ".qlearning." in train_module or ".sarsa." in train_module:
        model_opponent.policy = copy.deepcopy(model_train.policy)
        setattr(model_opponent, "_loaded_policy_path", get_primary_policy_path(model_train))
        return True

    if ".dqn." in train_module:
        model_opponent.policy.load_state_dict(model_train.policy.state_dict())
        if hasattr(model_opponent, "target_policy"):
            model_opponent.target_policy.load_state_dict(
                model_opponent.policy.state_dict()
            )
        setattr(model_opponent, "_loaded_policy_path", get_primary_policy_path(model_train))
        return True

    if ".ppo." in train_module or ".a2c." in train_module or ".ddpg." in train_module:
        if hasattr(model_train, "actor") and hasattr(model_opponent, "actor"):
            model_opponent.actor.load_state_dict(model_train.actor.state_dict())
            if hasattr(model_train, "critic") and hasattr(model_opponent, "critic"):
                model_opponent.critic.load_state_dict(model_train.critic.state_dict())
            if hasattr(model_opponent, "actor_old"):
                model_opponent.actor_old.load_state_dict(model_opponent.actor.state_dict())
            setattr(model_opponent, "_loaded_policy_path", get_primary_policy_path(model_train))
            return True
    return False


def resolve_runtime_setting(conf_value, train_conf, train_key, default):
    if conf_value is None and isinstance(train_conf, dict):
        conf_value = train_conf.get(train_key, default)
    if conf_value is None:
        return default
    return conf_value


def parse_algorithm_policy_spec(value):
    algorithm_name, sep, policy_name = str(value).strip().partition(":")
    algorithm_name = algorithm_name.strip().lower()
    policy_name = None if (not sep or policy_name.strip() in ("", "None", "none")) else policy_name.strip()
    return algorithm_name, policy_name


def parse_curriculum_schedule(value, default_spec="rule"):
    schedule_text = str(value or "").strip()
    if schedule_text == "":
        base_spec = str(default_spec or "rule").strip() or "rule"
        return [(0, base_spec)]

    schedule = []
    for chunk in schedule_text.split(","):
        entry = chunk.strip()
        if entry == "":
            continue
        start_text, sep, opponent_spec = entry.partition("=")
        if sep == "":
            start_episode = 0
            opponent_spec = start_text
        else:
            try:
                start_episode = int(start_text.strip())
            except ValueError:
                continue
        opponent_spec = str(opponent_spec).strip()
        if opponent_spec == "":
            continue
        schedule.append((max(0, int(start_episode)), opponent_spec))

    if not schedule:
        base_spec = str(default_spec or "rule").strip() or "rule"
        return [(0, base_spec)]

    schedule.sort(key=lambda item: item[0])
    if schedule[0][0] != 0:
        base_spec = str(default_spec or "rule").strip() or "rule"
        schedule.insert(0, (0, base_spec))

    deduplicated_schedule = []
    for start_episode, opponent_spec in schedule:
        if deduplicated_schedule and deduplicated_schedule[-1][0] == start_episode:
            deduplicated_schedule[-1] = (start_episode, opponent_spec)
        else:
            deduplicated_schedule.append((start_episode, opponent_spec))
    return deduplicated_schedule


def resolve_curriculum_opponent(schedule, episode_idx, default_spec="rule"):
    active_spec = str(default_spec or "rule").strip() or "rule"
    for start_episode, opponent_spec in schedule:
        if int(episode_idx) >= int(start_episode):
            active_spec = str(opponent_spec).strip() or active_spec
        else:
            break
    return active_spec


def format_curriculum_schedule(schedule):
    return ",".join(f"{int(start)}={spec}" for start, spec in schedule)


def resolve_training_side(default_train_side, side_mode, episode_index):
    normalized_mode = str(side_mode).strip().lower()
    normalized_default_side = str(default_train_side).strip().lower()
    base_side = "1p" if normalized_default_side in ("1p", "player1", "p1") else "2p"
    if normalized_mode == "alternate":
        if int(episode_index) % 2 == 0:
            return base_side
        return "2p" if base_side == "1p" else "1p"
    return base_side


def get_self_play_snapshot_dir(model):
    primary_policy_path = get_primary_policy_path(model)
    policy_root, _ = os.path.splitext(primary_policy_path)
    policy_name = os.path.basename(policy_root)
    snapshot_dir = os.path.join(
        os.path.dirname(primary_policy_path),
        "_self_play_pool",
        policy_name,
    )
    os.makedirs(snapshot_dir, exist_ok=True)
    return snapshot_dir


def get_self_play_pool_path(model):
    primary_policy_path = get_primary_policy_path(model)
    policy_root, _ = os.path.splitext(primary_policy_path)
    policy_name = os.path.basename(policy_root)
    snapshot_dir = get_self_play_snapshot_dir(model)
    return os.path.join(snapshot_dir, f"{policy_name}.selfplay_pool.json")


def load_self_play_pool(path):
    try:
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        episode = int(item.get("episode", 0))
        policy_path = str(item.get("policy_path", "")).strip()
        if policy_path == "" or not os.path.exists(policy_path):
            continue
        secondary_policy_path = str(item.get("secondary_policy_path", "")).strip()
        record = {
            "episode": max(0, episode),
            "policy_path": policy_path,
        }
        if secondary_policy_path != "":
            record["secondary_policy_path"] = secondary_policy_path
        records.append(record)

    records.sort(key=lambda record: int(record["episode"]))
    return records


def save_self_play_pool(path, records):
    payload = []
    for record in records:
        if not isinstance(record, dict):
            continue
        policy_path = str(record.get("policy_path", "")).strip()
        if policy_path == "" or not os.path.exists(policy_path):
            continue
        entry = {
            "episode": int(record.get("episode", 0)),
            "policy_path": policy_path,
        }
        secondary_policy_path = str(record.get("secondary_policy_path", "")).strip()
        if secondary_policy_path != "":
            entry["secondary_policy_path"] = secondary_policy_path
        payload.append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
    os.replace(tmp_path, path)


def save_self_play_snapshot(model, episode):
    primary_policy_path = get_primary_policy_path(model)
    policy_root, policy_ext = os.path.splitext(primary_policy_path)
    policy_name = os.path.basename(policy_root)
    snapshot_dir = get_self_play_snapshot_dir(model)
    snapshot_policy_path = os.path.join(
        snapshot_dir,
        f"{policy_name}_sp_ep{int(episode)}{policy_ext}",
    )
    snapshot_secondary_path = get_secondary_policy_path(model, snapshot_policy_path)
    save_policy_snapshot(model, snapshot_policy_path)
    record = {
        "episode": int(episode),
        "policy_path": snapshot_policy_path,
    }
    if snapshot_secondary_path is not None:
        record["secondary_policy_path"] = snapshot_secondary_path
    return record


def _remove_self_play_snapshot_files(record):
    if not isinstance(record, dict):
        return
    for key in ("policy_path", "secondary_policy_path"):
        file_path = str(record.get(key, "")).strip()
        if file_path == "":
            continue
        file_name = os.path.basename(file_path)
        if "_sp_ep" not in file_name:
            continue
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


def register_self_play_pool_snapshot(records, snapshot_record, max_size):
    filtered = []
    snapshot_path = str(snapshot_record.get("policy_path", "")).strip()
    snapshot_episode = int(snapshot_record.get("episode", 0))

    for record in records:
        if not isinstance(record, dict):
            continue
        policy_path = str(record.get("policy_path", "")).strip()
        if policy_path == "" or not os.path.exists(policy_path):
            continue
        if policy_path == snapshot_path:
            continue
        if int(record.get("episode", -1)) == snapshot_episode:
            continue
        filtered.append(record)

    filtered.append(snapshot_record)
    filtered.sort(key=lambda record: int(record.get("episode", 0)))

    trimmed = []
    for record in filtered:
        trimmed.append(record)
    if max_size > 0 and len(trimmed) > max_size:
        removed = trimmed[:-max_size]
        for record in removed:
            _remove_self_play_snapshot_files(record)
        trimmed = trimmed[-max_size:]
    return trimmed


def select_self_play_pool_opponent(
    model_train,
    model_opponent,
    records,
    latest_prob,
    warmup_episode,
    episodes_completed,
):
    candidate_records = [
        record for record in records
        if int(record.get("episode", 0)) >= int(warmup_episode)
    ]

    if (
        episodes_completed < warmup_episode
        or len(candidate_records) == 0
        or random.random() < latest_prob
    ):
        if sync_self_play_snapshot(model_train, model_opponent):
            return "self:latest", True
        return "self", True

    selected = random.choice(candidate_records)
    selected_path = str(selected.get("policy_path", "")).strip()
    selected_episode = int(selected.get("episode", 0))
    if selected_path != "" and load_policy_snapshot(model_opponent, selected_path):
        return f"self:pool@{selected_episode}", False

    if sync_self_play_snapshot(model_train, model_opponent):
        return "self:latest", True
    return "self", True


def create_training_opponent(conf, model_train, opponent_spec):
    opponent_spec = str(opponent_spec or "rule").strip()
    normalized_spec = opponent_spec.lower()
    if normalized_spec in ("", "rule"):
        return "RULE", "rule"
    if normalized_spec == "human":
        return "HUMAN", "human"
    if normalized_spec == "self":
        opponent_conf = copy.deepcopy(conf)
        opponent_conf.train_rewrite = False
        opponent_model = _20_model.create_model(
            opponent_conf,
            algorithm_name=str(conf.train_algorithm).strip().lower(),
            policy_name_for_play=getattr(conf, "train_policy", None),
        )
        sync_self_play_snapshot(model_train, opponent_model)
        return opponent_model, "self"

    algorithm_name, policy_name = parse_algorithm_policy_spec(opponent_spec)
    if algorithm_name == "rule":
        return "RULE", "rule"
    if algorithm_name == "human":
        return "HUMAN", "human"

    opponent_conf = copy.deepcopy(conf)
    opponent_conf.train_rewrite = False
    opponent_model = _20_model.create_model(
        opponent_conf,
        algorithm_name=algorithm_name,
        policy_name_for_play=policy_name,
    )
    setattr(opponent_model, "_loaded_policy_path", get_primary_policy_path(opponent_model))
    opponent_label = algorithm_name if policy_name is None else f"{algorithm_name}:{policy_name}"
    return opponent_model, opponent_label


def get_model_algorithm_name(model):
    module_name = str(model.__class__.__module__).strip().lower()
    pieces = module_name.split(".")
    if len(pieces) >= 2:
        return str(pieces[1]).strip().lower()
    return ""


def build_worker_opponent_descriptor(opponent, label, train_policy_path):
    if isinstance(opponent, str):
        return {
            "kind": "builtin",
            "value": str(opponent),
            "label": str(label),
        }

    policy_path = str(getattr(opponent, "_loaded_policy_path", "")).strip()
    if policy_path == "" or not os.path.exists(policy_path):
        policy_path = get_primary_policy_path(opponent)

    normalized_label = str(label).strip().lower()
    worker_policy_path = str(train_policy_path).strip()
    if normalized_label in ("self", "self:latest"):
        if worker_policy_path != "":
            policy_path = worker_policy_path
    elif normalized_label.startswith("self:"):
        if policy_path == "" or not os.path.exists(policy_path):
            if worker_policy_path != "":
                policy_path = worker_policy_path

    return {
        "kind": "model",
        "algorithm": get_model_algorithm_name(opponent),
        "policy_name": getattr(opponent, "policy_name", None),
        "policy_path": str(policy_path),
        "label": str(label),
    }


def run_parallel_episode_worker(task):
    # Worker-side rollout collection uses CPU to avoid GPU contention with learner.
    try:
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass

    seed = int(task.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    from _10_config.conf import Config

    conf_local = Config()
    conf_payload = task.get("conf_payload", {})
    if isinstance(conf_payload, dict):
        for key, value in conf_payload.items():
            setattr(conf_local, key, value)

    conf_local.mode = "train"
    conf_local.train_algorithm = str(task["train_algorithm"])
    conf_local.train_policy = task.get("train_policy_name")
    conf_local.train_side = str(task["train_side"])
    conf_local.train_rewrite = False
    conf_local.seed = seed
    conf_local.target_score_train = int(task["target_score"])

    model_train = _20_model.create_model(
        conf_local,
        algorithm_name=str(task["train_algorithm"]),
        policy_name_for_play=task.get("train_policy_name"),
    )
    if not load_policy_snapshot(model_train, str(task["train_policy_path"])):
        raise RuntimeError(f"failed to load train policy: {task['train_policy_path']}")
    if hasattr(model_train, "set_training_progress") and callable(model_train.set_training_progress):
        model_train.set_training_progress(int(task.get("episodes_completed", 0)))

    opponent_desc = task.get("opponent", {})
    opponent = "RULE"
    if isinstance(opponent_desc, dict) and opponent_desc.get("kind") == "builtin":
        opponent = str(opponent_desc.get("value", "RULE"))
    else:
        opponent_algorithm = str(opponent_desc.get("algorithm", "")).strip().lower()
        opponent_policy_name = opponent_desc.get("policy_name")
        opponent_policy_path = str(opponent_desc.get("policy_path", "")).strip()
        if opponent_algorithm == "":
            raise RuntimeError(f"invalid opponent descriptor: {opponent_desc}")
        opponent_conf = copy.deepcopy(conf_local)
        opponent_conf.train_rewrite = False
        opponent_conf.train_algorithm = opponent_algorithm
        opponent_conf.train_policy = opponent_policy_name
        opponent = _20_model.create_model(
            opponent_conf,
            algorithm_name=opponent_algorithm,
            policy_name_for_play=opponent_policy_name,
        )
        if not load_policy_snapshot(opponent, opponent_policy_path):
            raise RuntimeError(f"failed to load opponent policy: {opponent_policy_path}")

    env = create_training_env(
        target_score=int(task["target_score"]),
        seed=seed,
    )

    train_side = str(task["train_side"])
    force_player2_serve = bool(task.get("force_player2_serve", False))
    set_training_match(
        env,
        train_side,
        model_train,
        opponent,
        force_player2_serve=force_player2_serve,
    )

    state_mat = env.get_state(player=train_side)
    transitions = []
    episode_reward = 0.0
    episode_steps = 0
    final_score = {"p1": 0, "p2": 0, "events": {}}

    while True:
        transition, state_next_mat = model_train.get_transition(env, state_mat)
        state, action_idx, log_prob_old, state_next, reward_next, done, score = transition
        reward_value = float(reward_next)
        episode_reward += reward_value
        episode_steps += 1
        transitions.append(
            (
                state,
                int(action_idx),
                float(log_prob_old),
                state_next,
                reward_value,
                bool(done),
                None,
            )
        )
        state_mat = state_next_mat
        if done:
            final_score = {
                "p1": int(score.get("p1", 0)),
                "p2": int(score.get("p2", 0)),
                "events": score.get("events") or {},
            }
            break

    env.close()
    return {
        "episode_index": int(task["episode_index"]),
        "train_side": train_side,
        "opponent_label": str(task.get("opponent_label", "")),
        "episode_reward": float(episode_reward),
        "episode_steps": int(episode_steps),
        "score": final_score,
        "transitions": transitions,
        "self_play_mode": bool(task.get("self_play_mode", False)),
    }


def run(conf):
    """====================================================================================================
    ## Create Required Instances
    ===================================================================================================="""
    # - Create Envionment Instance
    env = create_environment_instance(conf)

    """====================================================================================================
    ## Run a number of Episodes for Training
    ===================================================================================================="""
    # - Load Models for Training and Opponent Players
    model_train = load_training_model(conf)
    train_model_module_name = str(model_train.__class__.__module__).lower()
    train_conf = getattr(model_train, "train_conf", None)
    self_play_snapshot_interval = int(resolve_runtime_setting(
        getattr(conf, "self_play_snapshot_interval", None),
        train_conf,
        "self_play_snapshot_interval",
        1000,
    ))
    if self_play_snapshot_interval < 1:
        self_play_snapshot_interval = 1
    train_side_mode = str(resolve_runtime_setting(
        getattr(conf, "train_side_mode", None),
        train_conf,
        "train_side_mode",
        "fixed",
    )).strip().lower()
    if train_side_mode not in ("fixed", "alternate"):
        train_side_mode = "fixed"
    train_num_workers = int(resolve_runtime_setting(
        getattr(conf, "train_num_workers", None),
        train_conf,
        "train_num_workers",
        1,
    ))
    if train_num_workers < 1:
        train_num_workers = 1
    parallel_rollout_enabled = train_num_workers > 1 and ".ppo." in train_model_module_name
    if train_num_workers > 1 and not parallel_rollout_enabled:
        print(
            f"[MultiProcess] algorithm={conf.train_algorithm} does not support parallel rollout; "
            "fallback to 1 worker."
        )
        train_num_workers = 1

    self_play_pool_enabled = bool(resolve_runtime_setting(
        getattr(conf, "self_play_pool_enabled", None),
        train_conf,
        "self_play_pool_enabled",
        False,
    ))
    self_play_pool_size = int(resolve_runtime_setting(
        getattr(conf, "self_play_pool_size", None),
        train_conf,
        "self_play_pool_size",
        0,
    ))
    self_play_pool_latest_prob = float(resolve_runtime_setting(
        getattr(conf, "self_play_pool_latest_prob", None),
        train_conf,
        "self_play_pool_latest_prob",
        0.5,
    ))
    self_play_pool_resample_interval = int(resolve_runtime_setting(
        getattr(conf, "self_play_pool_resample_interval", None),
        train_conf,
        "self_play_pool_resample_interval",
        1,
    ))
    self_play_pool_warmup_episode = int(resolve_runtime_setting(
        getattr(conf, "self_play_pool_warmup_episode", None),
        train_conf,
        "self_play_pool_warmup_episode",
        0,
    ))
    self_play_rule_mix_prob = float(resolve_runtime_setting(
        getattr(conf, "self_play_rule_mix_prob", None),
        train_conf,
        "self_play_rule_mix_prob",
        0.0,
    ))
    if self_play_pool_size < 1:
        self_play_pool_enabled = False
    if self_play_pool_latest_prob < 0.0:
        self_play_pool_latest_prob = 0.0
    if self_play_pool_latest_prob > 1.0:
        self_play_pool_latest_prob = 1.0
    if self_play_pool_resample_interval < 1:
        self_play_pool_resample_interval = 1
    if self_play_pool_warmup_episode < 0:
        self_play_pool_warmup_episode = 0
    if self_play_rule_mix_prob < 0.0:
        self_play_rule_mix_prob = 0.0
    if self_play_rule_mix_prob > 1.0:
        self_play_rule_mix_prob = 1.0
    default_train_opponent_spec = str(
        getattr(conf, "train_opponent", "rule") or "rule"
    ).strip() or "rule"
    curriculum_enabled = bool(resolve_runtime_setting(
        getattr(conf, "curriculum_enabled", None),
        train_conf,
        "curriculum_enabled",
        False,
    ))
    curriculum_schedule_raw = resolve_runtime_setting(
        getattr(conf, "curriculum_schedule", None),
        train_conf,
        "curriculum_schedule",
        f"0={default_train_opponent_spec}",
    )
    curriculum_schedule = parse_curriculum_schedule(
        curriculum_schedule_raw,
        default_train_opponent_spec,
    )
    active_train_opponent_spec = default_train_opponent_spec
    if curriculum_enabled:
        active_train_opponent_spec = resolve_curriculum_opponent(
            curriculum_schedule,
            0,
            default_train_opponent_spec,
        )
    model_opponent, active_train_opponent_label = create_training_opponent(
        conf,
        model_train,
        active_train_opponent_spec,
    )
    self_play_mode = str(active_train_opponent_spec).strip().lower() == "self"
    self_play_runtime_label = active_train_opponent_label
    self_play_using_latest = True
    last_self_play_resample_episode = -1
    self_play_mix_rule_model = "RULE"
    self_play_mix_rule_label = "rule"
    self_play_pool_path = get_self_play_pool_path(model_train)
    self_play_pool_records = []
    if self_play_pool_enabled:
        self_play_pool_records = load_self_play_pool(self_play_pool_path)
        if len(self_play_pool_records) > self_play_pool_size:
            self_play_pool_records = self_play_pool_records[-self_play_pool_size:]
            save_self_play_pool(self_play_pool_path, self_play_pool_records)
    state_path = get_training_state_path(model_train)
    rewrite_mode = bool(getattr(conf, "train_rewrite", False))
    reset_epsilon_mode = bool(getattr(conf, "reset_epsilon", False))
    resume_state = None if rewrite_mode else load_training_state(state_path)
    start_episode = 0
    win_count = 0
    loss_count = 0
    draw_count = 0
    episodes_completed = 0
    epsilon_value = getattr(model_train, "epsilon", None)
    last_saved_policy_path = get_primary_policy_path(model_train)

    if resume_state is not None:
        resume_policy_path = str(
            resume_state.get("policy_path", get_primary_policy_path(model_train))
        ).strip()
        if load_policy_snapshot(model_train, resume_policy_path):
            start_episode = int(resume_state.get("episodes_completed", 0))
            if start_episode < 0:
                start_episode = 0
            win_count = int(resume_state.get("win_count", 0))
            loss_count = int(resume_state.get("loss_count", 0))
            draw_count = int(resume_state.get("draw_count", 0))
            episodes_completed = start_episode
            epsilon_saved = resume_state.get("epsilon", None)
            if hasattr(model_train, "epsilon"):
                if reset_epsilon_mode:
                    epsilon_start = None
                    train_conf = getattr(model_train, "train_conf", None)
                    if isinstance(train_conf, dict):
                        epsilon_start = train_conf.get("epsilon_start", None)
                    if epsilon_start is not None:
                        model_train.epsilon = float(epsilon_start)
                elif epsilon_saved is not None:
                    model_train.epsilon = float(epsilon_saved)
            epsilon_value = getattr(model_train, "epsilon", None)
            last_saved_policy_path = resume_policy_path
            resume_suffix = " reset_epsilon=True" if reset_epsilon_mode else ""
            print(
                f"[Resume] episode={start_episode} epsilon={epsilon_value} "
                f"policy={resume_policy_path}{resume_suffix}"
            )
    elif rewrite_mode:
        print(
            f"[Rewrite] start from scratch: policy={get_primary_policy_path(model_train)}"
        )

    if curriculum_enabled:
        # Align active opponent with resumed episode index before entering the loop.
        resume_active_spec = resolve_curriculum_opponent(
            curriculum_schedule,
            start_episode,
            default_train_opponent_spec,
        )
        if resume_active_spec != active_train_opponent_spec:
            active_train_opponent_spec = resume_active_spec
            model_opponent, active_train_opponent_label = create_training_opponent(
                conf,
                model_train,
                active_train_opponent_spec,
            )
            self_play_mode = str(active_train_opponent_spec).strip().lower() == "self"
        print(
            f"[Curriculum] schedule={format_curriculum_schedule(curriculum_schedule)} "
            f"active={active_train_opponent_label}"
        )
    print(
        f"[TrainSide] mode={train_side_mode} base={conf.train_side} "
        f"workers={train_num_workers}"
    )
    if self_play_pool_enabled:
        print(
            f"[SelfPlayPool] enabled size={self_play_pool_size} "
            f"latest_prob={self_play_pool_latest_prob:.2f} "
            f"resample={self_play_pool_resample_interval} "
            f"warmup={self_play_pool_warmup_episode} "
            f"loaded={len(self_play_pool_records)}"
        )
    if self_play_mode:
        print(f"[SelfPlayMix] rule_mix={self_play_rule_mix_prob:.2f}")

    if self_play_mode and not isinstance(model_opponent, str):
        if self_play_pool_enabled:
            self_play_runtime_label, self_play_using_latest = select_self_play_pool_opponent(
                model_train,
                model_opponent,
                self_play_pool_records,
                self_play_pool_latest_prob,
                self_play_pool_warmup_episode,
                episodes_completed,
            )
            last_self_play_resample_episode = start_episode
            print(
                f"[SelfPlay] opponent={self_play_runtime_label} "
                f"snapshot_interval={self_play_snapshot_interval}"
            )
        elif sync_self_play_snapshot(model_train, model_opponent):
            print(
                f"[SelfPlay] snapshot initialized interval={self_play_snapshot_interval}"
            )

    # - Run a number of Episodes for Training
    if start_episode > int(conf.num_episode):
        start_episode = int(conf.num_episode)
        episodes_completed = start_episode

    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    total_episodes = int(conf.num_episode)
    log_interval = 100
    checkpoint_interval = 10000
    recent_results = deque(maxlen=log_interval)
    recent_episode_rewards = deque(maxlen=log_interval)
    started_at = time.perf_counter()

    if parallel_rollout_enabled:
        worker_snapshot_path = ""
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm_parallel(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _handle_sigterm_parallel)
        conf_payload = dict(vars(conf))

        try:
            if hasattr(model_train, "set_training_progress") and callable(model_train.set_training_progress):
                model_train.set_training_progress(start_episode)

            policy_root, policy_ext = os.path.splitext(get_primary_policy_path(model_train))
            worker_snapshot_path = f"{policy_root}.__mp_worker{policy_ext}"

            with ProcessPoolExecutor(max_workers=train_num_workers) as executor:
                epi_idx = start_episode
                while epi_idx < total_episodes:
                    batch_end = min(epi_idx + train_num_workers, total_episodes)

                    # Snapshot current learner policy once per batch for worker rollout.
                    save_policy_snapshot(model_train, worker_snapshot_path)

                    episode_plans = []
                    for current_epi in range(epi_idx, batch_end):
                        desired_train_opponent_spec = active_train_opponent_spec
                        if curriculum_enabled:
                            desired_train_opponent_spec = resolve_curriculum_opponent(
                                curriculum_schedule,
                                current_epi,
                                default_train_opponent_spec,
                            )

                        if desired_train_opponent_spec != active_train_opponent_spec:
                            active_train_opponent_spec = desired_train_opponent_spec
                            model_opponent, active_train_opponent_label = create_training_opponent(
                                conf,
                                model_train,
                                active_train_opponent_spec,
                            )
                            self_play_mode = str(active_train_opponent_spec).strip().lower() == "self"
                            self_play_runtime_label = active_train_opponent_label
                            self_play_using_latest = True
                            last_self_play_resample_episode = -1
                            print()
                            print(
                                f"[Curriculum] episode={current_epi + 1} opponent={active_train_opponent_label}",
                                flush=True,
                            )
                            if self_play_mode and not isinstance(model_opponent, str):
                                if self_play_pool_enabled:
                                    self_play_runtime_label, self_play_using_latest = select_self_play_pool_opponent(
                                        model_train,
                                        model_opponent,
                                        self_play_pool_records,
                                        self_play_pool_latest_prob,
                                        self_play_pool_warmup_episode,
                                        episodes_completed,
                                    )
                                    last_self_play_resample_episode = current_epi
                                    print(
                                        f"[SelfPlay] opponent={self_play_runtime_label} "
                                        f"snapshot_interval={self_play_snapshot_interval}"
                                    )
                                elif sync_self_play_snapshot(model_train, model_opponent):
                                    print(
                                        f"[SelfPlay] snapshot initialized interval={self_play_snapshot_interval}"
                                    )

                        episode_train_side = resolve_training_side(
                            conf.train_side,
                            train_side_mode,
                            current_epi - start_episode,
                        )
                        episode_train_player_key = "p1" if episode_train_side == "1p" else "p2"
                        force_player2_serve = bool((current_epi - start_episode) % 2 == 1)

                        base_model_opponent = model_opponent
                        episode_opponent = model_opponent
                        episode_opponent_label = active_train_opponent_label
                        if self_play_mode and not isinstance(model_opponent, str):
                            if random.random() < self_play_rule_mix_prob:
                                episode_opponent = self_play_mix_rule_model
                                episode_opponent_label = f"mix:{self_play_mix_rule_label}"
                            else:
                                if self_play_pool_enabled:
                                    should_resample_self_play = (
                                        last_self_play_resample_episode < 0
                                        or (current_epi - last_self_play_resample_episode)
                                        >= self_play_pool_resample_interval
                                    )
                                    if should_resample_self_play:
                                        self_play_runtime_label, self_play_using_latest = select_self_play_pool_opponent(
                                            model_train,
                                            model_opponent,
                                            self_play_pool_records,
                                            self_play_pool_latest_prob,
                                            self_play_pool_warmup_episode,
                                            episodes_completed,
                                        )
                                        last_self_play_resample_episode = current_epi
                                    episode_opponent_label = self_play_runtime_label
                                else:
                                    episode_opponent_label = "self:latest"

                        episode_plans.append({
                            "episode_index": int(current_epi),
                            "train_side": episode_train_side,
                            "train_player_key": episode_train_player_key,
                            "force_player2_serve": force_player2_serve,
                            "episode_opponent": episode_opponent,
                            "episode_opponent_label": episode_opponent_label,
                            "base_model_opponent": base_model_opponent,
                            "self_play_mode": bool(self_play_mode),
                        })

                    futures = []
                    for plan in episode_plans:
                        opponent_descriptor = build_worker_opponent_descriptor(
                            plan["episode_opponent"],
                            plan["episode_opponent_label"],
                            worker_snapshot_path,
                        )
                        task = {
                            "conf_payload": conf_payload,
                            "episode_index": plan["episode_index"],
                            "train_algorithm": str(conf.train_algorithm).strip().lower(),
                            "train_policy_name": getattr(model_train, "policy_name", getattr(conf, "train_policy", None)),
                            "train_policy_path": worker_snapshot_path,
                            "train_side": plan["train_side"],
                            "opponent": opponent_descriptor,
                            "opponent_label": plan["episode_opponent_label"],
                            "force_player2_serve": plan["force_player2_serve"],
                            "target_score": int(conf.target_score_train),
                            "episodes_completed": int(episodes_completed),
                            "self_play_mode": bool(plan["self_play_mode"]),
                            "seed": int(conf.seed) + int(plan["episode_index"]) + 1,
                        }
                        futures.append(executor.submit(run_parallel_episode_worker, task))

                    for plan, future in zip(episode_plans, futures):
                        worker_result = future.result()
                        transitions = worker_result.get("transitions", [])
                        for transition in transitions:
                            model_train.update(transition)

                        score = worker_result.get("score", {})
                        p1_score = int(score.get("p1", 0))
                        p2_score = int(score.get("p2", 0))
                        if p1_score == p2_score:
                            result = "DRAW"
                            draw_count += 1
                        else:
                            winner_key = "p1" if p1_score > p2_score else "p2"
                            if winner_key == plan["train_player_key"]:
                                result = "WIN"
                                win_count += 1
                            else:
                                result = "LOSS"
                                loss_count += 1

                        episodes_completed = int(plan["episode_index"]) + 1
                        if hasattr(model_train, "set_training_progress") and callable(model_train.set_training_progress):
                            model_train.set_training_progress(episodes_completed)
                        if ".dqn." in train_model_module_name and hasattr(model_train, "update_epsilon") and callable(model_train.update_epsilon):
                            model_train.update_epsilon()
                        epsilon_value = getattr(model_train, "epsilon", None)
                        event_info = score.get("events") or {}
                        timeout_note = " timeout" if event_info.get("timeout", False) else ""
                        episode_reward = float(worker_result.get("episode_reward", 0.0))
                        episode_steps = int(worker_result.get("episode_steps", 0))
                        recent_results.append(result)
                        recent_episode_rewards.append(float(episode_reward))
                        snapshot_note = ""

                        if (
                            plan["self_play_mode"]
                            and not isinstance(plan["base_model_opponent"], str)
                            and episodes_completed % self_play_snapshot_interval == 0
                        ):
                            if self_play_pool_enabled:
                                try:
                                    snapshot_record = save_self_play_snapshot(
                                        model_train,
                                        episodes_completed,
                                    )
                                    self_play_pool_records = register_self_play_pool_snapshot(
                                        self_play_pool_records,
                                        snapshot_record,
                                        self_play_pool_size,
                                    )
                                    save_self_play_pool(self_play_pool_path, self_play_pool_records)
                                    snapshot_note = " sp_pool"
                                except OSError:
                                    snapshot_note = " sp_pool_err"
                                if self_play_using_latest and sync_self_play_snapshot(model_train, plan["base_model_opponent"]):
                                    snapshot_note = f"{snapshot_note}+sync"
                            elif sync_self_play_snapshot(model_train, plan["base_model_opponent"]):
                                snapshot_note = " sp_sync"

                        if episodes_completed % checkpoint_interval == 0:
                            checkpoint_path = save_checkpoint(model_train, plan["episode_index"] + 1)
                            last_saved_policy_path = checkpoint_path
                            save_training_state(
                                state_path,
                                policy_path=checkpoint_path,
                                episodes_completed=episodes_completed,
                                win_count=win_count,
                                loss_count=loss_count,
                                draw_count=draw_count,
                                epsilon=epsilon_value,
                            )

                        recent_win_count = sum(1 for item in recent_results if item == "WIN")
                        recent_loss_count = sum(1 for item in recent_results if item == "LOSS")
                        recent_draw_count = sum(1 for item in recent_results if item == "DRAW")
                        recent_count = len(recent_results)
                        recent_win_rate = (
                            (recent_win_count / recent_count) * 100.0 if recent_count > 0 else 0.0
                        )
                        recent_avg_reward = (
                            sum(recent_episode_rewards) / float(len(recent_episode_rewards))
                            if len(recent_episode_rewards) > 0
                            else 0.0
                        )
                        elapsed_sec = max(time.perf_counter() - started_at, 1e-9)
                        elapsed_total_sec = int(elapsed_sec)
                        elapsed_h = elapsed_total_sec // 3600
                        elapsed_m = (elapsed_total_sec % 3600) // 60
                        elapsed_s = elapsed_total_sec % 60
                        elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"
                        trained_episodes = max(episodes_completed - start_episode, 0)
                        episodes_per_sec = trained_episodes / elapsed_sec if trained_episodes > 0 else 0.0
                        progress_pct = (
                            (episodes_completed / total_episodes) * 100.0 if total_episodes > 0 else 100.0
                        )
                        epsilon_display = (
                            f"{float(epsilon_value):.6f}" if epsilon_value is not None else "N/A"
                        )
                        entropy_value = getattr(model_train, "entropy_coefficient", None)
                        entropy_display = (
                            f"{float(entropy_value):.4f}" if entropy_value is not None else "N/A"
                        )
                        approx_kl_value = getattr(model_train, "last_approx_kl", None)
                        approx_kl_display = (
                            f"{float(approx_kl_value):.4f}" if approx_kl_value is not None else "N/A"
                        )
                        status_line = (
                            f"[{episodes_completed}/{total_episodes} {progress_pct:5.1f}%] "
                            f"side={plan['train_side']} "
                            f"opp={plan['episode_opponent_label']} "
                            f"eps={epsilon_display} "
                            f"ent={entropy_display} "
                            f"kl={approx_kl_display} "
                            f"r={episode_reward:6.2f} r{log_interval}={recent_avg_reward:6.2f} "
                            f"recent{recent_count} W/L/D={recent_win_count}/{recent_loss_count}/{recent_draw_count} "
                            f"win={recent_win_rate:4.1f}% "
                            f"last={result} {p1_score}:{p2_score} steps={episode_steps}{timeout_note}{snapshot_note} "
                            f"t={elapsed_str} {episodes_per_sec:.2f}epi/s"
                        )
                        term_width = max(shutil.get_terminal_size(fallback=(120, 24)).columns, 40)
                        if len(status_line) >= term_width:
                            if term_width > 4:
                                status_line = status_line[:term_width - 4] + "..."
                            else:
                                status_line = status_line[:term_width - 1]
                        print(f"\r\033[2K{status_line}", end="", flush=True)

                    epi_idx = batch_end

        except KeyboardInterrupt:
            model_train.save()
            last_saved_policy_path = get_primary_policy_path(model_train)
            epsilon_value = getattr(model_train, "epsilon", None)
            save_training_state(
                state_path,
                policy_path=last_saved_policy_path,
                episodes_completed=episodes_completed,
                win_count=win_count,
                loss_count=loss_count,
                draw_count=draw_count,
                epsilon=epsilon_value,
            )
            print(
                f"[Interrupt] saved policy={last_saved_policy_path} "
                f"episode={episodes_completed} epsilon={epsilon_value}"
            )
            return
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            if worker_snapshot_path != "":
                worker_critic_snapshot_path = get_secondary_policy_path(
                    model_train, worker_snapshot_path
                )
                for file_path in (worker_snapshot_path, worker_critic_snapshot_path):
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
            if episodes_completed > 0:
                print()

        model_train.save()
        last_saved_policy_path = get_primary_policy_path(model_train)
        epsilon_value = getattr(model_train, "epsilon", None)
        save_training_state(
            state_path,
            policy_path=last_saved_policy_path,
            episodes_completed=episodes_completed,
            win_count=win_count,
            loss_count=loss_count,
            draw_count=draw_count,
            epsilon=epsilon_value,
        )
        return

    try:
        if hasattr(model_train, "set_training_progress") and callable(model_train.set_training_progress):
            model_train.set_training_progress(start_episode)
        for epi_idx in range(start_episode, total_episodes):
            desired_train_opponent_spec = active_train_opponent_spec
            if curriculum_enabled:
                desired_train_opponent_spec = resolve_curriculum_opponent(
                    curriculum_schedule,
                    epi_idx,
                    default_train_opponent_spec,
                )

            if desired_train_opponent_spec != active_train_opponent_spec:
                active_train_opponent_spec = desired_train_opponent_spec
                model_opponent, active_train_opponent_label = create_training_opponent(
                    conf,
                    model_train,
                    active_train_opponent_spec,
                )
                self_play_mode = str(active_train_opponent_spec).strip().lower() == "self"
                self_play_runtime_label = active_train_opponent_label
                self_play_using_latest = True
                last_self_play_resample_episode = -1
                print()
                print(
                    f"[Curriculum] episode={epi_idx + 1} opponent={active_train_opponent_label}",
                    flush=True,
                )
                if self_play_mode and not isinstance(model_opponent, str):
                    if self_play_pool_enabled:
                        self_play_runtime_label, self_play_using_latest = select_self_play_pool_opponent(
                            model_train,
                            model_opponent,
                            self_play_pool_records,
                            self_play_pool_latest_prob,
                            self_play_pool_warmup_episode,
                            episodes_completed,
                        )
                        last_self_play_resample_episode = epi_idx
                        print(
                            f"[SelfPlay] opponent={self_play_runtime_label} "
                            f"snapshot_interval={self_play_snapshot_interval}"
                        )
                    elif sync_self_play_snapshot(model_train, model_opponent):
                        print(
                            f"[SelfPlay] snapshot initialized interval={self_play_snapshot_interval}"
                        )

            episode_train_side = resolve_training_side(
                conf.train_side,
                train_side_mode,
                epi_idx - start_episode,
            )
            model_train.conf.train_side = episode_train_side
            episode_train_player_key = "p1" if episode_train_side == "1p" else "p2"

            force_player2_serve = bool((epi_idx - start_episode) % 2 == 1)
            episode_opponent = model_opponent
            episode_opponent_label = active_train_opponent_label
            if self_play_mode and not isinstance(model_opponent, str):
                if random.random() < self_play_rule_mix_prob:
                    episode_opponent = self_play_mix_rule_model
                    episode_opponent_label = f"mix:{self_play_mix_rule_label}"
                else:
                    if self_play_pool_enabled:
                        should_resample_self_play = (
                            last_self_play_resample_episode < 0
                            or (epi_idx - last_self_play_resample_episode) >= self_play_pool_resample_interval
                        )
                        if should_resample_self_play:
                            self_play_runtime_label, self_play_using_latest = select_self_play_pool_opponent(
                                model_train,
                                model_opponent,
                                self_play_pool_records,
                                self_play_pool_latest_prob,
                                self_play_pool_warmup_episode,
                                episodes_completed,
                            )
                            last_self_play_resample_episode = epi_idx
                        episode_opponent_label = self_play_runtime_label
                    else:
                        episode_opponent_label = "self:latest"
            # - Set the Environment
            set_training_match(
                env,
                episode_train_side,
                model_train,
                episode_opponent,
                force_player2_serve=force_player2_serve,
            )

            # - Get Initial State
            state_mat = env.get_state(player=episode_train_side)

            # - Run an Episode
            episode_reward = 0.0
            episode_steps = 0
            while True:
                # - Get Transition by Action Selection and Environment Run
                transition, state_next_mat = model_train.get_transition(env, state_mat)
                reward_step = float(transition[4])
                episode_reward += reward_step
                episode_steps += 1

                # - Update Policy by Transition
                model_train.update(transition)
                env = model_train.env

                # - Update State
                state_mat = state_next_mat

                # - Check Terminate Condition
                done = transition[-2]
                if done:
                    score = transition[-1]
                    p1_score = int(score.get("p1", 0))
                    p2_score = int(score.get("p2", 0))
                    if p1_score == p2_score:
                        result = "DRAW"
                        draw_count += 1
                    else:
                        winner_key = "p1" if p1_score > p2_score else "p2"
                        if winner_key == episode_train_player_key:
                            result = "WIN"
                            win_count += 1
                        else:
                            result = "LOSS"
                            loss_count += 1

                    episodes_completed = epi_idx + 1
                    if hasattr(model_train, "set_training_progress") and callable(model_train.set_training_progress):
                        model_train.set_training_progress(episodes_completed)
                    if ".dqn." in train_model_module_name and hasattr(model_train, "update_epsilon") and callable(model_train.update_epsilon):
                        model_train.update_epsilon()
                    epsilon_value = getattr(model_train, "epsilon", None)
                    event_info = score.get("events") or {}
                    timeout_note = " timeout" if event_info.get("timeout", False) else ""
                    recent_results.append(result)
                    recent_episode_rewards.append(float(episode_reward))
                    snapshot_note = ""

                    if (
                        self_play_mode
                        and not isinstance(model_opponent, str)
                        and episodes_completed % self_play_snapshot_interval == 0
                    ):
                        if self_play_pool_enabled:
                            try:
                                snapshot_record = save_self_play_snapshot(
                                    model_train,
                                    episodes_completed,
                                )
                                self_play_pool_records = register_self_play_pool_snapshot(
                                    self_play_pool_records,
                                    snapshot_record,
                                    self_play_pool_size,
                                )
                                save_self_play_pool(self_play_pool_path, self_play_pool_records)
                                snapshot_note = " sp_pool"
                            except OSError:
                                snapshot_note = " sp_pool_err"
                            if self_play_using_latest and sync_self_play_snapshot(model_train, model_opponent):
                                snapshot_note = f"{snapshot_note}+sync"
                        elif sync_self_play_snapshot(model_train, model_opponent):
                            snapshot_note = " sp_sync"

                    if episodes_completed % checkpoint_interval == 0:
                        checkpoint_path = save_checkpoint(model_train, epi_idx + 1)
                        last_saved_policy_path = checkpoint_path
                        save_training_state(
                            state_path,
                            policy_path=checkpoint_path,
                            episodes_completed=episodes_completed,
                            win_count=win_count,
                            loss_count=loss_count,
                            draw_count=draw_count,
                            epsilon=epsilon_value,
                        )

                    recent_win_count = sum(1 for item in recent_results if item == "WIN")
                    recent_loss_count = sum(1 for item in recent_results if item == "LOSS")
                    recent_draw_count = sum(1 for item in recent_results if item == "DRAW")
                    recent_count = len(recent_results)
                    recent_win_rate = (
                        (recent_win_count / recent_count) * 100.0 if recent_count > 0 else 0.0
                    )
                    recent_avg_reward = (
                        sum(recent_episode_rewards) / float(len(recent_episode_rewards))
                        if len(recent_episode_rewards) > 0
                        else 0.0
                    )
                    elapsed_sec = max(time.perf_counter() - started_at, 1e-9)
                    elapsed_total_sec = int(elapsed_sec)
                    elapsed_h = elapsed_total_sec // 3600
                    elapsed_m = (elapsed_total_sec % 3600) // 60
                    elapsed_s = elapsed_total_sec % 60
                    elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"
                    trained_episodes = max(episodes_completed - start_episode, 0)
                    episodes_per_sec = trained_episodes / elapsed_sec if trained_episodes > 0 else 0.0
                    progress_pct = (
                        (episodes_completed / total_episodes) * 100.0 if total_episodes > 0 else 100.0
                    )
                    epsilon_display = (
                        f"{float(epsilon_value):.6f}" if epsilon_value is not None else "N/A"
                    )
                    entropy_value = getattr(model_train, "entropy_coefficient", None)
                    entropy_display = (
                        f"{float(entropy_value):.4f}" if entropy_value is not None else "N/A"
                    )
                    approx_kl_value = getattr(model_train, "last_approx_kl", None)
                    approx_kl_display = (
                        f"{float(approx_kl_value):.4f}" if approx_kl_value is not None else "N/A"
                    )
                    status_line = (
                        f"[{episodes_completed}/{total_episodes} {progress_pct:5.1f}%] "
                        f"side={episode_train_side} "
                        f"opp={episode_opponent_label} "
                        f"eps={epsilon_display} "
                        f"ent={entropy_display} "
                        f"kl={approx_kl_display} "
                        f"r={episode_reward:6.2f} r{log_interval}={recent_avg_reward:6.2f} "
                        f"recent{recent_count} W/L/D={recent_win_count}/{recent_loss_count}/{recent_draw_count} "
                        f"win={recent_win_rate:4.1f}% "
                        f"last={result} {p1_score}:{p2_score} steps={episode_steps}{timeout_note}{snapshot_note} "
                        f"t={elapsed_str} {episodes_per_sec:.2f}epi/s"
                    )
                    term_width = max(shutil.get_terminal_size(fallback=(120, 24)).columns, 40)
                    if len(status_line) >= term_width:
                        if term_width > 4:
                            status_line = status_line[:term_width - 4] + "..."
                        else:
                            status_line = status_line[:term_width - 1]
                    print(f"\r\033[2K{status_line}", end="", flush=True)
                    break
    except KeyboardInterrupt:
        model_train.save()
        last_saved_policy_path = get_primary_policy_path(model_train)
        epsilon_value = getattr(model_train, "epsilon", None)
        save_training_state(
            state_path,
            policy_path=last_saved_policy_path,
            episodes_completed=episodes_completed,
            win_count=win_count,
            loss_count=loss_count,
            draw_count=draw_count,
            epsilon=epsilon_value,
        )
        print(
            f"[Interrupt] saved policy={last_saved_policy_path} "
            f"episode={episodes_completed} epsilon={epsilon_value}"
        )
        return
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if episodes_completed > 0:
            print()

    """====================================================================================================
    ## Save Trained Policy at the End of Episode
    ===================================================================================================="""
    model_train.save()
    last_saved_policy_path = get_primary_policy_path(model_train)
    epsilon_value = getattr(model_train, "epsilon", None)
    save_training_state(
        state_path,
        policy_path=last_saved_policy_path,
        episodes_completed=episodes_completed,
        win_count=win_count,
        loss_count=loss_count,
        draw_count=draw_count,
        epsilon=epsilon_value,
    )


def create_environment_instance(conf):
    """====================================================================================================
    ## Creation of Environment Instance
    ===================================================================================================="""
    # - Load Configuration
    RENDER_MODE = "log"
    TARGET_SCORE = conf.target_score_train
    SEED = conf.seed

    # - Create Envionment Instance
    env = create_training_env(
        target_score=TARGET_SCORE,
        seed=SEED,
    )

    # - Return Environment Instance
    return env


def load_training_model(conf):
    """====================================================================================================
    ## Loading Training Model from train_* Arguments
    ===================================================================================================="""
    algorithm_name = str(getattr(conf, "train_algorithm", "") or "").strip().lower()
    policy_name = getattr(conf, "train_policy", None)

    if algorithm_name in ("", "rule", "human"):
        raise ValueError(
            f"invalid train_algorithm for learning: {algorithm_name}"
        )

    model = _20_model.create_model(
        conf,
        algorithm_name=algorithm_name,
        policy_name_for_play=policy_name,
    )
    return model


if __name__ == "__main__":
    pass

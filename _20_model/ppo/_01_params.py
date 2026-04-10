def get_train_params():
    """====================================================================================================
    ## Hyperparameter Setting for training
    ===================================================================================================="""
    TRAIN_PARAMS = {
        # Learning Rate
        "learning_rate_actor": 2.5e-4,
        "learning_rate_critic": 3e-4,

        # Discount Factor
        "gamma": 0.995,

        # PPO Update Parameters
        "clip_epsilon": 0.2,
        "value_clip_epsilon": 0.2,
        "vf_coefficient": 0.5,
        "update_epochs": 8,
        "minibatch_size": 64,
        "target_kl": 0.03,
        "entropy_coefficient": 0.01,
        "entropy_coefficient_end": 0.0025,
        "entropy_decay_episodes": 120000,
        "max_grad_norm": 0.5,
        "advantage_epsilon": 1e-8,
        "gae_lambda": 0.95,

        # Neural Network Architecture Parameters
        "hidden_dim": 128,
        "hidden_layer_count": 3,

        # Maximum Steps per Episode
        "max_steps_per_episode": 30*45,

        # Training Schedule Defaults
        "curriculum_enabled": True,
        "curriculum_schedule": "0=rule,30000=self",
        "train_side_mode": "alternate",
        "train_num_workers": 1,
        "self_play_snapshot_interval": 2000,
        "self_play_pool_enabled": True,
        "self_play_pool_size": 24,
        "self_play_pool_latest_prob": 0.35,
        "self_play_pool_resample_interval": 25,
        "self_play_pool_warmup_episode": 10000,
        "self_play_rule_mix_prob": 0.30,

        # Progress Display Options
        "show_progress": True,
        "progress_interval": 50,
    }
    return TRAIN_PARAMS


def get_play_params():
    """====================================================================================================
    ## Hyperparameter Setting for Playing
    ===================================================================================================="""
    PLAY_PARAMS = {
        "max_steps": 30*60*60,
    }
    return PLAY_PARAMS

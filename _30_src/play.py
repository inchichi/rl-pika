# Import Required Internal Modules
from pathlib import Path

import _00_environment
import _20_model


def run(conf):
    """====================================================================================================
    ## Creation of Environment Instance and Loading model for Each Player
    ===================================================================================================="""
    # - Create Envionment Instance
    env = create_invironment_instance(conf)

    # - Load the model for each player
    model_1p = load_model(conf, player='1p')
    model_2p = load_model(conf, player='2p')

    """====================================================================================================
    ## Playing Episode
    ===================================================================================================="""
    # - Set Environment with Selected Algorithm and Policy for Each Player
    env.set(player1=model_1p, player2=model_2p, random_serve=conf.random_serve)

    # - Wait for 's' key to Start Episode
    env.wait_key_for_start(key=ord('s'))

    # - Run Episode
    while True:
        # - Get Play Result for Each Step
        play_result = env.get_play_result()
        done = play_result['done']
        score = play_result['score']

        # - Consume Viewer Command
        command = env.consume_viewer_command()

        # - Check Terminate Condition
        if command == "quit":
            break

        if done is True:
            # - Print Winner and Final Score
            print("winner: {}"
                  .format('player1' if score['p1'] > score['p2'] else 'player2'))
            print(f"final score: {score['p1']}:{score['p2']}")

            # - Escape Loop to Terminate Episode
            break

    # - Terminate Episode and Close Environment
    env.close()


def create_invironment_instance(conf):
    """====================================================================================================
    ## Creation of Environment Instance
    ===================================================================================================="""

    # - Load Configuration
    RENDER_MODE = "human"
    TARGET_SCORE = conf.target_score_play
    SEED = conf.seed

    # - Create Envionment Instance
    env = _00_environment.Env(render_mode=RENDER_MODE,
                         target_score=TARGET_SCORE,
                         seed=SEED)

    # - Return Environment Instance
    return env


def resolve_policy_name_for_play(conf, algorithm, policy_name):
    if policy_name is None:
        return None

    normalized_policy_name = str(policy_name).strip()
    if normalized_policy_name == "":
        return None
    if "/" in normalized_policy_name or "\\" in normalized_policy_name:
        return normalized_policy_name

    try:
        policy_dir = Path(_20_model.get_model_policy_dir(conf, algorithm))
    except Exception:
        return normalized_policy_name

    candidate_matches = []
    for pattern in (
        f"{normalized_policy_name}.pth",
        f"{normalized_policy_name}.json",
        f"{normalized_policy_name}.pkl",
        f"**/{normalized_policy_name}.pth",
        f"**/{normalized_policy_name}.json",
        f"**/{normalized_policy_name}.pkl",
    ):
        candidate_matches.extend(policy_dir.glob(pattern))

    deduped_matches = []
    seen_paths = set()
    for match_path in candidate_matches:
        resolved = str(match_path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        deduped_matches.append(match_path)

    if len(deduped_matches) == 1:
        matched_path = deduped_matches[0]
        relative_without_suffix = matched_path.relative_to(policy_dir).with_suffix("")
        return relative_without_suffix.as_posix()

    return normalized_policy_name


def load_model(conf, player):
    """====================================================================================================
    ## Loading Policy for Each Player
    ===================================================================================================="""
    # - Check Algorithm and Policy Name for Each Player
    ALGORITHM = conf.algorithm_1p if player == '1p' else conf.algorithm_2p
    POLICY_NAME = conf.policy_1p if player == '1p' else conf.policy_2p
    POLICY_NAME = resolve_policy_name_for_play(conf, ALGORITHM, POLICY_NAME)

    # - Load Selected Policy for Each Player
    if ALGORITHM == 'human':
        model = 'HUMAN'

    elif ALGORITHM == 'rule':
        model = 'RULE'

    else:
        model = _20_model.create_model(
            conf,
            algorithm_name=ALGORITHM,
            policy_name_for_play=POLICY_NAME,
        )

    # - Return Loaded Model for Each Player
    return model

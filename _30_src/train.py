# Import Required External Libraries
from tqdm import tqdm

# Import Required Internal Modules
import _00_environment
import _20_model
from _30_src.play import load_model


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
    model_train = load_model(conf, player=conf.train_side)
    model_opponent = load_model(
        conf, player='1p' if conf.train_side == '2p' else '2p')

    # - Run a number of Episodes for Training
    for epi_idx in tqdm(range(conf.num_episode), desc="Training Progress"):
        # - Set the Environment
        if conf.train_side == '1p':
            env.set(player1=model_train, player2=model_opponent,
                    random_serve=conf.random_serve, return_state=False)
        else:
            env.set(player1=model_opponent, player2=model_train,
                    random_serve=conf.random_serve, return_state=False)

        # - Get Initial State
        state_mat = env.get_state(player=conf.train_side)

        # - Run an Episode
        while True:
            # - Get Transition by Action Selection and Environment Run
            transition, state_next_mat = model_train.get_transition(env, state_mat)

            # - Update Policy by Transition
            model_train.update(transition)
            env = model_train.env

            # - Update State
            state_mat = state_next_mat

            # - Check Terminate Condition
            done = transition[4]
            if done:
                break

    """====================================================================================================
    ## Save Trained Policy at the End of Episode
    ===================================================================================================="""
    model_train.save()


def create_environment_instance(conf):
    """====================================================================================================
    ## Creation of Environment Instance
    ===================================================================================================="""
    # - Load Configuration
    RENDER_MODE = "log"
    TARGET_SCORE = conf.target_score_train
    SEED = conf.seed

    # - Create Envionment Instance
    env = _00_environment.Env(
        render_mode=RENDER_MODE,
        target_score=TARGET_SCORE,
        seed=SEED,
    )

    # - Return Environment Instance
    return env


def load_model(conf, player):
    """====================================================================================================
    ## Loading Policy for Each Player
    ===================================================================================================="""
    # - Check Algorithm and Policy Name for Each Player
    ALGORITHM = conf.algorithm_1p if player == '1p' else conf.algorithm_2p
    POLICY_NAME = conf.policy_1p if player == '1p' else conf.policy_2p

    # - Load Selected Policy for Each Player
    if ALGORITHM == 'human':
        model = 'HUMAN'

    elif ALGORITHM == 'rule':
        model = 'RULE'

    elif ALGORITHM == 'qlearning':
        model = _20_model.qlearning._00_model.Qlearning(
            conf, policy_name_for_play=POLICY_NAME)

    elif ALGORITHM == 'sarsa':
        model = _20_model.sarsa._00_model.Sarsa(
            conf, policy_name_for_play=POLICY_NAME)

    elif ALGORITHM == 'dqn':
        model = _20_model.dqn._00_model.Dqn(
            conf, policy_name_for_play=POLICY_NAME)

    # - Return Loaded Model for Each Player
    return model


if __name__ == "__main__":
    pass

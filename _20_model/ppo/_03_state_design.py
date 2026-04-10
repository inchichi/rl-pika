# Import require internal Packages
from _00_environment.constants import BALL_TOUCHING_GROUND_Y_COORD
from _00_environment.constants import GROUND_WIDTH


ACTION_GROUP_CODE = {
    "idle": 0,
    "forward": 1,
    "backward": 2,
    "jump": 3,
    "jump_forward": 4,
    "jump_backward": 5,
    "dive_forward": 6,
    "dive_backward": 7,
    "spike": 8,
}


def normalize_minmax(value, minimum_value, maximum_value):
    """====================================================================================================
    ## Min-Max Normalization Wrapper
    ===================================================================================================="""
    if maximum_value <= minimum_value:
        return 0.0

    normalized_value = (float(value) - float(minimum_value)) / \
        (float(maximum_value) - float(minimum_value))

    if normalized_value < 0.0:
        return 0.0
    if normalized_value > 1.0:
        return 1.0
    return float(normalized_value)


def normalize_action_group(action_name):
    normalized_action_name = str(action_name).strip().lower()

    if normalized_action_name.startswith("spike_"):
        normalized_action_name = "spike"
    elif normalized_action_name not in ACTION_GROUP_CODE:
        normalized_action_name = "idle"

    action_group = int(ACTION_GROUP_CODE[normalized_action_name])
    return normalize_minmax(action_group, 0, len(ACTION_GROUP_CODE) - 1)


def calculate_state_key(materials):
    """====================================================================================================
    ## Configuration for Action Group Mapping and Normalization
    ===================================================================================================="""
    velocity_min = -30
    velocity_max = 30

    raw = materials["raw"]
    materials = {
        "self_position": (raw["self"]["x"], raw["self"]["y"]),
        "self_action_name": raw["self"]["action_name"],
        "opponent_position": (raw["opponent"]["x"], raw["opponent"]["y"]),
        "opponent_action_name": raw["opponent"]["action_name"],
        "ball_position": (raw["ball"]["x"], raw["ball"]["y"]),
        "ball_velocity": (raw["ball"]["x_velocity"], raw["ball"]["y_velocity"]),
        "expected_landing_x": raw["ball"]["expected_landing_x"],
    }

    self_x, self_y = materials["self_position"]
    self_x = normalize_minmax(float(self_x), 0, GROUND_WIDTH - 1)
    self_y = normalize_minmax(float(self_y), 0, BALL_TOUCHING_GROUND_Y_COORD)

    self_action_group = normalize_action_group(materials["self_action_name"])

    opponent_x, opponent_y = materials["opponent_position"]
    opponent_x = normalize_minmax(float(opponent_x), 0, GROUND_WIDTH - 1)
    opponent_y = normalize_minmax(
        float(opponent_y), 0, BALL_TOUCHING_GROUND_Y_COORD)

    opponent_action_group = normalize_action_group(
        materials["opponent_action_name"])

    ball_x, ball_y = materials["ball_position"]
    ball_x = normalize_minmax(float(ball_x), 0, GROUND_WIDTH - 1)
    ball_y = normalize_minmax(float(ball_y), 0, BALL_TOUCHING_GROUND_Y_COORD)

    ball_velocity_x, ball_velocity_y = materials["ball_velocity"]
    ball_velocity_x = normalize_minmax(
        float(ball_velocity_x), velocity_min, velocity_max)
    ball_velocity_y = normalize_minmax(
        float(ball_velocity_y), velocity_min, velocity_max)

    landing_x = float(materials["expected_landing_x"])
    landing_x = normalize_minmax(landing_x, 0, GROUND_WIDTH - 1)

    DESIGNED_STATE_VECTOR = [
        self_x,
        self_y,
        self_action_group,
        opponent_x,
        opponent_y,
        opponent_action_group,
        ball_x,
        ball_y,
        ball_velocity_x,
        ball_velocity_y,
        landing_x,
    ]
    return DESIGNED_STATE_VECTOR


def get_state_dim():
    """====================================================================================================
    ## Get the Dimension of Designed State Vector
    ===================================================================================================="""
    return 11

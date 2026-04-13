# Import Required Internal Libraries
from _00_environment.constants import BALL_TOUCHING_GROUND_Y_COORD
from _00_environment.constants import GROUND_HALF_WIDTH


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


def clip_value(value, minimum_value, maximum_value):
    if value < minimum_value:
        return float(minimum_value)
    if value > maximum_value:
        return float(maximum_value)
    return float(value)


def select_mat_for_reward(materials):
    """====================================================================================================
    ## Load materials for reward design
    ===================================================================================================="""
    self_position = materials["self_position"]
    opponent_position = materials["opponent_position"]
    ball_position = materials["ball_position"]
    ball_velocity = materials.get("ball_velocity", (0.0, 0.0))
    self_action_name = str(materials["self_action_name"])
    opponent_action_name = str(materials["opponent_action_name"])
    rally_total_frames_until_point = float(
        materials["rally_total_frames_until_point"])
    point_scored = int(materials["point_result"]["scored"])
    point_lost = int(materials["point_result"]["lost"])
    self_spike_used = int(self_action_name.startswith("spike_"))
    self_dive_used = int(self_action_name.startswith("dive_"))
    opponent_dive_used = int(opponent_action_name.startswith("dive_"))
    opponent_spike_used = int(opponent_action_name.startswith("spike_"))
    self_aggressive_attack = int(
        self_action_name in (
            "spike_fast_up",
            "spike_fast_flat",
            "spike_fast_down",
            "spike_soft_down",
        )
    )
    touch_signal_value = materials.get("self_touched_ball", None)
    touch_signal_available = int(touch_signal_value is not None)
    self_touched_ball = 0
    if touch_signal_available > 0:
        self_touched_ball = int(float(touch_signal_value) > 0.5)
    self_dive_missed = int(
        touch_signal_available > 0 and self_dive_used > 0 and self_touched_ball == 0
    )
    expected_landing_x = float(materials.get("expected_landing_x", ball_position[0]))
    match_won = int(materials["match_result"]["won"] > 0.5)
    self_net_distance = abs(self_position[0] - GROUND_HALF_WIDTH)
    opponent_net_distance = abs(opponent_position[0] - GROUND_HALF_WIDTH)
    ball_to_self_distance = abs(ball_position[0] - self_position[0])
    ball_to_opponent_distance = abs(ball_position[0] - opponent_position[0])
    ball_on_self_side = int(float(ball_position[0]) <= float(GROUND_HALF_WIDTH))
    predicted_self_landing = int(float(expected_landing_x) <= float(GROUND_HALF_WIDTH))
    landing_delta_x = float(expected_landing_x) - float(self_position[0])
    landing_delta_abs = abs(landing_delta_x)
    landing_move_threshold = float(GROUND_HALF_WIDTH) * 0.10
    ready_wait_threshold = float(GROUND_HALF_WIDTH) * 0.13
    micro_adjust_threshold = float(GROUND_HALF_WIDTH) * 0.05

    ball_velocity_x = 0.0
    ball_velocity_y = 0.0
    if isinstance(ball_velocity, (list, tuple)) and len(ball_velocity) >= 2:
        ball_velocity_x = float(ball_velocity[0])
        ball_velocity_y = float(ball_velocity[1])

    self_forward_commit = int(
        self_action_name in ("forward", "jump_forward", "dive_forward")
    )
    self_backward_recover = int(
        self_action_name in ("backward", "jump_backward", "dive_backward")
    )
    self_jump_used = int(
        self_action_name in ("jump", "jump_forward", "jump_backward")
    )
    self_grounded_move = int(self_action_name in ("forward", "backward"))
    self_backward_jump = int(self_action_name == "jump_backward")
    self_backward_dive = int(self_action_name == "dive_backward")

    standby_target_x = float(GROUND_HALF_WIDTH) * 0.50
    intercept_target_x = clip_value(expected_landing_x, 0.0, float(GROUND_HALF_WIDTH))
    anticipation_weight = 0.0
    if ball_on_self_side > 0:
        anticipation_weight = 1.0
    elif predicted_self_landing > 0:
        anticipation_weight = 0.72
    elif opponent_spike_used > 0:
        anticipation_weight = 0.30
    else:
        anticipation_weight = 0.15
    position_target_x = (
        (1.0 - anticipation_weight) * standby_target_x
        + anticipation_weight * intercept_target_x
    )

    position_error = abs(float(self_position[0]) - float(position_target_x))
    self_overfront = int(
        ball_on_self_side < 1 and float(self_position[0]) > float(GROUND_HALF_WIDTH) * 0.78
    )
    self_overback = int(
        ball_on_self_side < 1 and float(self_position[0]) < float(GROUND_HALF_WIDTH) * 0.22
    )
    ball_on_opponent_side = int(ball_on_self_side < 1)
    self_front_zone = int(float(self_position[0]) > float(GROUND_HALF_WIDTH) * 0.75)
    self_overchase = int(
        ball_on_opponent_side > 0 and self_front_zone > 0 and self_forward_commit > 0
    )
    self_recover = int(
        ball_on_opponent_side > 0 and self_front_zone > 0 and self_backward_recover > 0
    )
    defensive_pressure = int(
        predicted_self_landing > 0
        or ball_on_self_side > 0
        or opponent_spike_used > 0
    )
    defensive_touch = int(
        self_touched_ball > 0
        and (
            defensive_pressure > 0
            or self_dive_used > 0
            or self_backward_recover > 0
        )
    )
    stable_defense_touch = int(defensive_touch > 0 and self_spike_used < 1)
    neutral_keep_play_touch = int(
        self_touched_ball > 0
        and defensive_pressure < 1
        and self_spike_used < 1
        and self_dive_used < 1
    )

    landing_need_backward = int(
        predicted_self_landing > 0 and landing_delta_x < -landing_move_threshold
    )
    landing_need_forward = int(
        predicted_self_landing > 0 and landing_delta_x > landing_move_threshold
    )
    timely_backward_move = int(landing_need_backward > 0 and self_backward_recover > 0)
    timely_forward_move = int(landing_need_forward > 0 and self_forward_commit > 0)
    wrong_way_move = int(
        (landing_need_backward > 0 and self_forward_commit > 0)
        or (landing_need_forward > 0 and self_backward_recover > 0)
    )
    front_defense_target = int(
        landing_need_forward > 0
        and predicted_self_landing > 0
        and float(expected_landing_x) >= float(GROUND_HALF_WIDTH) * 0.68
    )

    ball_drop_factor = normalize_minmax(
        float(ball_position[1]),
        0.0,
        BALL_TOUCHING_GROUND_Y_COORD,
    )
    ball_descend_factor = normalize_minmax(ball_velocity_y, -18.0, 18.0)
    landing_urgency = clip_value(
        (0.55 * ball_drop_factor) + (0.45 * ball_descend_factor),
        0.0,
        1.0,
    )
    front_low_ball = int(
        float(ball_position[1]) >= float(BALL_TOUCHING_GROUND_Y_COORD) * 0.42
    )
    front_defense_pressure = int(
        front_defense_target > 0
        and (
            landing_urgency > 0.48
            or front_low_ball > 0
            or ball_on_self_side > 0
        )
    )
    front_save_move = int(
        front_defense_pressure > 0
        and self_action_name in ("forward", "jump_forward", "dive_forward")
    )
    front_save_dive = int(
        front_defense_pressure > 0 and self_action_name == "dive_forward"
    )
    hesitated_front_defense = int(
        front_defense_pressure > 0
        and self_action_name in ("idle", "backward", "jump_backward", "dive_backward")
    )
    back_defense_target = int(
        landing_need_backward > 0
        and predicted_self_landing > 0
        and float(expected_landing_x) <= float(GROUND_HALF_WIDTH) * 0.36
    )
    back_defense_pressure = int(
        back_defense_target > 0
        and (
            landing_urgency > 0.46
            or ball_on_self_side > 0
            or opponent_spike_used > 0
        )
    )
    wall_corner_target = int(
        predicted_self_landing > 0
        and float(expected_landing_x) <= float(GROUND_HALF_WIDTH) * 0.16
    )
    serve_setup_like = int(
        ball_on_self_side > 0
        and float(self_position[0]) <= float(GROUND_HALF_WIDTH) * 0.18
        and float(expected_landing_x) <= float(GROUND_HALF_WIDTH) * 0.16
        and ball_to_self_distance <= float(GROUND_HALF_WIDTH) * 0.07
        and opponent_spike_used < 1
        and landing_urgency < 0.58
    )
    wall_ball_approaching = int(
        float(ball_position[0]) <= float(GROUND_HALF_WIDTH) * 0.24
        and ball_velocity_y > 0.0
    )
    wall_bounce_pressure = int(
        wall_corner_target > 0
        and serve_setup_like < 1
        and (wall_ball_approaching > 0 or landing_urgency > 0.42)
    )
    back_save_move = int(
        back_defense_pressure > 0
        and self_action_name in ("backward", "jump_backward", "dive_backward")
    )
    back_save_dive = int(
        back_defense_pressure > 0
        and self_action_name == "dive_backward"
        and landing_urgency > 0.62
    )
    hesitated_back_defense = int(
        back_defense_pressure > 0
        and self_action_name in ("idle", "forward", "jump_forward", "dive_forward")
    )
    wall_bounce_read_move = int(
        wall_bounce_pressure > 0
        and self_action_name in ("backward", "jump_backward", "dive_backward")
    )
    wall_bounce_read_dive = int(
        wall_bounce_pressure > 0 and self_action_name == "dive_backward"
    )
    wall_bounce_hesitation = int(
        wall_bounce_pressure > 0
        and self_action_name in ("idle", "forward", "jump_forward", "dive_forward")
    )
    premature_back_jump = int(
        self_backward_jump > 0
        and landing_need_backward < 1
        and ball_on_self_side < 1
        and landing_urgency < 0.65
    )
    emergency_back_dive = int(
        self_backward_dive > 0
        and landing_need_backward > 0
        and landing_urgency > 0.72
    )
    urgent_dive_context = int(
        front_save_dive > 0
        or back_save_dive > 0
        or wall_bounce_read_dive > 0
        or emergency_back_dive > 0
    )
    unnecessary_back_dive = int(
        self_backward_dive > 0
        and back_defense_pressure < 1
        and wall_bounce_pressure < 1
        and landing_need_backward < 1
    )
    non_emergency_back_dive = int(
        self_backward_dive > 0
        and wall_bounce_read_dive < 1
        and emergency_back_dive < 1
    )
    passive_back_action = int(
        defensive_pressure < 1
        and self_action_name in ("backward", "jump_backward", "dive_backward")
    )
    serve_setup_back_dive = int(serve_setup_like > 0 and self_backward_dive > 0)
    serve_control_phase = int(
        ball_on_self_side > 0
        and float(self_position[0]) <= float(GROUND_HALF_WIDTH) * 0.24
        and ball_to_self_distance <= float(GROUND_HALF_WIDTH) * 0.14
        and float(ball_position[1]) <= float(BALL_TOUCHING_GROUND_Y_COORD) * 0.22
        and opponent_spike_used < 1
    )
    serve_control_back_action = int(
        serve_control_phase > 0
        and self_action_name in ("backward", "jump_backward", "dive_backward")
    )
    serve_control_back_dive = int(
        serve_control_phase > 0 and self_action_name == "dive_backward"
    )
    serve_control_ready_action = int(
        serve_control_phase > 0
        and self_action_name in ("idle", "forward", "jump_forward", "dive_forward")
    )
    controlled_ball_dive = int(
        self_dive_used > 0
        and ball_on_self_side > 0
        and ball_to_self_distance <= float(GROUND_HALF_WIDTH) * 0.10
        and landing_urgency < 0.64
        and abs(ball_velocity_x) <= 2.5
        and opponent_spike_used < 1
    )
    unnecessary_dive = int(
        self_dive_used > 0
        and urgent_dive_context < 1
        and (
            landing_urgency < 0.68
            or landing_delta_abs <= ready_wait_threshold
            or defensive_pressure < 1
        )
    )
    grounded_timed_move = int(
        self_grounded_move > 0
        and (
            (landing_need_backward > 0 and self_action_name == "backward")
            or (landing_need_forward > 0 and self_action_name == "forward")
        )
    )
    jump_timed_touch = int(
        self_jump_used > 0
        and self_touched_ball > 0
        and predicted_self_landing > 0
    )
    premature_jump = int(
        self_jump_used > 0
        and self_touched_ball < 1
        and (
            (predicted_self_landing < 1 and ball_on_self_side < 1)
            or landing_urgency < 0.58
        )
    )
    wrong_jump_direction = int(
        (landing_need_backward > 0 and self_action_name == "jump_forward")
        or (landing_need_forward > 0 and self_action_name == "jump_backward")
        or (
            self_action_name == "jump"
            and landing_delta_abs > landing_move_threshold
            and predicted_self_landing > 0
        )
    )
    well_prepared_idle = int(
        defensive_pressure > 0
        and self_action_name == "idle"
        and position_error <= ready_wait_threshold
        and landing_urgency > 0.42
    )
    settled_near_target = int(
        defensive_pressure > 0 and landing_delta_abs <= ready_wait_threshold
    )
    restless_ground_adjust = int(
        settled_near_target > 0
        and self_grounded_move > 0
        and landing_delta_abs <= micro_adjust_threshold
        and landing_urgency < 0.82
    )
    idle_under_pressure = int(
        defensive_pressure > 0
        and self_touched_ball < 1
        and self_action_name == "idle"
        and landing_urgency > 0.42
        and position_error > ready_wait_threshold
    )
    reckless_forward_under_pressure = int(
        defensive_pressure > 0
        and self_forward_commit > 0
        and landing_need_backward > 0
    )
    attacking_opportunity = int(
        defensive_pressure < 1
        and float(ball_position[1]) < float(BALL_TOUCHING_GROUND_Y_COORD) * 0.72
    )
    assertive_attack_action = int(
        self_aggressive_attack > 0 and attacking_opportunity > 0
    )
    assertive_attack_touch = int(
        self_touched_ball > 0
        and self_aggressive_attack > 0
        and attacking_opportunity > 0
    )

    SELECTED_MATARIALS = {
        "self_position": self_position,
        "opponent_position": opponent_position,
        "ball_position": ball_position,
        "self_action_name": self_action_name,
        "opponent_action_name": opponent_action_name,
        "self_net_distance": self_net_distance,
        "opponent_net_distance": opponent_net_distance,
        "ball_to_self_distance": ball_to_self_distance,
        "ball_to_opponent_distance": ball_to_opponent_distance,
        "point_scored": point_scored,
        "point_lost": point_lost,
        "self_spike_used": self_spike_used,
        "self_dive_used": self_dive_used,
        "self_aggressive_attack": self_aggressive_attack,
        "self_touched_ball": self_touched_ball,
        "self_dive_missed": self_dive_missed,
        "opponent_dive_used": opponent_dive_used,
        "opponent_spike_used": opponent_spike_used,
        "match_won": match_won,
        "ball_on_self_side": ball_on_self_side,
        "predicted_self_landing": predicted_self_landing,
        "landing_delta_x": landing_delta_x,
        "landing_delta_abs": landing_delta_abs,
        "landing_need_backward": landing_need_backward,
        "landing_need_forward": landing_need_forward,
        "timely_backward_move": timely_backward_move,
        "timely_forward_move": timely_forward_move,
        "front_defense_pressure": front_defense_pressure,
        "front_save_move": front_save_move,
        "front_save_dive": front_save_dive,
        "hesitated_front_defense": hesitated_front_defense,
        "back_defense_pressure": back_defense_pressure,
        "back_save_move": back_save_move,
        "back_save_dive": back_save_dive,
        "hesitated_back_defense": hesitated_back_defense,
        "wall_bounce_pressure": wall_bounce_pressure,
        "wall_bounce_read_move": wall_bounce_read_move,
        "wall_bounce_read_dive": wall_bounce_read_dive,
        "wall_bounce_hesitation": wall_bounce_hesitation,
        "grounded_timed_move": grounded_timed_move,
        "jump_timed_touch": jump_timed_touch,
        "premature_jump": premature_jump,
        "wrong_way_move": wrong_way_move,
        "wrong_jump_direction": wrong_jump_direction,
        "landing_urgency": landing_urgency,
        "premature_back_jump": premature_back_jump,
        "emergency_back_dive": emergency_back_dive,
        "unnecessary_back_dive": unnecessary_back_dive,
        "non_emergency_back_dive": non_emergency_back_dive,
        "passive_back_action": passive_back_action,
        "serve_setup_back_dive": serve_setup_back_dive,
        "serve_control_back_action": serve_control_back_action,
        "serve_control_back_dive": serve_control_back_dive,
        "serve_control_ready_action": serve_control_ready_action,
        "controlled_ball_dive": controlled_ball_dive,
        "unnecessary_dive": unnecessary_dive,
        "position_target_x": position_target_x,
        "position_error": position_error,
        "self_overfront": self_overfront,
        "self_overback": self_overback,
        "self_overchase": self_overchase,
        "self_recover": self_recover,
        "defensive_pressure": defensive_pressure,
        "defensive_touch": defensive_touch,
        "stable_defense_touch": stable_defense_touch,
        "neutral_keep_play_touch": neutral_keep_play_touch,
        "assertive_attack_action": assertive_attack_action,
        "assertive_attack_touch": assertive_attack_touch,
        "well_prepared_idle": well_prepared_idle,
        "settled_near_target": settled_near_target,
        "restless_ground_adjust": restless_ground_adjust,
        "idle_under_pressure": idle_under_pressure,
        "reckless_forward_under_pressure": reckless_forward_under_pressure,
        "rally_total_frames_until_point": rally_total_frames_until_point,
    }
    return SELECTED_MATARIALS


def calculate_reward(materials):
    """====================================================================================================
    ## Load Materials For Reward Design
    ===================================================================================================="""
    mat = select_mat_for_reward(materials)

    SCALE_POINT_SCORE_REWARD = 25.0
    SCALE_POINT_LOST_PENALTY = 25.0
    SCALE_SELF_DIVE_MISS_PENALTY = 0.25
    SCALE_SELF_OVERCHASE_PENALTY = 0.26
    SCALE_SELF_RECOVER_BONUS = 0.16
    SCALE_ANTICIPATION_BACKWARD_BONUS = 0.18
    SCALE_ANTICIPATION_FORWARD_BONUS = 0.15
    SCALE_FRONT_SAVE_MOVE_BONUS = 0.16
    SCALE_FRONT_SAVE_DIVE_BONUS = 0.11
    SCALE_HESITATION_FRONT_PENALTY = 0.16
    SCALE_BACK_SAVE_MOVE_BONUS = 0.18
    SCALE_BACK_SAVE_DIVE_BONUS = 0.11
    SCALE_HESITATION_BACK_PENALTY = 0.18
    SCALE_WALL_BOUNCE_READ_BONUS = 0.16
    SCALE_WALL_BOUNCE_DIVE_BONUS = 0.08
    SCALE_WALL_BOUNCE_HESITATION_PENALTY = 0.18
    SCALE_GROUNDED_TIMELY_MOVE_BONUS = 0.11
    SCALE_WRONG_DIRECTION_PENALTY = 0.14
    SCALE_PREMATURE_JUMP_PENALTY = 0.10
    SCALE_WRONG_JUMP_DIRECTION_PENALTY = 0.12
    SCALE_PREMATURE_BACK_JUMP_PENALTY = 0.08
    SCALE_EMERGENCY_BACK_DIVE_BONUS = 0.10
    SCALE_UNNECESSARY_BACK_DIVE_PENALTY = 0.12
    SCALE_NON_EMERGENCY_BACK_DIVE_PENALTY = 0.08
    SCALE_PASSIVE_BACK_ACTION_PENALTY = 0.08
    SCALE_SERVE_SETUP_BACK_DIVE_PENALTY = 0.20
    SCALE_SERVE_CONTROL_BACK_ACTION_PENALTY = 0.12
    SCALE_SERVE_CONTROL_BACK_DIVE_PENALTY = 0.36
    SCALE_SERVE_CONTROL_READY_BONUS = 0.08
    SCALE_CONTROLLED_BALL_DIVE_PENALTY = 0.24
    SCALE_UNNECESSARY_DIVE_PENALTY = 0.14
    SCALE_ASSERTIVE_ATTACK_ACTION_BONUS = 0.12
    SCALE_WELL_PREPARED_IDLE_BONUS = 0.05
    SCALE_RESTLESS_GROUND_ADJUST_PENALTY = 0.06
    SCALE_IDLE_UNDER_PRESSURE_PENALTY = 0.16
    SCALE_RECKLESS_FORWARD_PENALTY = 0.12
    SCALE_OPPONENT_DIVE_BONUS = 0.00
    SCALE_OPPONENT_SPIKE_PENALTY = 0.10
    SCALE_POSITION_ALIGNMENT_REWARD = 0.16
    SCALE_STANDBY_OVERFRONT_PENALTY = 0.12
    SCALE_STANDBY_OVERBACK_PENALTY = 0.08
    SCALE_IDLE_FAR_PENALTY = 0.14
    SCALE_SCORE_SPEED_BONUS = 0.45
    SCALE_QUICK_LOSS_PENALTY = 0.70
    RALLY_SPEED_REFERENCE_FRAMES = 180.0
    SCALE_MATCH_WIN_BONUS = 20.0

    reward = 0.0
    reward += SCALE_POINT_SCORE_REWARD * mat["point_scored"]
    reward -= SCALE_POINT_LOST_PENALTY * mat["point_lost"]
    reward -= SCALE_SELF_DIVE_MISS_PENALTY * mat["self_dive_missed"]
    reward -= SCALE_SELF_OVERCHASE_PENALTY * mat["self_overchase"]
    reward += SCALE_SELF_RECOVER_BONUS * mat["self_recover"]
    reward += (
        SCALE_ANTICIPATION_BACKWARD_BONUS
        * mat["timely_backward_move"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_ANTICIPATION_FORWARD_BONUS
        * mat["timely_forward_move"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_FRONT_SAVE_MOVE_BONUS
        * mat["front_save_move"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_FRONT_SAVE_DIVE_BONUS
        * mat["front_save_dive"]
        * (0.50 + (0.50 * mat["landing_urgency"]))
    )
    reward -= (
        SCALE_HESITATION_FRONT_PENALTY
        * mat["hesitated_front_defense"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_BACK_SAVE_MOVE_BONUS
        * mat["back_save_move"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_BACK_SAVE_DIVE_BONUS
        * mat["back_save_dive"]
        * (0.50 + (0.50 * mat["landing_urgency"]))
    )
    reward -= (
        SCALE_HESITATION_BACK_PENALTY
        * mat["hesitated_back_defense"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_WALL_BOUNCE_READ_BONUS
        * mat["wall_bounce_read_move"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_WALL_BOUNCE_DIVE_BONUS
        * mat["wall_bounce_read_dive"]
        * (0.50 + (0.50 * mat["landing_urgency"]))
    )
    reward -= (
        SCALE_WALL_BOUNCE_HESITATION_PENALTY
        * mat["wall_bounce_hesitation"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward += (
        SCALE_GROUNDED_TIMELY_MOVE_BONUS
        * mat["grounded_timed_move"]
        * (0.60 + (0.40 * mat["landing_urgency"]))
    )
    reward -= (
        SCALE_WRONG_DIRECTION_PENALTY
        * mat["wrong_way_move"]
        * (0.50 + (0.50 * mat["landing_urgency"]))
    )
    reward -= (
        SCALE_PREMATURE_JUMP_PENALTY
        * mat["premature_jump"]
        * (0.70 + (0.30 * (1.0 - mat["landing_urgency"])))
    )
    reward -= (
        SCALE_WRONG_JUMP_DIRECTION_PENALTY
        * mat["wrong_jump_direction"]
        * (0.60 + (0.40 * mat["landing_urgency"]))
    )
    reward -= SCALE_PREMATURE_BACK_JUMP_PENALTY * mat["premature_back_jump"]
    reward += SCALE_EMERGENCY_BACK_DIVE_BONUS * mat["emergency_back_dive"]
    reward -= (
        SCALE_UNNECESSARY_BACK_DIVE_PENALTY
        * mat["unnecessary_back_dive"]
        * (0.70 + (0.30 * (1.0 - mat["landing_urgency"])))
    )
    reward -= (
        SCALE_NON_EMERGENCY_BACK_DIVE_PENALTY
        * mat["non_emergency_back_dive"]
        * (0.65 + (0.35 * (1.0 - mat["landing_urgency"])))
    )
    reward -= SCALE_PASSIVE_BACK_ACTION_PENALTY * mat["passive_back_action"]
    reward -= SCALE_SERVE_SETUP_BACK_DIVE_PENALTY * mat["serve_setup_back_dive"]
    reward -= SCALE_SERVE_CONTROL_BACK_ACTION_PENALTY * mat["serve_control_back_action"]
    reward -= SCALE_SERVE_CONTROL_BACK_DIVE_PENALTY * mat["serve_control_back_dive"]
    reward += SCALE_SERVE_CONTROL_READY_BONUS * mat["serve_control_ready_action"]
    reward -= (
        SCALE_CONTROLLED_BALL_DIVE_PENALTY
        * mat["controlled_ball_dive"]
        * (0.80 + (0.20 * (1.0 - mat["landing_urgency"])))
    )
    reward -= (
        SCALE_UNNECESSARY_DIVE_PENALTY
        * mat["unnecessary_dive"]
        * (0.75 + (0.25 * (1.0 - mat["landing_urgency"])))
    )
    reward += SCALE_ASSERTIVE_ATTACK_ACTION_BONUS * mat["assertive_attack_action"]
    reward += SCALE_WELL_PREPARED_IDLE_BONUS * mat["well_prepared_idle"]
    reward -= (
        SCALE_RESTLESS_GROUND_ADJUST_PENALTY
        * mat["restless_ground_adjust"]
        * (1.00 - (0.35 * mat["landing_urgency"]))
    )
    reward -= (
        SCALE_IDLE_UNDER_PRESSURE_PENALTY
        * mat["idle_under_pressure"]
        * (0.55 + (0.45 * mat["landing_urgency"]))
    )
    reward -= SCALE_RECKLESS_FORWARD_PENALTY * mat["reckless_forward_under_pressure"]
    reward += SCALE_OPPONENT_DIVE_BONUS * mat["opponent_dive_used"]
    reward -= SCALE_OPPONENT_SPIKE_PENALTY * mat["opponent_spike_used"]
    position_alignment = 1.0 - normalize_minmax(
        mat["position_error"],
        0,
        GROUND_HALF_WIDTH,
    )
    if mat["settled_near_target"] > 0:
        centered_position_alignment = 1.0
    else:
        centered_position_alignment = (position_alignment - 0.5) * 2.0
    reward += SCALE_POSITION_ALIGNMENT_REWARD * centered_position_alignment
    reward -= SCALE_STANDBY_OVERFRONT_PENALTY * mat["self_overfront"]
    reward -= SCALE_STANDBY_OVERBACK_PENALTY * mat["self_overback"]

    if (
        mat["ball_on_self_side"] > 0.5
        and mat["self_touched_ball"] < 0.5
        and mat["position_error"] > (GROUND_HALF_WIDTH * 0.45)
    ):
        reward -= SCALE_IDLE_FAR_PENALTY

    if mat["point_scored"] > 0.5:
        score_speed_factor = 1.0 - normalize_minmax(
            mat["rally_total_frames_until_point"],
            0.0,
            RALLY_SPEED_REFERENCE_FRAMES,
        )
        reward += SCALE_SCORE_SPEED_BONUS * score_speed_factor
    elif mat["point_lost"] > 0.5:
        quick_loss_factor = 1.0 - normalize_minmax(
            mat["rally_total_frames_until_point"],
            0.0,
            RALLY_SPEED_REFERENCE_FRAMES,
        )
        reward -= SCALE_QUICK_LOSS_PENALTY * quick_loss_factor

    reward += SCALE_MATCH_WIN_BONUS * mat["match_won"]
    return reward

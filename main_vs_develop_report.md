# main vs develop 보고서

기준은 현재 로컬 `main`(32f99ad)과 `develop`(54bb315a) 비교다.  
이 문서는 요약본이 아니라, 실제로 어떤 값을 어떻게 바꿔서 `수비형` PPO를 만들었는지를 코드 기준으로 풀어쓴 보고서다.

## 튜닝 상세

### PPO 학습 파라미터

`develop`의 PPO는 "느리게 오래"가 아니라 "초기 탐색은 넓게, 학습은 짧고 자주, 종료는 KL로 제어" 쪽으로 바뀌었다.  
가장 큰 축은 학습률과 네트워크 크기, 그리고 업데이트 방식이다.

| 항목 | main | develop | 튜닝 의도 |
|---|---:|---:|---|
| Actor lr | `1e-5` | `2.5e-4` | 너무 느리던 정책 업데이트를 실제로 움직이게 함 |
| Critic lr | `1e-5` | `3e-4` | 가치함수 수렴을 더 빠르게 함 |
| Gamma | `0.99` | `0.995` | 장기 수비와 랠리 지속 보상을 더 반영 |
| GAE lambda | 없음 | `0.95` | 긴 시퀀스에서 advantage 분산을 줄임 |
| Update epochs | `100` | `8` | 같은 rollout을 과도하게 반복 학습하는 걸 방지 |
| Minibatch size | 없음 | `64` | 전체 배치 1회 학습 대신 안정적인 SGD 형태로 변경 |
| Target KL | 없음 | `0.03` | 정책이 한 번에 너무 멀리 가는 걸 차단 |
| Value clip | 없음 | `0.2` | critic 폭주를 줄임 |
| VF coefficient | 단순 MSE | `0.5` | actor/critic 균형을 명시 |
| Entropy start/end | 없음 | `0.01 -> 0.0025` | 초반 탐색, 후반 수렴 |
| Entropy decay episodes | 없음 | `120000` | 12만 episode 동안 탐색량을 점진 감소 |
| Max grad norm | 없음 | `0.5` | gradient 폭주 방지 |
| Hidden dim | `32` | `128` | 수비 패턴/벽 반응 같은 복잡한 상태 표현 대응 |
| Hidden layers | `2` | `3` | 표현력 증가 |
| Max steps per episode | `30*30` | `30*45` | 더 긴 랠리와 수비 상황까지 학습 |

추가로 `select_action(..., deterministic=True)`가 들어가서, 플레이/평가 시에는 샘플링 대신 greedy 행동을 고를 수 있게 했다. 이건 학습 자체를 바꾸는 건 아니지만, 튜닝 결과를 재현성 있게 확인하는 데 중요하다.

#### 실제 업데이트 순서

`_20_model/ppo/_00_model.py`의 흐름은 아래처럼 바뀌었다.

1. `actor_old`로 행동을 뽑는다.
2. 환경에서 `state -> action -> next_state -> reward -> done` transition을 모은다.
3. 에피소드가 끝나면 rollout 전체를 `states`, `states_next`, `actions`, `old_log_probs`, `rewards`, `dones`로 묶는다.
4. `critic(states)`와 `critic(states_next)`로 TD residual을 만들고, 여기서 GAE를 역방향으로 누적한다.
5. advantage를 평균 0, 분산 1에 가깝게 정규화한다.
6. epoch마다 minibatch shuffle을 하고, actor는 clipped surrogate loss + entropy bonus로 업데이트한다.
7. critic은 value clipping이 켜져 있으면 clipped/unclipped loss 중 큰 쪽을 사용하고, `vf_coefficient`를 곱해 학습한다.
8. actor/critic 둘 다 `max_grad_norm=0.5`로 gradient clipping을 건다.
9. epoch별 근사 KL이 `target_kl=0.03`을 넘으면 조기 종료한다.
10. 업데이트가 끝나면 `actor_old`를 최신 actor로 동기화하고 rollout buffer를 비운다.

즉, 이 튜닝은 "같은 rollout을 100번 반복"하던 방식에서, "한 rollout을 몇 epoch만 안정적으로 학습하고 바로 다음 데이터를 받는 방식"으로 바뀐 것이다.

### 상태 표현 튜닝

상태 차원은 그대로 `11`을 유지했다. 대신 5번째 슬롯을 그냥 `opponent_y`로 쓰지 않고, `wall_bounce_risk`를 섞은 `opponent_context`로 바꿨다.  
이 방식은 체크포인트 호환성을 지키면서도, 수비형 핵심 상황인 "벽 근처에서 튀는 공"을 더 잘 보게 만드는 타협이다.

| 항목 | main | develop | 의미 |
|---|---|---|---|
| Action group | 4종 근사 분류 | `idle/forward/backward/jump/jump_forward/jump_backward/dive_forward/dive_backward/spike` | 더 정교한 움직임 구분 |
| Opponent context | `opponent_y` | `0.7 * opponent_y + 0.3 * wall_bounce_risk` | 벽 공포 상황을 context에 주입 |
| Wall bounce risk | 없음 | `ball x`, `landing x`, `ball vx/vy`, self side 기반 계산 | 벽 대응 학습 신호 |
| State dim | `11` | `11` | 기존 checkpoint 호환성 유지 |

`wall_bounce_risk`는 단순 위치가 아니라, 공이 내쪽 코트에 있고 왼쪽 벽 쪽으로 접근 중인지, 낙하가 임박했는지, 착지점이 벽 쪽인지까지 같이 본다. 즉, "막아야 할 공"을 별도 문맥으로 만들어 준 셈이다.

#### 왜 이 신호가 필요한가

수비형 학습에서 가장 흔한 실패는 "공이 실제로 위험한데도 중립 상태처럼 보고 가만히 있음"이다.  
`wall_bounce_risk`는 이런 경우를 줄이기 위해 `ball_x`, `expected_landing_x`, `ball_velocity_x`, `ball_velocity_y`, 그리고 내가 어느 쪽 코트에 있는지까지 한 번에 압축한 문맥값이다.

이 값이 들어가면 모델은 단순히 `opponent_y`만 보는 게 아니라:

- 공이 내쪽에 이미 들어왔는지
- 공이 벽 쪽으로 붙어서 튀고 있는지
- 착지점이 왼쪽 벽에 가까운지
- 지금이 대기인지, 이동인지, 점프인지

를 구분하기 쉬워진다. 수비형 정책에서 이 구분이 중요하다.

### 보상 설계 튜닝

이 부분이 사실상 `develop`의 핵심이다. 기존에는 점수/패배처럼 희소한 신호 중심이었고, `develop`에서는 "언제 움직였는지", "어떤 방향으로 움직였는지", "공이 위험한데 버텼는지"를 촘촘히 점수화한다.

| 카테고리 | 신호 | 스케일/조건 | 해석 |
|---|---|---|---|
| 점수 | `point_scored` | `+25` | 득점 자체 보상 |
| 점수 | `point_lost` | `-25` | 실점 패널티 |
| 생존 | `self_dive_missed` | `-0.25` | 다이브했는데 맞추지 못함 |
| 생존 | `self_overchase` | `-0.26` | 앞쪽으로 과도하게 쫓아감 |
| 생존 | `self_recover` | `+0.16` | 공이 반대편일 때 복귀 동작 |
| 기본 수비 | `timely_backward_move` | `+0.18 * urgency` | 뒤로 빠지는 타이밍이 맞음 |
| 기본 수비 | `timely_forward_move` | `+0.15 * urgency` | 앞으로 전진하는 타이밍이 맞음 |
| 전방 대응 | `front_save_move` | `+0.16 * urgency` | 앞쪽 수비를 실제로 수행 |
| 전방 대응 | `front_save_dive` | `+0.11 * urgency` | 앞쪽 다이브 수비 |
| 전방 대응 | `hesitated_front_defense` | `-0.16 * urgency` | 앞쪽 공인데 망설임 |
| 후방 대응 | `back_save_move` | `+0.18 * urgency` | 뒤쪽 수비를 실제로 수행 |
| 후방 대응 | `back_save_dive` | `+0.11 * urgency` | 뒤쪽 다이브 수비 |
| 후방 대응 | `hesitated_back_defense` | `-0.18 * urgency` | 뒤쪽 공인데 망설임 |
| 벽 대응 | `wall_bounce_read_move` | `+0.16 * urgency` | 벽 튀김을 읽고 움직임 |
| 벽 대응 | `wall_bounce_read_dive` | `+0.08 * urgency` | 벽 튀김에 다이브 대응 |
| 벽 대응 | `wall_bounce_hesitation` | `-0.18 * urgency` | 벽 공인데 안 움직임 |
| 지면 대응 | `grounded_timed_move` | `+0.11 * urgency` | 바닥 이동 타이밍 일치 |
| 방향 오류 | `wrong_way_move` | `-0.14 * urgency` | 필요와 반대 방향으로 감 |
| 점프 오류 | `premature_jump` | `-0.10 * (1-urgency)` | 너무 일찍 점프 |
| 점프 오류 | `wrong_jump_direction` | `-0.12 * urgency` | 점프 방향이 틀림 |
| 점프 오류 | `premature_back_jump` | `-0.08` | 필요 없는 후방 점프 |
| 다이브 | `emergency_back_dive` | `+0.10` | 진짜 급한 상황의 후방 다이브 |
| 다이브 | `unnecessary_back_dive` | `-0.12 * ...` | 급하지 않은 후방 다이브 |
| 다이브 | `non_emergency_back_dive` | `-0.08 * ...` | 긴급성이 낮은 후방 다이브 |
| 다이브 | `passive_back_action` | `-0.08` | 수비 압박 없는데 뒤로만 감 |
| 서브 | `serve_setup_back_dive` | `-0.20` | 서브 준비 중 과한 다이브 |
| 서브 | `serve_control_back_action` | `-0.12` | 서브 제어 상황에서 불필요한 후퇴 |
| 서브 | `serve_control_back_dive` | `-0.36` | 서브 제어 상황에서 특히 강한 패널티 |
| 서브 | `serve_control_ready_action` | `+0.08` | 서브 제어 상태에서 대기/준비 |
| 공 제어 | `controlled_ball_dive` | `-0.24 * ...` | 낮고 가까운 공에 과한 다이브 |
| 공 제어 | `unnecessary_dive` | `-0.14 * ...` | 급하지 않은데 다이브 |
| 공격 | `assertive_attack_action` | `+0.12` | 공격 기회에서 적극 행동 |
| 포지션 | `well_prepared_idle` | `+0.05` | 자리 잘 잡고 대기 |
| 포지션 | `restless_ground_adjust` | `-0.06` | 이미 맞는데 미세 조정만 반복 |
| 압박 | `idle_under_pressure` | `-0.16` | 위험한데 가만히 있음 |
| 압박 | `reckless_forward_under_pressure` | `-0.12` | 뒤로 가야 할 상황에 전진 |
| 상대참조 | `opponent_dive_used` | `+0.00` | 영향 없음 |
| 상대참조 | `opponent_spike_used` | `-0.10` | 상대 스파이크는 위험 신호 |
| 위치보정 | `position_alignment` | `+0.16` | 목표 위치와의 정렬 |
| 위치보정 | `self_overfront` | `-0.12` | 너무 앞쪽에 나감 |
| 위치보정 | `self_overback` | `-0.08` | 너무 뒤쪽에 박힘 |
| 위치보정 | `idle_far` | `-0.14` | 공이 내쪽인데 멀리서 정지 |
| 속도보정 | `score_speed_bonus` | `+0.45 * factor` | 빠른 득점 선호 |
| 속도보정 | `quick_loss_penalty` | `-0.70 * factor` | 빨리 진 랠리 억제 |
| 최종 | `match_won` | `+20` | 경기 승리 보상 |

여기서 핵심은 3가지다.
첫째, `landing_urgency`에 가중치를 붙여서 공이 진짜 위험할 때만 강하게 벌점이 들어가게 했다.  
둘째, 수비 상황에서의 "올바른 방향"과 "잘못된 방향"을 분리해서 점수를 줬다.  
셋째, 서브/벽/낙하지점처럼 특정 상황 전용 신호를 따로 만들었다.

그래서 이 튜닝은 단순히 "수비 보너스 증가"가 아니라, "위험 상황을 읽고, 그 상황에 맞는 움직임을 선택하고, 불필요한 다이브/점프를 줄이는 방향"으로 policy를 밀어준다.

#### 보상 계산의 실제 해석

보상은 크게 5단계로 쌓인다.

1. **결과 보상**
   - `point_scored * 25`
   - `point_lost * -25`
   - `match_won * 20`

2. **생존/회수 보상**
   - `self_recover`, `self_dive_missed`, `self_overchase` 같은 항목으로
   - "살릴 수 있었는가"와 "괜히 과하게 움직였는가"를 동시에 본다.

3. **상황 적합성 보상**
   - `landing_urgency`가 높을 때만 `timely_backward_move`, `front_save_move`, `wall_bounce_read_move` 같은 점수가 강해진다.
   - 즉, 같은 행동이라도 위험도가 낮으면 점수가 작고, 정말 급하면 점수가 커진다.

4. **오류 억제**
   - `premature_jump`, `wrong_jump_direction`, `idle_under_pressure`, `unnecessary_dive` 같은 항목으로
   - "해야 할 때 안 한 것"과 "하면 안 될 때 한 것"을 같이 처벌한다.

5. **리듬 보정**
   - `score_speed_bonus`와 `quick_loss_penalty`로
   - 길게 끌어서 겨우 이기거나, 빨리 무너지는 패턴을 조정한다.

여기서 가장 중요한 건 `position_target_x`다.  
이 값은 `standby_target_x`와 `intercept_target_x`를 섞어서 만들고, `ball_on_self_side`, `predicted_self_landing`, `opponent_spike_used`에 따라 섞는 비율이 달라진다.  
즉, "기본 위치"와 "지금 당장 막아야 하는 위치" 사이를 유동적으로 오가게 만든다.

#### 수비형 튜닝의 실제 방향

이 보상은 공격을 없애려는 게 아니다.  
정확히는 다음 우선순위를 학습시키려는 것이다.

- 공이 멀고 안전하면 준비 위치를 유지한다.
- 공이 내쪽으로 오면 필요한 방향으로 먼저 이동한다.
- 벽쪽으로 튀는 공이면 일반 수비가 아니라 벽 대응으로 처리한다.
- 정말 급한 경우만 dive를 허용한다.
- 급하지 않은 jump/dive는 오히려 패널티를 준다.

그래서 "수비형"이라는 이름이 붙었지만, 실제로는 "불필요한 과잉 행동을 줄이고, 필요한 행동만 정확히 하게 만드는 정책"에 가깝다.

### 학습 운영 튜닝

훈련 루프도 같이 손봤다. 핵심은 self-play와 curriculum, 재개 가능 상태 저장, 그리고 플레이어 측면 편향 제거다.

| 항목 | main | develop | 역할 |
|---|---|---|---|
| Curriculum | 없음 | `0=rule,30000=self` | 초반은 rule, 이후 self-play |
| Train side mode | 고정 | `alternate` 지원 | 1P/2P 번갈아 학습 |
| Train workers | 1 | 1 기본, PPO만 병렬 허용 | 병렬 rollout 준비 |
| Self-play snapshot interval | 없음 | `2000` | 2천 episode마다 self snapshot |
| Self-play pool | 없음 | 사용 | 과거 자기 자신들을 opponent로 저장 |
| Pool size | 없음 | `32` | 오래된 self snapshot 제한 |
| Latest prob | 없음 | `0.35` | 최신 정책을 너무 자주만 쓰지 않음 |
| Resample interval | 없음 | `25` | 같은 self opponent만 붙잡지 않음 |
| Warmup episode | 없음 | `10000` | 초반엔 최신 self-play를 너무 빨리 쓰지 않음 |
| Rule mix prob | 없음 | `0.30` | 완전 self-play 편향을 줄임 |
| Checkpoint interval | 없음 | `10000` | 1만 episode마다 저장 |
| Resume state | 없음 | `*.train_state.json` | 중단 후 이어서 학습 |
| Reset epsilon | 없음 | 선택 가능 | 재개 시 탐색도 초기화 가능 |

운영 관점에서 보면, `develop`은 한 번 돌리고 끝내는 구조가 아니라, `rule -> self -> pool self-play -> checkpoint -> resume`의 순환 구조로 바뀌었다. 이게 사실상 "학습 설계"의 본체다.

#### self-play pool이 도는 방식

`_30_src/train.py`에서 self-play pool은 단순 로그가 아니라 실제 opponent 저장소다.

1. `get_self_play_snapshot_dir()`로 `_self_play_pool/<policy_name>/`를 만든다.
2. `save_self_play_snapshot()`이 현재 actor/critic 쌍을 `*_sp_ep{episode}` 이름으로 저장한다.
3. `register_self_play_pool_snapshot()`이 새 snapshot을 pool에 넣고, 크기가 `32`를 넘으면 오래된 항목을 지운다.
4. 지워질 때는 `*_sp_ep` 파일 자체도 같이 삭제한다.
5. `select_self_play_pool_opponent()`가 `warmup_episode=10000` 이전에는 주로 최신 정책을 쓰고, 이후에는 `latest_prob=0.35`를 기준으로 최신/과거 snapshot을 섞는다.
6. `resample_interval=25` 때문에 같은 self opponent만 너무 오래 쓰지 않는다.

이 구조 덕분에 self-play가 "최신 자기 자신과만 싸우는 폐쇄 루프"가 아니라, 조금 전의 자기 자신들까지 섞어서 상대하는 형태가 된다.

#### curriculum과 side alternate

`parse_curriculum_schedule()`는 `"0=rule,30000=self"`를 시작점으로 episode별 상대를 바꾼다.  
즉, 3만 전에는 rule opponent로 안정화하고, 3만 이후에는 self-play로 넘어간다.

`resolve_training_side()`는 `alternate`일 때 episode index 짝수/홀수에 따라 1P/2P를 바꾼다.  
이건 특정 한쪽에서만 학습하면서 생기는 자리 편향을 줄이기 위한 것이다.

#### checkpoint와 resume

`checkpoint_interval=10000`이기 때문에 1만 episode 단위로 `inchihci_epXXXX`가 저장된다.  
이 파일들은 단순 모델이 아니라, `train_state.json`과 같이 저장돼서 다음 항목을 같이 복원한다.

- `episodes_completed`
- `win_count`
- `loss_count`
- `draw_count`
- `epsilon`
- 마지막으로 저장한 `policy_path`

그래서 학습 중단 후 재개하면, 단순히 가중치만이 아니라 진행 상태까지 이어진다.  
`reset_epsilon=True`를 주면 정책은 이어받되 탐색도 다시 시작할 수 있다.

### 플레이/환경 튜닝

학습 외에도 평가와 플레이 흐름을 정리했다.

| 항목 | 변경 | 이유 |
|---|---|---|
| Viewer input sync | 단순화 | 입력 처리 복잡도 감소 |
| Play policy resolve | 추가 | 짧은 policy 이름도 자동 해석 |
| Deterministic play | 추가 | 같은 정책 결과를 재현하기 쉬움 |
| Restart loop | 축소 | 1판 종료형으로 정리 |

#### 플레이 진입점 변화

`_30_src/play.py`는 단순화되었지만, 실사용에서는 꽤 중요하다.

- `resolve_policy_name_for_play()`가 짧은 policy 이름을 실제 파일명으로 해석한다.
- `env.wait_key_for_start()` 뒤 restart 루프를 줄여서, 한 판 끝나면 바로 종료하는 쪽으로 정리했다.
- `draw` 처리는 별도 winner 텍스트 분기 없이 정리돼 있어서, 보고서 기준으로는 "플레이는 평가용 단판 실행"에 더 가깝다.

#### 환경 쪽 정리

`_00_environment/env.py`와 `viewer.py`는 수비형 학습을 직접 바꾸는 건 아니지만, 실험 안정성에는 영향을 준다.

- seed를 더 이상 내부에서 정규화하지 않고 그대로 쓴다.
- serve side 결정이 단순해져서 재현성이 좋아진다.
- 종료 시 전이 로직을 단순화했다.
- viewer 렌더 루프를 정리해서 학습 로그와 렌더링의 간섭을 줄였다.

## 개발 흐름

| 커밋 | 메시지 | 핵심 내용 |
|---|---|---|
| `184c3deb` | `make develop branch` | develop 베이스 생성, PPO/CLI/train/play 쪽 대규모 초안 반영 |
| `ebf0e906` | `튕기는 공에 대한 대응 추가` | 벽/튕김 대응 보강, self-play pool과 defensive reward 강화 시작 |
| `5c415297` | `update 12만 ppo` | 12만 구간까지 학습 파일 확장, self-play pool/체크포인트 확장 |
| `3091b4c4` | `학습파일 추가` | 학습 산출물 대량 추가, generated cache 정리 |
| `54bb315a` | `수비형` | 최종 수비형 튜닝, PPO 하이퍼파라미터/보상 재정렬 |

## 소스 변경

### `.gitignore`
- `_20_model/**/outputs/policy_trained/` 주석 처리로 학습 산출물을 추적 대상으로 바꿈.
- `__pycache__/`, `*.pyc`, `*.pkl`, `*.log` 를 추가로 무시하도록 변경.

### `_00_environment/env.py`
- seed 정규화 헬퍼를 제거하고 `seed`를 그대로 보관하도록 단순화.
- serve side 결정 로직을 직접 분기 방식으로 바꿈.
- 상태 전이 분기를 단순화해 재현성을 높임.

### `_00_environment/viewer.py`
- 키 입력 동기화 로직 단순화.
- 렌더 루프를 이벤트 대기 기반으로 바꿔 입력 처리/프레임 갱신을 단순화.

### `_10_config/conf.py`
- `target_score_train`을 3에서 5로 변경.
- `seed` 기본값을 `None`에서 `100`으로 변경.
- `train_side_mode`, `train_num_workers`, `reset_epsilon`, self-play pool, curriculum 관련 설정 추가.
- B/W 모드 비밀번호도 `301`로 변경.

### `cli.py`
- 문자열 기반 boolean 파서를 추가.
- 학습 관련 새 CLI 옵션 추가:
  - `train_side_mode`
  - `train_num_workers`
  - `reset_epsilon`
  - `self_play_snapshot_interval`
  - `self_play_pool_enabled`
  - `self_play_pool_size`
  - `self_play_pool_latest_prob`
  - `self_play_pool_resample_interval`
  - `self_play_pool_warmup_episode`
  - `self_play_rule_mix_prob`
  - `curriculum_enabled`
  - `curriculum_schedule`
- `train_rewrite`와 `random_serve`도 문자열 boolean으로 받도록 변경.

### `_20_model/ppo/_01_params.py`
- 학습률, 감마, 네트워크 크기, 에폭 수를 재튜닝.
- GAE, minibatch, KL early stop, entropy decay, grad clipping, value clipping 관련 파라미터 추가.
- 학습 스케줄 기본값 추가:
  - curriculum: `0=rule,30000=self`
  - train side alternate
  - self-play pool 사용

### `_20_model/ppo/_00_model.py`
- PPO를 GAE + minibatch update 구조로 확장.
- entropy bonus, value clipping, grad clipping, target KL early stop 추가.
- `select_action(..., deterministic=True)` 지원 추가.
- `set_training_progress()`로 entropy decay 진행률 반영.
- 학습 상태 추적값 `last_approx_kl`, `last_update_epochs` 추가.
- play 모드에서 checkpoint 미존재 시 명시적 에러 발생.

### `_20_model/ppo/_03_state_design.py`
- action group 매핑을 더 세분화.
- ball/landing/wall bounce 상황을 반영하는 `wall_bounce_risk`를 추가.
- state dimension은 11로 유지하면서 opponent context 자리에 wall-bounce awareness를 주입.

### `_20_model/ppo/_05_reward_design.py`
- 가장 큰 변경점 중 하나.
- 단순 득실 중심 보상에서 벗어나 수비형 행동을 촘촘히 보상/패널티화함.
- 추가된 축:
  - front/back defense 타이밍
  - wall bounce read
  - emergency back dive / unnecessary dive 구분
  - serve control / serve setup
  - premature jump / wrong direction / idle under pressure 억제
  - assertive attack, recover, position alignment 보너스
- 결과적으로 "공을 막는 플레이"와 "과한 점프/다이브 억제"에 가중치가 크게 이동함.

### `_20_model/ppo/_06_algorithm.py`
- 확률적 액션 선택 외에 deterministic 액션 선택 추가.
- 평가/플레이 시 재현성 있는 policy 선택이 가능해짐.

### `_30_src/play.py`
- 정책 이름을 policy 폴더 안에서 자동 탐색해 짧은 이름도 해석 가능하게 함.
- match restart 루프를 제거해서 1판 종료형으로 단순화.
- draw 처리 분기 제거로 인해 draw 시 winner 출력은 사실상 `player2`로 찍힐 수 있음.

### `_30_src/train.py`
- 전체 학습 러너가 사실상 재작성됨.
- 추가된 핵심 기능:
  - training state 저장/복원(`*.train_state.json`)
  - curriculum schedule 파싱
  - self-play opponent pool 저장/샘플링
  - train side alternate
  - PPO 전용 parallel rollout worker
  - worker policy snapshot 생성/정리
  - 정기 checkpoint 저장
  - entropy/kl/reward/win-rate 상태 로그
  - epsilon reset 옵션
- 학습이 단순 에피소드 루프가 아니라 "스냅샷 관리 + 상대 정책 관리 + 재개 가능 상태" 구조로 바뀜.

## 산출물 변경

### 학습 체크포인트
- `develop`에는 `inchihci.pth`, `inchihci_critic.pth`, `inchihci.train_state.json`이 포함됨.
- regular checkpoint는 `inchihci_ep10000`부터 `inchihci_ep250000`까지 1만 단위로 actor/critic 쌍이 존재함.
- self-play snapshot은 `inchihci_sp_ep190000`부터 `inchihci_sp_ep252000`까지 2천 단위로 actor/critic 쌍이 존재함.
- self-play pool 메타데이터는 `_self_play_pool/inchihci/inchihci.selfplay_pool.json`에 저장됨.

### 캐시/바이너리
- `__pycache__` 아래의 `*.cpython-312.pyc` 파일 다수가 변경 또는 삭제됨.
- 의미 있는 소스 변경은 아니고, 브랜치에 컴파일 산출물이 함께 들어간 결과임.

## 로컬 추가 코드

### `human_vs_ai_launcher.py`
- 이 파일은 inspect 당시 로컬 untracked 상태였고, 이후 `git clean`으로 정리되어 현재 브랜치 diff에는 포함되지 않는다.
- 내용상 목적은 human vs PPO 매치용 런처였다.
- 카드 UI로 `90000` / `250000` 두 checkpoint 중 상대를 고르게 되어 있었고, checkpoint 존재 여부도 검사했다.
- 즉, branch diff 외에 별도로 만든 실험용 실행 진입점이다.

## 파일 수

- source `.py`: 11
- metadata/config (`.gitignore`): 1
- training artifacts: 118 added
- cache/pyc: 82
- total diff files: 212

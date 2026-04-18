# main vs develop 보고서

기준은 현재 로컬 `main`(32f99ad)과 `develop`(54bb315a) 비교다.  
요약하면 `develop`은 PPO 학습 파이프라인, 수비형 보상 설계, self-play/curriculum, 체크포인트 관리, 플레이 UI를 크게 확장했고, 반대로 오디오/사운드 자산은 제거했다.

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

### `_00_environment/audio.py`
- `develop`에서 삭제됨.
- 오디오 재생/효과음 계층이 제거되면서 viewer 쪽 오디오 의존성도 같이 사라짐.

### `_00_environment/env.py`
- seed 정규화 헬퍼를 제거하고 `seed`를 그대로 보관하도록 단순화.
- serve side 결정 로직을 직접 분기 방식으로 바꿈.
- point/match 종료 시 선수 상태를 강제로 눕히는 처리 제거.
- 즉, 종료 연출보다 학습/재현성 위주의 간결한 상태 전이를 택함.

### `_00_environment/viewer.py`
- `ViewerAudio`와 모든 오디오 재생 로직 제거.
- 키 입력 동기화 로직 단순화.
- 게임 종료 애니메이션, 확대 배너, 종료 프레임 연출 제거.
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

### 제거된 오디오 자산
- `_00_environment/assets/WAVE140_1.wav`
- `_00_environment/assets/WAVE141_1.wav`
- `_00_environment/assets/WAVE142_1.wav`
- `_00_environment/assets/WAVE143_1.wav`
- `_00_environment/assets/WAVE144_1.wav`
- `_00_environment/assets/WAVE145_1.wav`
- `_00_environment/assets/WAVE146_1.wav`
- `_00_environment/assets/bgm.mp3`

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

- source `.py`: 12
- metadata/config (`.gitignore`): 1
- training artifacts: 118 added
- removed audio assets: 8
- cache/pyc: 82
- total diff files: 221

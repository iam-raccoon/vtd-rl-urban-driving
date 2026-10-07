# vtd-rl-urban-driving

Hexagon VTD 2025.2 의 도심 지도(LivingLab)에서 도로교통법을 지키며 달리는 **end-to-end 강화학습 운전 정책**을 만든 개인 프로젝트다.
2026 HL-FMA 대회에서 완주한 규칙 기반 주행 스택을 선생님으로 두고, 모방학습(DAgger)으로 시작해 PPO 로 다듬었다.

> **상태: 마무리(2026-10-07).** 학습과 평가는 VTD 가 아니라 **이 레포의 오프라인 세계**에서 했다([어디서 학습했나](#어디서-학습했나--오프라인-세계-vtd-아님)).
> 오프라인 시험 창에서 학생은 선생님에 2.3 점 못 미친다. **실제 VTD 에서는 선생님만 그대로 옮겨졌다**(신호만 코스 6/6 완주).
> 학생은 2/6 이다. 오프라인 세계에 접지 한계·언더스티어·일부 신호 상태가 없어서다([VTD 에서 달려 보기](#vtd-에서-달려-보기)).

- 설계: [docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md](docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md)
- 규칙 스택 공개 참고본: [HL-FMA2026-VTD](https://github.com/iam-raccoon/HL-FMA2026-VTD)

## 결과 한눈에

학습에도 고르기에도 쓰지 않은 시험 창 세 개(창마다 384 판, 합 1152 판)에서 학생과 선생님이 같은 판을 달렸다.
판은 단계 ③ 전체다: 코스 여섯(2.2~5.2 km), 신호 주기, 라바콘과 정지 차량, 변종마다 액터 자리를 흔든다.
점수는 대회 채점기를 옮긴 온라인 심판의 구간 평균(100 점 만점)이다.

| 운전자 | 완주 | 완주 포함 점수(미완주 0 점) | 완주한 판의 점수 |
|---|---:|---:|---:|
| 선생님(규칙 스택) | 1125 / 1152 (97.7%) | 94.65 | 96.9 |
| **학생(배포 묶음)** | 1110 / 1152 (96.4%) | 92.38 | 95.9 |

- 배포 묶음 = 그물 M7f 시드 2 · 12.5 만 걸음 + 관측의 앞당긴 제한속도(평가할 때 `--lim-anticipate` 를 켠다).
- 학생이 놓친 42 판은 전부 충돌이다(course_G 30, course_D 11, course_E 1). 정지차 앞에서 옆 차로로 빠져나가지 못한다.
  선생님이 놓친 27 판은 전부 정체다(course_B 23, course_H 4).
- 창마다 표: [docs/reports/final-summary.md](docs/reports/final-summary.md)

### 어떻게 여기까지 왔나

단계 ③ 전체 판에서 학생 성적이 크게 바뀐 마일스톤만 골랐다. 잰 창이 줄마다 다르다(보고 창 = 72 판, 시험 창 = 새 384 판).
같은 그물도 시험 창에 따라 5 점 가까이 흔들리므로(M7f: 88.9~93.6) 가까운 줄끼리의 작은 차이는 읽지 않는다. 점수는 M6u 부터 쟀다.

| 마일스톤 | 바꾼 것 | 창 | 완주율 | 완주 포함 점수 |
|---|---|---|---:|---:|
| M4i | 모방 학생(단계 ①② 데이터만) | 단계 ③ 판 18 | 0% | - |
| M5e | 단계 ③ 에서 DAgger 를 다시 모음 | 보고 창 | 3.6% | - |
| M6c | KL 앵커를 단 PPO | 보고 창 | 64.8% | - |
| M6h | 변종 판(액터 자리 흔들기)으로 학습 | 보고 창 | 80.6% | - |
| M6m | 큰 창(384 판)으로 그물 고르기 | 시험 창 | 87.8% | - |
| M6r | 최고 그물에서 시드 여러 개로 이어 돌려 고르기 | 시험 창 | 96.4% | - |
| M6w | 적색 신호 감속 곡선 벌 | 시험 창 | 97.1% | 85.50 |
| M7b | 경로 횡오차 벌(물체 80 m 안에서는 끔) | 시험 창 | 96.1% | 90.67 |
| M7d | 앞길 장애물 감속 곡선 벌 | 시험 창 | 98.2% | 93.58 |
| M7j | 관측에 앞당긴 제한속도(학습 없이 켜기만) | 시험 창 | 96.1% | 92.12 |

## 어디서 학습했나 — 오프라인 세계 (VTD 아님)

VTD 는 라이선스가 하나뿐이고 실시간(20 Hz)으로만 돈다. 리셋에 20 초가 걸리고 오래 돌리면 불안정하다.
PPO 는 실행 하나에 100 만 걸음을 쓰는데 VTD 로는 감당이 안 된다. 그래서 처음부터 **학습은 규칙 스택의 오프라인 시뮬
부품으로 만든 오프라인 세계(`vtd_rl/world`)에서 하고, VTD 는 검증·보정에만 쓴다**고 정했다(설계 문서).

오프라인 세계가 VTD 와 같은 것:

- **지도:** 대회 LivingLab xodr 그대로다. 경로·차로계획·정지선·신호 위치·보호구역도 지도에서 나온다.
- **입력:** VTD 9910 패킷과 같은 상태(자차 자세, 물체 30 개, 다음 신호 하나)를 20 Hz 로 낸다.
- **채점:** 대회 채점기(`score_fma.py`)를 프레임 단위로 옮긴 온라인 심판이다. 원본과 판정이 같은지 테스트로 잠갔다(M2a, M4d).
- **선생님:** 대회에서 완주한 규칙 스택 코드 그대로다(서브모듈, 수정하지 않는다).

VTD 와 다른 것(VTD 에서 성적이 달라질 수 있는 곳):

- **차량 동역학:** 자전거 모델 + 가속 1차 지연 + 조향 속도 제한이다. 계수를 VTD 차에 맞춰 보정하지 않았다(`world/dynamics.py`).
- **주변:** 대본대로 놓인 정지 차량·라바콘과 신호뿐이다. VTD 의 움직이는 교통·리스폰은 없다.
- **자차 속도:** 오프라인 학생은 정확한 속도를 본다. VTD 에서는 위치를 미분한 추정값을 본다(패킷에 속도가 없다).

## VTD 에서 달려 보기

`vtd_rl/bridge` 는 학습한 정책(또는 선생님)을 9910 으로 VTD 에 붙이는 주행기다. 오프라인 환경과 **같은 관측·행동·심판
코드**를 쓴다. 이 주행기를 오프라인 세계에 물리면 환경과 점수·채점표·종료 사유가 정확히 같다(`tests/bridge`).
그래서 VTD 에서 성적이 달라지면 그 차이는 VTD 쪽(동역학·속도 추정·교통)에서 온 것이다.

VTD 에서 시나리오를 올리고 시작한 뒤(규칙 스택 `vtd/load_scenario.py`) 붙인다. 판이 끝나면 브레이크를 물고 끊는다.

    env -u PYTHONPATH .venv/bin/python scripts/drive_vtd.py --board course_A --curriculum curricula/stage2.json \
        --ckpt <배포 그물> --lim-anticipate --out runs/$(hostname)/vtd/A-student
    env -u PYTHONPATH .venv/bin/python scripts/drive_vtd.py --board course_A --curriculum curricula/stage2.json \
        --teacher --out runs/$(hostname)/vtd/A-teacher

`<out>/result.json` 에 성적(구간 점수·채점표·종료 사유·리스폰)이, `<out>/rows.csv` 에 채점기 행이,
`<out>/ctrl.csv` 에 프레임마다 보낸 명령이 남는다. 첫 프레임의 자차가 판 출발점에서 10 m 넘게 떨어져 있으면 다른
시나리오가 올라온 것으로 보고 달리지 않는다. VTD 는 자차를 경로 출발 차로보다 한 차로 왼쪽에 놓으므로, 처음 경로 차로에
들어올 때까지는 도로 이탈 판정을 미룬다.

VTD PC 에서 시나리오 열두 개(코스 여섯 × 신호만 / `_EV`)를 차례로 올리고 선생님·학생을 한 판씩 달리는 스크립트:

    CKPT=<배포 그물> bash scripts/vtd_validate.sh runs/$(hostname)/vtd plain   # HL_FMA_NEW_<코스>.xml: 신호만(단계 ② 에 해당)
    CKPT=<배포 그물> bash scripts/vtd_validate.sh runs/$(hostname)/vtd ev      # HL_FMA_NEW_<코스>_EV.xml: 내 차로 정지차 + 대향차

### VTD 결과(2026-10-07)

OMEN 의 VTD 2025.2 에서 시나리오마다 한 판씩 달렸다. 숫자와 진단은 [docs/reports/vtd-validation.md](docs/reports/vtd-validation.md) 에 있다.

| 묶음 | 신호만 6 코스: 완주 · 완주 포함 점수 | `_EV` 6 코스: 완주 · 완주 포함 점수 |
|---|---:|---:|
| 선생님(규칙 스택) | 6/6 · 99.60 | 5/6 · 82.43 |
| 학생(배포 묶음) | 2/6 · 32.33 | 1/6 · 15.77 |
| 학생 + 어댑터 | 5/6 · 79.43 | 1/6 · 15.37 |
| 참고: 같은 신호만 코스를 오프라인에서(× 시드 3) — 선생님 | 18/18 · 99.53 | - |
| 참고: 같은 신호만 코스를 오프라인에서(× 시드 3) — 학생 | 18/18 · 98.83 | - |

- **선생님은 VTD 로 그대로 옮겨진다.** 지도·9910 입력·심판이 VTD 와 맞는다는 뜻이다.
- **학생은 옮겨지지 않는다.** 원인은 셋이다.
  1. 오프라인 자전거 모형에 접지 한계가 없다. 학생이 반지름 7 m 교차로를 40 km/h 남짓으로 돈다(측가속 1.7~1.9 g, 선생님은 20~23 km/h).
  2. VTD 차는 언더스티어가 있다. 요레이트 = v·tan(δ)/(L + K·v²), K ≈ 0.016 s²/m 이고, 가속은 오프라인 모형과 같다.
  3. VTD 녹색의 대부분이 오프라인 세계에 없는 "녹색 + 좌회전 화살표"(5) 다. 학생이 그 신호 앞에서 서 버린다.
- **어댑터는 응급처치다**(`--green-left-as-green --steer-comp 0.016`, 재학습 없음).
  - 신호 어댑터는 정체를 없앴다.
  - 조향 보정은 낮은 속도에서만 맞는다.
  - 교차로 과속과 정지차 충돌은 남는다.

## 무엇이 통했고 무엇이 안 통했나

통한 것:

- **KL 앵커를 단 PPO**(M6a·M6b): 출발 그물과의 KL 을 벌로 걸어야 PPO 가 배운 회피를 잃지 않고 출발점을 넘었다.
  앵커 없는 PPO 는 25 만~50 만 걸음 안에 회피를 무너뜨렸다.
- **변종 판**(M6e): 액터 자리를 흔든 판으로 학습하자 안 본 판 성적이 올랐다.
- **큰 창으로 고르기**(M6m·M6n): 72 판 창은 상위 그물을 거의 못 가른다(순위 상관 0.17). 384 판으로 고르고 새 창으로 확인했다.
- **곡선형 즉시 벌**(M6w·M7b·M7d): 걸음마다 "지금 설 수 있는 속도(또는 지켜야 할 자리)를 얼마나 넘었나" 를 깎았다.
  조금 고쳐도 벌이 줄어 기울기가 생긴다. 적색 중대 3.6 → 0.3 건, 차로 유지 중대 3.9 → 1.7 건, 장애물 앞 감속을 배웠다.
- **관측 하나 고치기**(M7h): 앞에 낮은 제한속도가 있다는 정보를 주자 학습 없이 보호구역 위반이 87% 줄었다.

안 통한 것:

- **단계 ①② 에서 모방 학생 위에 PPO**(M4a~M4g): 실행 13 개 모두 출발점(모방 학생)을 못 넘었다. 선생님이 99 점대라 이길 여지가 거의 없었다.
- **BC 로 새 단계 덧배우기**(M4l): 손실에 "알던 것을 유지하라" 는 항이 없어 1 에포크 만에 옛 운전을 버린다.
- **보조 모방**(M6q·M7k): PPO 에 선생님 라벨 모방을 섞으면 계수를 0.02 까지 줄여도 정책이 무너졌다.
- **벌 키우기**(M6u·M7a): 적색 위반 벌 5~10 배, 실패 벌 −200~−500 모두 행동을 못 바꿨다. 끝까지 가 보지 않는 행동에는 기울기가 없다.
- **늦게 오는 벌**(M6x): 채점기처럼 2.5 초 늦게 오는 차로 침범 벌은 차로 유지를 못 고쳤다. 바로 오는 횡오차 벌(M6y)은 고쳤다.
- **연습 늘리기**(M7c·M7l): 어려운 코스 덧보기, 어려운 장면 바로 앞에서 출발하는 짧은 판 모두 정지차 충돌을 못 줄였다.

## 남은 문제

- **정지차 앞에서 옆 차로로 빠져나가기.** 선생님은 정지차 앞에서 거의 선 뒤 옆 차로로 천천히 옮겨 지나간다. 학생은 이
  동작을 찾지 못한다. M7e 이후 학습 마일스톤 여덟 번은 지금 최고 그물을 확실히 넘지 못했다(큰 선택 창 최고점 90~91.4,
  M7k 는 정책이 무너졌다). 순수 강화학습으로는 여기서 막혔고, 다음 후보는 이 결정만 규칙 기반 차로변경에 맡기는 하이브리드다.
- **VTD 로 옮기기.** 학생은 오프라인 세계의 빈틈(접지 한계 없음, 녹색 한 가지)을 배웠다. VTD 에서 쓰려면 오프라인 세계에
  측가속 한계와 VTD 에서 식별한 옆 방향 응답, VTD 신호 상태(녹색+좌회전·점멸)를 넣고 다시 학습한 뒤 VTD 에서 다시 재야 한다.

## 배포 묶음으로 평가하기

그물 파일은 레포에 넣지 않는다(`runs/` 는 커밋하지 않는다). 지킴선(보고 창, 단계 ①②③a③b)과 큰 선택 창 점수는 아래처럼 잰다.
**앞당긴 제한속도를 꼭 켠다** — 끄면 보호구역 위반이 다시 는다.

    env -u PYTHONPATH .venv/bin/python scripts/eval_unseen.py --ckpt <배포 그물> --device cpu --lim-anticipate \
        --variants 4 --variant-offset 4 --eval-seeds 3 \
        --stage stage1 --stage stage2 --stage stage3a --stage stage3b --out runs/$(hostname)/guard.md

평가는 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 로 돌린다(여러 프로세스가 CPU 를 나눠 쓸 때 스레드가 겹치지 않게).

## 주의
`third_party/rule_stack` 서브모듈은 **비공개 레포**라 외부에서는 클론만으로 실행되지 않는다.
주최측 자료(지도 xodr, VTD 시나리오, 교육 자료)는 이 레포에 넣지 않는다.

## 설치
    git clone --recurse-submodules https://github.com/iam-raccoon/vtd-rl-urban-driving.git
    cd vtd-rl-urban-driving
    python3 -m venv .venv
    .venv/bin/pip install -e . -r requirements-dev.txt

## 테스트
테스트는 1175 개다. 한 번에 다 돌 수도 있지만(`env -u PYTHONPATH .venv/bin/pytest`, 조각 시간 합으로 25 분 남짓), 학습
테스트가 길어서 **세 조각**으로 나눠 돌리는 것을 권한다 — 한 번에 돌리면 도구·CI 의 실행 시간
상한에 걸린다.

    env -u PYTHONPATH .venv/bin/python -m pytest -q -m "not slow"                 # 1121개, 약 4분
    env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/ -q                     # 201개, 약 15분
    env -u PYTHONPATH .venv/bin/python -m pytest -q -m slow --ignore=tests/rl     # 30개, 약 7분

세 조각이 1175 개를 **덮지만 나누지는 않는다**: 첫 조각의 `-m "not slow"` 가 `tests/rl` 의 빠른 테스트
177 개까지 가져가서 둘째 조각과 겹친다(그 177 개는 두 번 돈다 — 빨라서 그냥 둔다). 합이 1352 인 것은
그래서다. 시간은 2026-10-07 lab-main 실측이다(다른 일과 CPU 를 나눠 쓴 채라 조금 길게 나왔을 수 있다).

**⚠ 둘째 조각(`tests/rl`)이 약 880 초로 도구 상한(600 초)을 넘는다** — 이 조각은 백그라운드로
돌리고 종료 코드를 파일로 받아야 한다. 셋째 조각도 430 초로 상한에 가깝다. 느린 테스트를 **여기
둘에는 더 넣지 말고** 첫 조각(빠른 조각)에 넣어라.

**판정은 종료 코드(`$?` = 0)로 한다** — `"N passed"` 를 grep 하면 같은 줄의 `"1 failed"` 를 삼킨다.
파이프(`| tail`)를 태우면 `$?` 가 pytest 가 아니라 `tail` 의 것이 되니, 출력을 파일로 받고 종료 코드를
따로 읽는다.

느린 조각을 채우는 것은 판을 끝까지 달리는 완주 테스트(`slow`, 단계 ① 판 6 개)와 PPO·DAgger 학습 스모크다.

ROS2 `setup.bash` 를 source 한 셸은 `PYTHONPATH` 에 `/opt/ros/humble/...` 가 들어 있다. 그러면 pytest 가
그곳의 플러그인(`launch_testing` 등)을 자동으로 불러오다 venv 에 없는 모듈(`yaml`)에서 시작도 못 한다.
`env -u PYTHONPATH` 로 그 변수만 빼고 돌린다(ROS2 를 source 하지 않은 셸은 `.venv/bin/pytest` 만으로 된다).

## 온라인 심판
`vtd_rl/referee` 는 주행 한 판을 한 프레임씩 받아 대회 채점기(`score_fma.py`)와 같은 감점 판정을 낸다.
판정 시점: ①②⑥⑦⑨⑪⑭⑮ 는 그 프레임에, ④⑤⑫ 는 위반이 최소 시간에 닿는 프레임에 낸다.
③ 은 최소 시간에 닿고 2.5 초 뒤, ⑩ 은 1 초 뒤, ⑬ 은 최대 8 초 뒤에 낸다.
⑧ 은 정차가 끝날 때 낸다(등급과 면책을 정차 전체로 정한다).
일치 검증 결과: [docs/reports/m2a-referee-parity.md](docs/reports/m2a-referee-parity.md)

    env -u PYTHONPATH .venv/bin/python scripts/referee_parity_report.py

## 강화학습 환경
`vtd_rl/env` 는 세계와 심판을 Gymnasium 환경(`VtdDriveEnv`)으로 묶는다. 관측은 자차·경로·차로계획·신호·
물체 16개의 정규화 벡터, 행동은 조향·가속 연속값과 지시등, 보상은 심판 감점으로 만든다(판단 10 Hz).
결과: [docs/reports/m2b-env.md](docs/reports/m2b-env.md)

    env -u PYTHONPATH .venv/bin/python scripts/run_m2b_env.py

## 모방학습(DAgger)
`vtd_rl/policy` 는 규칙 스택을 선생님 삼아 학생 신경망을 학습시킨다. 라운드마다 판을 모으고
(학생이 몰아도 정답은 그 프레임의 선생님 행동), 쌓인 데이터로 다시 학습한 뒤 학생 단독으로 평가한다.
결과: [docs/reports/m3-dagger.md](docs/reports/m3-dagger.md)

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py --out runs/$(hostname)/$(date +%F)-dagger

성적표는 기본이 `<--out>/report.md` 다. `docs/reports/` 아래 **커밋된** 성적표는 실행마다 새로 나오는
산출물이 아니라 그 마일스톤의 증거물이라, 다시 만들려면 경로를 `--report` 로 직접 적어야 한다
(`--report docs/reports/m3-dagger.md`).

이미 운전할 줄 아는 학생 위에 더 어려운 단계를 얹을 때는 `--init` 으로 그 체크포인트에서 출발한다.
안 주면 라운드마다 새 그물이라 운전을 처음부터 다시 배운다.

**⚠ 다만 `--init` 만으로는 안 된다.** 웜스타트는 정확히 작동하지만(0 에포크면 입력과 비트 동일),
**1 에포크만 학습해도 무너진다.** 원인은 학습률도 스쿼시 정렬도 아니고 **데이터 방향**이다 — 같은
거리를 무작위로 움직이면 83.3% 로 버티는데 단계 ③ 라벨 쪽으로 움직이면 0% 다. 손실에 **"알던 것을
유지하라" 는 항이 없어서**, 새 라벨이 옛 해와 불화하면 옛 해를 버리는 쪽이 항상 이긴다.
자세한 것은 [docs/reports/m4l-bc-damage.md](docs/reports/m4l-bc-damage.md).

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py \
        --out runs/$(hostname)/$(date +%F)-dagger-s3 --stage stage3 --variants 4 \
        --eval-stage stage1 --eval-stage stage2 --eval-stage stage3 \
        --init runs/omen/2026-09-26-m3-squash-warm/policy.pt

데이터·체크포인트는 `runs/` 아래에만 두고 커밋하지 않는다.

## 강화학습(PPO)
`vtd_rl/rl` 은 모방학습 학생을 출발점으로 PPO 를 돌린다. 보상은 온라인 심판의 감점(+ M6v 부터 곡선형 즉시 벌)이고,
M6 부터는 단계 ①(항상 초록)·②(신호 주기)·③(라바콘·정지 차량) 판을 섞어 학습한다. 마일스톤마다 성적표:

- [docs/reports/m4a-ppo.md](docs/reports/m4a-ppo.md) — PPO 첫 실행
- [docs/reports/m4b-ppo.md](docs/reports/m4b-ppo.md) — 왜 출발점을 못 넘는지 진단(승차감이 탐색을 벌한다)
- [docs/reports/m4c-split.md](docs/reports/m4c-split.md) — 판당 항목별 보상 실측(승차감 대 위반 배분)
- [docs/reports/m4c-refit.md](docs/reports/m4c-refit.md) — 학생을 tanh 스쿼시 매개화로 재적합
- [docs/reports/m4c-ppo.md](docs/reports/m4c-ppo.md) — 행동 상자를 닫고 승차감을 의도로 잰 뒤의 결과
- [docs/reports/m4d-ppo.md](docs/reports/m4d-ppo.md) — 보상의 위반 항을 대회 채점기와 맞춘 뒤의 결과
- [docs/reports/m4e-stall.md](docs/reports/m4e-stall.md) — 결정적 정책은 **왜** 멈추는가(진단)
- [docs/reports/m4e-ppo.md](docs/reports/m4e-ppo.md) — 그 개입 둘의 결과(모방 앵커 바닥 대 σ 어닐링)
- [docs/reports/m4f-ppo.md](docs/reports/m4f-ppo.md) — 개입 둘을 제대로 재고 **둘 다 기각**
- [docs/reports/m4g-probe.md](docs/reports/m4g-probe.md) — **★ PPO 는 학습 어느 시점에서도 출발점을 못 넘는다**
- [docs/reports/m4h-stage3-teacher.md](docs/reports/m4h-stage3-teacher.md) — **★ 단계 ③ 에서 선생님이 지는 지점**(+ [해석](docs/reports/m4h-stage3-notes.md))
- [docs/reports/m4i-student-stage3.md](docs/reports/m4i-student-stage3.md) — M3 학생은 단계 ③ 에서 **0/18**(전부 충돌)
- [docs/reports/m4j-obs-diag.md](docs/reports/m4j-obs-diag.md) — 학생은 물체를 **보고도 무시한다**(관측은 멀쩡하다)
- [docs/reports/m4l-bc-damage.md](docs/reports/m4l-bc-damage.md) — **★ BC 는 왜 좋은 정책을 파괴하는가**(손실에 '유지' 항이 없다)
- [docs/reports/m4m-anchor.md](docs/reports/m4m-anchor.md) — KL 앵커는 **유지에 성공**, 학습은 **회피가 데이터의 2%** 라 막힌다
- [docs/reports/m4n-weight.md](docs/reports/m4n-weight.md) — 그 2% 에 무게를 줘도 **안 배운다**(없는 상태엔 무게를 못 준다)
- [docs/reports/m4o-dagger-anchor.md](docs/reports/m4o-dagger-anchor.md) — **★ DAgger+앵커+선택: 처음으로 바늘이 움직였다**(유지 완전, 거리 +10~14%, 완주는 0)
- [docs/reports/m4p-stage3a.md](docs/reports/m4p-stage3a.md) — **★ 쉬운 단계 ③a: 처음으로 완주율이 0 이 아니다**(M3 16.7% → M4o 30%, 라바콘은 배웠고 정지차는 못 넘는다)
- [docs/reports/m4q-stage3b.md](docs/reports/m4q-stage3b.md) — **★ 정지차: 계수 10 앵커는 큰 변화를 막는다, 계수 3 에서 처음 넘었다**(정지차 완주 11.1%)
- [docs/reports/m4r-anchor3.md](docs/reports/m4r-anchor3.md) — 계수 3 DAgger: **안 쓴 판에서 처음 정지차 통과**(22.2 / 4.2 / 0.0%, 시드 하나만)
- [docs/reports/m4s-select-beta.md](docs/reports/m4s-select-beta.md) — **★ β 를 0.5 에서 멈추자 시드 다섯 모두 정지차 통과**(안 쓴 판 평균 37.8%, 30.6~43.1%) · 효과는 β 덕이고 선택 창은 같은 라운드를 골랐다
- [docs/reports/m4t-r4-drop.md](docs/reports/m4t-r4-drop.md) — r4 하락은 **운**이었다: 데이터·갱신 횟수를 바꿔 다시 학습한 그물 60 개 중 정지차 성공 23 개(38%), 원본 r3·r4 는 가중치까지 재현 · M4s 의 5/5 는 라운드 선택이 만든 것
- [docs/reports/m4u-ema.md](docs/reports/m4u-ema.md) — 가중치 평균(EMA)은 배치 시드 흔들림을 없애지만(표준편차 7.0 → 0.5 점) 평균은 그대로(③b 16%) · 흔들림은 궤적 끝 잡음이었고 선택이 그 꼬리를 고른다 · 평가는 CPU 로(GPU 와 걸음 수까지 같음)
- [docs/reports/m4v-ema-grid.md](docs/reports/m4v-ema-grid.md) — **★ 회피 표본 가중 4 배(30 m 안): 평균 그물의 정지차 통과 16% → 60%**(선택 창 EMA), raw 15/15 성공 · 안 쓴 판(보고 창)에서 EMA 55.5% · 고르지 않은 그물이 M4s 의 고른 그물(37.8%)보다 낫다
- [docs/reports/m4w-dagger-near.md](docs/reports/m4w-dagger-near.md) — 새 손실로 DAgger 를 다시 돌리자 **데이터가 나빠졌다**: 같은 설정 EMA 가 M4s 데이터 60.8% → M4w 데이터 15.6% · 근처 표본은 오히려 많다 → 표본의 종류(선생님 라벨) 가설
- [docs/reports/m4x-round-data.md](docs/reports/m4x-round-data.md) — 새 손실 학생이 모은 데이터는 **첫 라운드부터** 해롭다(라운드 0~1 만으로 35.3 대 12.0%) · M4s 데이터는 라운드 0~2 가 가장 좋다(66.4%) · 걸음마다 β 섞기가 선생님 계획기를 흔든다는 가설
- [docs/reports/m4y-best-data.md](docs/reports/m4y-best-data.md) — **★ 라운드 0~2 데이터 · 앵커 2 · 가중 3: 안 쓴 판(보고 창) 정지차 65.8%**(M4v 55.0% → 다섯 짝 모두 상승, ① 100%) · 정지차 흐름 1.4 → 8.8 → 37.8 → 55.0 → 65.8%
- [docs/reports/m4z-mix-unit.md](docs/reports/m4z-mix-unit.md) — 수집 β 를 걸음·3 초 구간·판 단위로 섞어도 새 손실 학생 데이터는 모두 나쁘다(7.2 / 12.2 / 6.1% 대 70.8%) → 섞는 단위가 아니라 **새 손실 학생이 만드는 상태**가 문제
- [docs/reports/m5a-near-states.md](docs/reports/m5a-near-states.md) — 정지차 앞 상태 비교(학습 없음): 선생님은 61% 서 있고, 옛 손실 학생은 덜 서고 달려들며(42%), 새 손실 학생은 선생님처럼 더 선다(51~55%) — 움직이며 다가가는 상태의 라벨이 필요하다는 가설
- [docs/reports/m5b-more-good-data.md](docs/reports/m5b-more-good-data.md) — 좋은 수집자로 r1·r2 만 두 배로 늘리자 **오히려 내려갔다**(선택 창 70.8 → 52.0%, 갱신 수를 묶어도 62.2%) · r0(선생님) 비중이 41 → 25~26% 로 준 탓이라는 가설
- [docs/reports/m5c-ratio.md](docs/reports/m5c-ratio.md) — 구성비가 아니라 **갱신 횟수**였다: 전 라운드 두 배 · 8 에폭 53.6% → 갱신을 M4y 와 같게 묶으면 67.5%(선택 창) · 데이터를 늘려도 기준(70.8%)을 못 넘는다
- [docs/reports/m5d-epochs.md](docs/reports/m5d-epochs.md) — 학습 길이 곡선은 산 모양(2·4·6·8·12 에폭 = 33 · 69 · 73 · 66 · 63%)이지만 6 에폭도 미리 적은 문턱(4/5 시드)을 못 넘는다 → 이 설정은 평탄한 구간
- [docs/reports/m5e-scorecard.md](docs/reports/m5e-scorecard.md) — **현재 최고 그물 종합 성적(보고 창)**: ① 100% · 98.5 / ② 100% · 93.8 / ③a 74.7% / ③b 65.8% / ③ 전체 3.6%(출발점 0%, 총걸음 2 배)
- [docs/reports/m6a-ppo-probe.md](docs/reports/m6a-ppo-probe.md) — 정지차 단계 PPO 탐침: **앵커 없는 PPO 는 배운 회피를 무너뜨린다**(선택 창 ③b 68.5 → 48.1%, 25 만~50 만 걸음 안에 무너짐) → 다음은 KL 앵커
- [docs/reports/m6b-ppo-anchor.md](docs/reports/m6b-ppo-anchor.md) — **KL 앵커를 단 PPO 가 처음으로 출발점을 크게 넘었다**: 계수 0.03 에서 ③b 선택 창 68.5 → 97.2% · 보고 창 59.7 → 92.6%(①② 100% 유지). 앵커 없이도 고른 체크포인트는 오르지만 끝점이 출렁인다. 덤: ③a 81.5 → 97.7%, ③ 전체 8.8 → 37.0%(선택 창, 학습에 안 쓴 단계)
- [docs/reports/m6c-ppo-stage3.md](docs/reports/m6c-ppo-stage3.md) — **앵커 PPO 를 장애물 여럿 단계로 넓혔다**: M6b 그물에서 ③a·③ 전체까지 학습해 ③ 전체 선택 창 37.0 → 68.5% · 보고 창 39.4 → 64.8%(③a·③b·①② 유지). 같은 보고 창에서 모방 최고(M4y) 3.3% → 64.8%. 남은 실패는 전부 충돌이고 1 위는 정지차 바로 뒤 라바콘
- [docs/reports/m6d-ppo-chain.md](docs/reports/m6d-ppo-chain.md) — 같은 판으로 앵커 PPO 를 한 번 더 이어도 **차이 없다**(③ 전체 선택 창 68.5 → 71.3%(계수 0.03) · 66.2%(0.01)), 시드 사이 편차만 커진다 → 다음은 변종 판으로 학습
- [docs/reports/m6e-ppo-variants.md](docs/reports/m6e-ppo-variants.md) — **변종 판(v0~v3)으로 학습하자 ③ 전체가 올랐다**: 선택 창 68.5 → 76.4% · 보고 창 64.8 → 71.8%(3/3), 같은 조건 원본 판만(M6d A 71.3 · 66.7%)보다 높고 끝점도 안정. ③a·③b·①② 유지. 남은 1 위 실패는 정지차 직후 라바콘
- [docs/reports/m6f-variant-pool.md](docs/reports/m6f-variant-pool.md) — 리셋마다 변종 판을 지어 풀 32·1000 으로 학습해도 **차이 없다**(③ 전체 선택 창 69.9 · 72.2%, 변종 4 개와 비슷). 두 번째 단계 네 칸이 모두 70% 근처라 M6e 의 +5 는 시드 운일 수 있다. 남은 실패는 정지차 직후 라바콘·course_E 오른쪽 라바콘 그대로
- [docs/reports/m6g-stage3c.md](docs/reports/m6g-stage3c.md) — 남은 장면 전용 학습 커리큘럼 ③c(선생님 72/72 완주)를 더해도 ③ 전체는 **차이 없다**(선택 창 70.8 · 보고 창 70.4%). 정지차 직후 라바콘 충돌은 15 → 5 판으로 줄었지만 다른 실패가 늘었다. 흔들림은 PPO 시드보다 출발 그물 갈래 차이가 크다
- [docs/reports/m6h-pick-best.md](docs/reports/m6h-pick-best.md) — **지금 가장 좋은 그물**(두 번째 단계 25 개 중 선택 창으로 고름, M6e V 시드 1): 보고 창 ① 100% · ② 100% · ③a 95.8% · ③b 93.1% · **③ 전체 80.6%**(모방 최고 M5e 3.6%). 출발 그물 갈래 차이(23 점)가 학습 판 칸 차이(10 점)의 두 배
- [docs/reports/m6i-lineage.md](docs/reports/m6i-lineage.md) — 갈래 가르기: 출발 그물을 고정하고 PPO 난수만 바꾸니 **둘 다 섞여 있다**. 갈래가 평균을 10~14 점 옮기고(선택·보고 창), 난수가 같은 갈래 안에서 15 점 흔든다. 최고 그물은 그대로
- [docs/reports/m6j-one-stage.md](docs/reports/m6j-one-stage.md) — M4y 에서 바로 다섯 단계+변종 4 로 **한 단계 학습하면 세 단계보다 낮다**(③ 전체 선택 창 63.9 대 76.4%, 보고 창 56.0 대 71.8%). 정지차 하나부터 넓히는 순서가 낫다. 처음 쓴 갈래 3·4 도 이 경로에서는 평범
- [docs/reports/m6k-ckpt-select.md](docs/reports/m6k-ckpt-select.md) — **지금 가장 좋은 그물(갱신)**: 중간 체크포인트까지 선택 창으로 고르니 M6g C 시드 1 · 150 만 걸음. 보고 창 ① 100% · ② 100% · ③a 100% · ③b 100% · **③ 전체 83.3%**(앞선 최고 80.6%, 차이는 잡음 안)
- [docs/reports/m6l-lineages34.md](docs/reports/m6l-lineages34.md) — 갈래 3·4 를 세 단계로 키워도 **갈래 1 을 못 넘는다**(③ 전체 선택 창 70.8 · 58.3%, 갈래 1 86.1%). 갈래 3·4 는 첫 단계(③b)부터 안 올랐다. 최고 그물 그대로
- [docs/reports/m6m-test-window.md](docs/reports/m6m-test-window.md) — **큰 시험 창(그물마다 384 판)으로 다시 재니 72 판 선택 창은 상위 그물을 거의 못 가른다**(순위 상관 0.17). 배포용 최고 그물 = M6g C 시드 1 마지막: ③ 전체 **87.8%**(시험 창), ①②③a③b 100%. 1~3 위(87~88%)는 구분이 안 된다
- [docs/reports/m6n-big-select.md](docs/reports/m6n-big-select.md) — **큰 선택 창(384 판)으로 고르고 새 시험 창으로 확인한 최고 그물**: M6g C 시드 1 마지막, 새 시험 창 ③ 전체 **88.0%**(84.4~90.9), 큰 창 셋에서 87~88% 로 일관. 큰 창으로 본 칸 평균은 72 판 판정과 대체로 같고, 변종 학습이 원본 판보다 8 점 높다
- [docs/reports/m6o-best-failures.md](docs/reports/m6o-best-failures.md) — 최고 그물의 남은 실패(384 판 중 50 판)는 **전부 두 코스**: course_A 라바콘 줄 바로 뒤 신호 교차로(64 판 모두 적색 위반, 40% 가 교차로 안에서 멈춤)와 course_G 마지막 정지차. 나머지 코스 넷은 100%
- [docs/reports/m6p-courseA-dagger.md](docs/reports/m6p-courseA-dagger.md) — course_A 선생님 라벨(DAgger)만으로 앵커 모방 다듬기: EMA 그물은 course_A 40.6 → 87.5%(멈춤 0) 지만 **다른 코스를 잃어**(course_A 밖 308 → 246/320) 지킴선 못 지킴 → 최고 그물 그대로. 좁은 데이터에는 앵커 계수 2 가 모자라다
- [docs/reports/m6q-ppo-imitation.md](docs/reports/m6q-ppo-imitation.md) — PPO 에 course_A 라벨 보조 모방을 섞으면 **무너진다**(앵커 KL 2 대). 모방 없이 이어 돌린 PPO 가 course_A 를 고친다: 후보 새 시험 창 89.3% 대 최고 그물 86.5%(+2.8, 교체 문턱 3 점 미달) → 최고 그물 그대로
- [docs/reports/m6r-continue-seeds.md](docs/reports/m6r-continue-seeds.md) — **새 최고 그물**: 최고 그물에서 모방 없이 PPO 를 시드 여섯으로 더 이어 돌려 체크포인트 32 개 중 큰 창으로 고름(M6r 시드 4 · 50 만). 새 시험 창 ③ 전체 **96.4%**(앞선 최고 90.1%, 같은 창), ①②③a③b 100%. course_A 멈춤 0
- [docs/reports/m6s-continue-again.md](docs/reports/m6s-continue-again.md) — 같은 방법을 한 번 더 이었더니 **차이가 잡음 안**(+1.3 점) → 최고 그물 그대로. 최고 그물은 새 창 둘에서 96.4 · 96.9%(합 96.6%). 이어 돌릴수록 ①② 점수가 내려가 다음 축은 점수
- [docs/reports/m6t-score-sheet.md](docs/reports/m6t-score-sheet.md) — 점수 진단: **학생 그물은 모두(모방 포함) 적색 신호를 거의 매번 지나간다**(완주 판당 적색 중대 3.6 건, 선생님 0.09). ② 점수 89.7~93.8 대 선생님 99.5 의 대부분이 이것. γ=0.99 할인으로는 서는 쪽이 손해라 PPO 가 못 고친다(추정)
- [docs/reports/m6u-red-light.md](docs/reports/m6u-red-light.md) — 적색 위반 벌을 5 배·10 배로 키워도 **적색 중대가 그대로**(완주 판당 3.5, 대조와 같음). 끝까지 서 보지 않으니 덜 빨리 지나가도 벌이 같아 기울기가 없다. 고른 그물은 새 시험 창 +2.18 점(문턱 3 점 미달) → 최고 그물 그대로
- [docs/reports/m6v-red-profile.md](docs/reports/m6v-red-profile.md) — 적색 감속 곡선 벌(c=0.3)은 **서는 법을 가르친다**(체크포인트 3 개가 적색 중대 3.6 → 1.4). 하지만 과속·멀리 서기·충돌이 늘어 점수는 그대로이고, 배운 것이 다음 체크포인트에서 사라진다. 고른 그물 −1.96 점 → 최고 그물 그대로
- [docs/reports/m6w-red-hold.md](docs/reports/m6w-red-hold.md) — **새 최고 그물**: 감속 곡선을 세게(c=1.0) + 과속 벌 3 배로 하자 세 시드 모두 적색에 서고, 체크포인트 48 개 중 큰 창으로 고른 그물(M6w W10 시드 0 · 75 만)이 새 시험 창 완주 포함 점수 **85.50 대 81.44(+4.07)**, 완주 97.1%, 적색 중대 3.63 → 1.88. 다음 큰 몫은 차로 유지(완주 판당 중대 3.9)
- [docs/reports/m6x-lane-profile.md](docs/reports/m6x-lane-profile.md) — 채점기가 세는 차로 침범 깊이에 벌(2.5 초 늦게 옴)을 줘도 **차로 유지가 안 준다**(대조 4.14 · c=5 4.02 · c=20 3.72). 앵커를 새 최고로 바꾸자 적색은 칸 평균 0.83~0.88 건으로 더 줄었다. 고른 그물 −0.11 점 → 최고 그물 그대로
- [docs/reports/m6y-lateral-profile.md](docs/reports/m6y-lateral-profile.md) — 경로 기준 횡오차에 **바로 오는** 벌(c=0.5)은 차로 유지를 크게 줄인다(칸 평균 4.14 → 2.41, 단계 ② 3.67 → 0.28). 고른 그물은 완주 판 점수 93.4(최고 88.1)·단계 ② 96.7(선생님 99.5)이지만 course_G 충돌 24 판으로 새 시험 창은 같다(−0.09) → 최고 그물 그대로
- [docs/reports/m6z-continue-lat.md](docs/reports/m6z-continue-lat.md) — 차로를 지키는 그물에서 이어 돌리니 **완주한 판의 운전은 선생님 수준**(단계 ② 98.5 · 선생님 99.5, 시험 창 완주 판 95.4)이지만 충돌·이탈·정체가 늘어 완주가 흔들린다(92.2%). 새 시험 창 +1.94(문턱 미달) → 최고 그물 그대로. 판당 되돌림이 음수가 돼 실패를 피할 유인이 약해진 것으로 보인다
- [docs/reports/m7a-failure-penalty.md](docs/reports/m7a-failure-penalty.md) — 실패 벌을 키워도(충돌·이탈·정체 −200·−500) **완주는 안 돌아온다**: 충돌이 줄면 장애물 앞 정체가 는다(정체 벌은 600 걸음 뒤라 할인으로 거의 0). 고른 그물 −2.32 점 → 최고 그물 그대로
- [docs/reports/m7b-lat-free-80.md](docs/reports/m7b-lat-free-80.md) — **새 최고 그물**(M7b Fz 시드 1 · 50 만): 횡오차 벌을 끄는 거리 80 m 로 이어 돌려 체크포인트 48 개 중 큰 창으로 고름. 새 시험 창 완주 포함 점수 **90.67 대 87.28(+3.39)**, 완주 판 점수 94.4(앞선 88.0), 적색 중대 1.90 → 0.29, 차로 유지 3.92 → 1.73, 완주 96.1%(앞선 99.2%)
- [docs/reports/m7c-hard-courses.md](docs/reports/m7c-hard-courses.md) — 학습 판에 장애물 판(③c)·어려운 코스 덧보기(③h)를 더해도 **충돌은 안 준다**(대조 53.9 · ③c 68.8 · ③h 72.9 판). 고른 그물 +0.57 점 → 최고 그물 그대로. 끈질긴 실패는 course_G 4 번째 정지차
- [docs/reports/m7d-obstacle-profile.md](docs/reports/m7d-obstacle-profile.md) — **새 최고 그물**(M7d O10 시드 1 · 50 만): 앞길 위 장애물 감속 곡선(통로 안 물체 앞에 설 수 있는 속도 초과분에 벌)으로 학생이 장애물에 반응. 새 시험 창 완주 **98.2%**, 완주 포함 점수 **93.58 대 88.94(+4.63)**, 완주 판 점수 95.3, 단계 ② 98.5(선생님 99.5)
- [docs/reports/m7e-continue-obs.md](docs/reports/m7e-continue-obs.md) — 새 최고 그물에서 이어 돌리니 큰 창에서 넘은 체크포인트 1/48, 새 시험 창 +2.56(문턱 미달) → 최고 그물 그대로. course_G 4 번째 정지차가 여전히 남은 실패의 큰 몫
- [docs/reports/m7f-variant-pool.md](docs/reports/m7f-variant-pool.md) — 변종 풀 1000 개로 학습해도 course_G 는 그대로(40.2 대 39.5). 고른 그물(M7f 시드 2 · 12.5 만)이 새 시험 창 +3.25 로 문턱을 넘어 **규칙대로 교체**했지만 큰 선택 창에서는 −0.38 이라 근거는 약하다(앞선 최고가 창마다 88.9~93.6 으로 흔들림). 다음부터 교체 규칙을 조인다
- [docs/reports/m7g-block-penalty.md](docs/reports/m7g-block-penalty.md) — 장애물 뒤에 머무는 벌(c=1.0)은 칸 평균으로 충돌을 줄이고 완주를 늘린다(충돌 57.3 → 37.6, 완주 312.7 → 333.2). course_G 는 기준에 조금 못 미침(+7.4). 고른 그물 +0.89 → 최고 그물 그대로
- [docs/reports/m7h-lim-anticipate.md](docs/reports/m7h-lim-anticipate.md) — **앞당긴 제한속도 관측**(앞에 낮은 제한속도가 있으면 미리 낮아진 값): 학습 없이 켜기만 해도 보호구역 중대 0.69 → 0.09(−87%). 같은 그물에서 두 창 모두 +0.9 점으로 일관되지만 교체 문턱 3 점 미달 → 배포 묶음 그대로
- [docs/reports/m7i-overlap-profile.md](docs/reports/m7i-overlap-profile.md) — 통로와 겹친 폭 벌(c=2)은 칸 평균 충돌을 줄인다(57.3 → 35.1, 완주 312.7 → 338.9). 고른 그물 +0.50 → 최고 그물 그대로. 점수는 큰 창 90 근처에서 정체
- [docs/reports/m7j-combine.md](docs/reports/m7j-combine.md) — **배포 묶음 갱신**: M7f 시드 2 · 12.5 만 + 앞당긴 제한속도 관측(미리 정한 짝 비교 기준으로 새 창 v22000 +0.69, 완주 같음, 보호구역 중대 0.70 → 0.08). 머무는 벌·겹친 폭 벌·앞당김을 묶어 이어 돌려도 지금 최고를 넘는 체크포인트 없음
- [docs/reports/m7k-obstacle-imitation.md](docs/reports/m7k-obstacle-imitation.md) — 장애물 앞 장면 DAgger 행(β 0.5)으로 작은 계수(0.02·0.05) 보조 모방을 섞자 **정책이 무너졌다**(완주 3/384, 정체 312). 고른 행의 46% 가 장애물 뒤에 서 있는 상태라 "멈춤" 을 배웠다. 최고 그물 그대로
- [docs/reports/m7l-scene-slices.md](docs/reports/m7l-scene-slices.md) — 어려운 장면 바로 앞에서 출발하는 짧은 판(판 항목 `slice`, `stage3s`)을 섞어도 course_G 는 그대로(47.5 대 46.1). 큰 창에서 넘은 그물은 새 시험 창 −1.95 → 최고 그물 그대로. 완주 포함 점수는 시험 창 91~93 에서 정체
- [docs/reports/final-summary.md](docs/reports/final-summary.md) — **마무리 성적표**: 시험 창 셋(1152 판)에서 학생(배포 묶음) 완주 96.4% · 완주 포함 점수 92.38, 선생님 97.7% · 94.65. 학생은 정지차 충돌로, 선생님은 정체로 진다. VTD 주행기(`vtd_rl/bridge`)를 더했고, 오프라인 세계에 물리면 환경과 같은 판을 낸다
- [docs/reports/vtd-validation.md](docs/reports/vtd-validation.md) — **VTD 검증**: 선생님은 VTD 에서도 그대로(신호만 6/6 · 99.6), 학생은 2/6. 원인은 접지 한계 없는 오프라인 동역학·VTD 언더스티어(K≈0.016)·처음 보는 신호 상태(녹색+좌회전). 어댑터(신호 읽기 + 조향 보정)로 신호만 5/6 이지만 교차로 과속·정지차 충돌은 남는다

### 학습 실행
M7 의 이어 돌리기는 아래 꼴이다(M7h 실험 B 설정). 출발 그물을 `--init` 으로 주고 같은 그물에 KL 앵커를 건다.
체크포인트는 12.5 만 걸음마다 `policy-<걸음>.pt` 로 남고, 그물 고르기는 큰 선택 창(v1100~v1131 × 시드 2, 384 판)으로 한다.

    env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py --out runs/$(hostname)/$(date +%F)-ppo --init <출발 그물> \
      --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
      --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
      --violation-mode once_per_section --comfort-on-intent --seed 0 \
      --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
      --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0 --lim-anticipate

M4 시기(단계 ①② 에서 PPO 가 모방 학생을 못 넘던 때)의 진단은 위 목록의 M4a~M4g 성적표에 있다.
그때의 성적표는 `scripts/report_m4a.py` 가 실행이 남긴 `log.jsonl`·`ac-best.pt` 만으로 만들었다.

데이터·체크포인트는 `runs/` 아래에만 두고 커밋하지 않는다.

## M1 성적표 다시 만들기
[docs/reports/m1-stage1-teacher.md](docs/reports/m1-stage1-teacher.md) 는 아래 두 줄로 만든다
(스텝 속도를 먼저 재고, 그 JSON 을 넘겨 단계 ① 판을 달린다. 주행 CSV 는 `runs/m1/` 에 남고 커밋하지 않는다).

    B=$(env -u PYTHONPATH .venv/bin/python scripts/bench_world.py --board course_H --seconds 120)
    env -u PYTHONPATH .venv/bin/python scripts/run_stage1_teacher.py --bench "$B"

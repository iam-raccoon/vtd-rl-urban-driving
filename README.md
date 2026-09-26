# vtd-rl-urban-driving

Hexagon VTD 2025.2 의 도심 지도(LivingLab)에서 도로교통법을 지키며 달리는 **end-to-end 강화학습 운전 정책**을 만든다.
2026 HL-FMA 대회에서 완주한 규칙 기반 주행 스택을 선생님으로 두고, 오프라인 세계에서 모방학습(DAgger) → PPO 로 학습한다.

- 설계: [docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md](docs/superpowers/specs/2026-09-15-vtd-rl-urban-driving-design.md)
- 규칙 스택 공개 참고본: [HL-FMA2026-VTD](https://github.com/iamracco0n/HL-FMA2026-VTD)

## 주의
`third_party/rule_stack` 서브모듈은 **비공개 레포**라 외부에서는 클론만으로 실행되지 않는다.
주최측 자료(지도 xodr, VTD 시나리오, 교육 자료)는 이 레포에 넣지 않는다.

## 설치
    git clone --recurse-submodules https://github.com/iamracco0n/vtd-rl-urban-driving.git
    cd vtd-rl-urban-driving
    python3 -m venv .venv
    .venv/bin/pip install -e . -r requirements-dev.txt

## 테스트
    env -u PYTHONPATH .venv/bin/pytest

ROS2 `setup.bash` 를 source 한 셸은 `PYTHONPATH` 에 `/opt/ros/humble/...` 가 들어 있다. 그러면 pytest 가
그곳의 플러그인(`launch_testing` 등)을 자동으로 불러오다 venv 에 없는 모듈(`yaml`)에서 시작도 못 한다.
`env -u PYTHONPATH` 로 그 변수만 빼고 돌린다(ROS2 를 source 하지 않은 셸은 `.venv/bin/pytest` 만으로 된다).
단계 ① 완주 테스트(`slow`, 판 6개를 끝까지 달린다)도 기본으로 함께 돈다 — 전체 1 분 안쪽.

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

데이터·체크포인트는 `runs/` 아래에만 두고 커밋하지 않는다.

## 강화학습(PPO)
`vtd_rl/rl` 은 M3 모방학습 학생을 출발점으로 PPO 를 돌린다. 보상은 온라인 심판의 감점이고,
단계 ①(항상 초록)과 ②(신호 주기) 판을 섞어 학습한다.

- [docs/reports/m4a-ppo.md](docs/reports/m4a-ppo.md) — PPO 첫 실행
- [docs/reports/m4b-ppo.md](docs/reports/m4b-ppo.md) — 왜 출발점을 못 넘는지 진단(승차감이 탐색을 벌한다)
- [docs/reports/m4c-split.md](docs/reports/m4c-split.md) — 판당 항목별 보상 실측(승차감 대 위반 배분)
- [docs/reports/m4c-refit.md](docs/reports/m4c-refit.md) — 학생을 tanh 스쿼시 매개화로 재적합
- [docs/reports/m4c-ppo.md](docs/reports/m4c-ppo.md) — 행동 상자를 닫고 승차감을 의도로 잰 뒤의 결과

    env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py --out runs/$(hostname)/$(date +%F)-ppo \
      --init <M3 체크포인트> --dagger-data <M3 데이터 폴더>

성적표는 `scripts/report_m4a.py` 가 위 실행이 남긴 `log.jsonl`·`ac-best.pt` 만으로 만든다(숫자를
손으로 옮겨 적지 않는다) — M3 체크포인트와 선생님 대비 표, 항목별 위반 표, 목표 네 줄 판정까지 전부
그 스크립트의 산출물이다.

    env -u PYTHONPATH .venv/bin/python scripts/report_m4a.py --run runs/omen/<날짜>-ppo \
      --m3 <M3 체크포인트> --out docs/reports/m4a-ppo.md --eval-seeds 3

데이터·체크포인트는 `runs/` 아래에만 두고 커밋하지 않는다.

## M1 성적표 다시 만들기
[docs/reports/m1-stage1-teacher.md](docs/reports/m1-stage1-teacher.md) 는 아래 두 줄로 만든다
(스텝 속도를 먼저 재고, 그 JSON 을 넘겨 단계 ① 판을 달린다. 주행 CSV 는 `runs/m1/` 에 남고 커밋하지 않는다).

    B=$(env -u PYTHONPATH .venv/bin/python scripts/bench_world.py --board course_H --seconds 120)
    env -u PYTHONPATH .venv/bin/python scripts/run_stage1_teacher.py --bench "$B"

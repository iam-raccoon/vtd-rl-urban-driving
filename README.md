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
테스트는 663 개다. 한 번에 다 돌 수도 있지만(`env -u PYTHONPATH .venv/bin/pytest`, 약 20 분), 학습
테스트가 길어져서 **세 조각**으로 나눠 돌리는 것을 권한다 — 한 번에 돌리면 도구·CI 의 실행 시간
상한에 걸린다.

    env -u PYTHONPATH .venv/bin/python -m pytest -q -m "not slow"                 # 612개, 약 1분 45초
    env -u PYTHONPATH .venv/bin/python -m pytest tests/rl/ -q                     # 148개, 약 12분
    env -u PYTHONPATH .venv/bin/python -m pytest -q -m slow --ignore=tests/rl     # 30개, 약 7분

세 조각이 663 개를 **덮지만 나누지는 않는다**: 첫 조각의 `-m "not slow"` 가 `tests/rl` 의 빠른 테스트
127 개까지 가져가서 둘째 조각과 겹친다(그 127 개는 두 번 돈다 — 빨라서 그냥 둔다). 합이 790 인 것은
그래서다.

**⚠ 둘째 조각(`tests/rl`)이 718 초로 도구 상한(600 초)을 이미 넘었다** — 이 조각은 백그라운드로
돌리고 종료 코드를 파일로 받아야 한다. 셋째 조각도 427 초로 상한에 가깝다. 느린 테스트를 **여기
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
`vtd_rl/rl` 은 M3 모방학습 학생을 출발점으로 PPO 를 돌린다. 보상은 온라인 심판의 감점이고,
단계 ①(항상 초록)과 ②(신호 주기) 판을 섞어 학습한다.

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

### 여기까지 온 곳
**PPO 는 M3 모방 학생을 한 번도 못 넘었다 — 학습의 어느 시점에서도.** 목표 판정은 M4a 부터 지금까지
0/2 다. 실행 **13 개 전부** 자기 최선 체크포인트에서 출발점 아래이고, 5 만 스텝 간격으로 촘촘히 본
관측에서도 **20 개 평가 지점 중 출발점을 넘은 것이 하나도 없다**(`m4g-probe.md`). 즉 지금까지의
마일스톤들이 다룬 것은 **이미 음수인 것을 덜 음수로 만드는 일**이었다. 그래도 그 과정에서 두 가지가
풀렸고, 아래 기전들은 남는다.

보상은 이제 대회 채점기와 같은 셈을 한다(M4d — 위반을 **구간당 한 번**만 물린다. 채점기와 무작위 순열
4000 개를 대조해 불일치 0). 그러자 판당 롤아웃 리턴이 **−65.3 → +12.0 으로 이 프로젝트 첫 양수**가 됐다
— PPO 의 목적함수가 처음으로 과제와 같은 방향을 가리켰다는 뜻이다.

남은 병목은 **결정적 모드**다: 학습된 정책이 확률적으로는 판을 거의 100% 완주하는데 **평균 행동으로는
22~48%** 만 완주하고, 실패는 **전부 정체**다(시간초과 0). 배포되는 것은 평균이므로 이것이 지금의 1 순위다.
원인은 "가속이 모자라서" 가 아니라 **제동을 학습해서**다 — 정체 직전 100 걸음의 가속이 −0.81~−0.87 로
거의 최대 제동이고, 학습이 진행될수록 평균 가속이 선생님(0.129) 아래 0 쪽으로 내려간다.

그 병목에 개입 둘을 걸어 봤고(모방 앵커 바닥·σ 후반 어닐링) **M4f 에서 둘 다 기각됐다.** σ 어닐링은
결정적 완주율을 단계① −27.8pp 떨어뜨리고, **확률적 완주율까지 100% → 81.5% 로 무너뜨린다.** 모방 바닥은
M4e 가 시드 3 개에서 본 +22.2pp 가 **시드 6 개로 늘리자 사라졌다**(단계① −2.8pp, p=1.000).

**σ 어닐링이 남긴 것이 지금 가장 쓸모 있는 단서다.** 정체 직전 구간에서 확률 표본은 평균보다 **+0.151
만큼 덜 제동**하는데, σ 를 하한까지 몰면 그 여유가 **+0.005** 로 사라진다. 즉 **잡음은 탐색이 아니라
정책을 정체에서 꺼내 주던 버팀목**이었고, 줄이면 나쁜 평균이 고쳐지는 게 아니라 드러난다.

그리고 M4e·M4f 실행 **21 개 전부** 최선 체크포인트가 0.5M~1M 이다 — 3M 을 돌리는데 **나머지 2M 은 정책을
나쁘게만 만든다.**

**★ 다만 그 결론은 단계 ①② 한정이다.** 스펙의 M4 기준은 "단계 ⑤ 진급" 인데(`design.md:233`) 그동안
존재한 판은 ①② 뿐이었고, 거기서는 **선생님이 99.6 / 99.5 라 이길 여지가 거의 없었다.** M4h 에서 단계 ③
(사물·정지차) 판을 만들어 재 보니 **선생님이 96.8 로 −2.8 점을 잃고, 그 대부분이 장애물을 피하면서
지시등을 안 켜는 것**이다(항목 ⑬, 42 슬롯, ③ 에서 새로 열림). 지시등은 **이미 정책의 행동 출력**이라,
규칙 스택이 확실히 틀리고 정책이 고칠 수 있는 결정이 처음으로 나왔다. **모방학습은 이 실수를 그대로
배우므로, 여기서 PPO 가 학생을 넘는지가 진짜 시험이다.**

**다만 학생은 아직 단계 ③ 을 달리지도 못한다** — 18/18 전부 첫 액터에서 충돌한다(`m4i`). 이유는
관측 버그가 아니다: 물체는 80 m 에서 정확히 들어와 충돌 직전까지 55~125 걸음 살아 있는데, 그 동안
**정책의 조향·가속이 소수 둘째 자리까지 안 움직인다**(`m4j`). M3 DAgger 데이터가 전부 물체 0 개인
단계 ①② 에서 나와 **물체 슬롯 16 개가 학습 내내 마스크였기 때문**이다. 그래서 다음은 단계 ③ 에서의
**DAgger 재수집**이고, 첫 목표는 지시등이 아니라 **"안 박고 끝내기"** 다.

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

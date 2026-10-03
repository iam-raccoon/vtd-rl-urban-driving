# M6a — 정지차 단계 PPO 탐침(코드 변경 없음)

**Goal:** 지금까지 PPO(M4a~M4g)는 단계 ①② 에서만 돌았고, 출발 그물이 이미 98 점대라 12 번 모두 출발점보다 못했다. 정지차 단계(③b)는 출발점이 66% 라 여지가 크다. **지금 있는 `train_ppo.py` 를 그대로** 써서 M4y 최고 그물에서 PPO 를 돌리면 안 쓴 판의 정지차 성적이 오르는지 싸게 본다. 코드를 고치기(KL 앵커, 변종 판 학습 등) 전에 방향이 맞는지 확인하는 탐침이다.

**Spec:** 이 문서. 직전 증거 `docs/reports/m5e-scorecard.md`, PPO 이력 `docs/reports/m4a~m4g`.

## 본 실행 (OMEN)

데이터 시드 `S ∈ {0, 1, 2}` 마다:

```
OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-04-m6a-s$S --init runs/omen/2026-10-04-m4y/K2-a2-w3-s$S/policy-ema2500.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3b.json \
  --steps 1000000 --eval-every 250000 --eval-seeds 1 --final-eval-seeds 3 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S
```

- 세 시드를 **동시에**(시드마다 환경 10 개) 돌린다. 중간 평가(단계마다 원본 판 6 개를 한 프로세스가 차례로 몬다)가 학습보다 오래 걸려 25 만 걸음마다로 줄였다. `--smoke` 로 명령·체크포인트·정책망 꺼내기를 먼저 확인했다.

- 학습 판은 단계 ①·②·③b 원본 판 18 개(③b 원본은 v0 와 같은 배치 — 수집 창 안). 판은 리셋마다 무작위.
- 보상은 M4d 에서 고친 설정(위반은 구간당 한 번, 승차감은 평균 행동 변화로).
- 모방 앵커(`--dagger-data`)는 **안 쓴다** — PPO 의 모방 손실은 회피 가중 없는 기본 손실이라, 정지차를 못 넘던 그물(15.8%) 쪽으로 끌어당긴다.
- 끝나면 `ac-best.pt` 와 `ac-1000000.pt`(마지막)에서 정책망을 꺼내(`ActorCritic.load(…).policy.save(…)`) `eval_unseen.py --device cpu` 로 단계 ①·②(원본)·③b(선택 창 v8~v11·보고 창 v4~v7)를 잰다.

## 판정 기준 (미리 적는다)

출발점 = M4y 승자 EMA(같은 시드)의 선택 창 ③b: 73.6 · 69.4 · 62.5(평균 68.5%), 보고 창 61.1 · 59.7 · 58.3(평균 59.7%).

PPO 그물은 시드마다 `ac-best` 와 마지막 중 **선택 창 ③b 가 높은 쪽**을 고른다(보고 창은 고르기에 안 쓴다).

- **오른다** = 고른 그물의 선택 창 ③b 평균 ≥ 73.5%(+5) **그리고** 세 시드 중 2 개 이상이 출발점보다 높음 **그리고** ①·② 완주율 평균이 둘 다 ≥ 90%.
- **내린다** = 선택 창 ③b 평균 ≤ 63.5%(−5) 이거나 ① 또는 ② 평균 < 90%.
- 아니면 **차이 없다**.

보고 창 숫자는 판정과 별도로 적는다.

## 이 탐침이 미루는 것

- PPO 에 KL 앵커 넣기, 변종 판으로 학습하기, 단계 ③ 전체로 넓히기 — 이 탐침 결과를 보고 정한다.

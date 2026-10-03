# M4w — 새 손실(회피 가중 4)로 DAgger 다시

**Goal:** M4v 의 승자 설정(앵커 3 · 전방 30 m 안 표본 손실 × 4)은 **M4s 학생들이 모은 데이터**로 학습했는데도 안 쓴 판에서 정지차 55% 를 넘었다. 이 설정으로 DAgger 를 처음부터 다시 돌려, **정지차를 절반 넘는 학생이 모은 데이터**가 더 나은 그물을 만드는지 잰다.

**코드 변경 없음.** `run_dagger.py` 는 이미 `--near-m`·`--near-weight` 를 받는다(`run_dagger.py:984-986`). 사후 측정은 `refit_round.py --ema-halflife` 와 `eval_unseen.py --device cpu` 로 한다.

**Spec:** 이 문서. 직전 증거 `docs/reports/m4v-ema-grid.md`.

## 본 실행 (OMEN)

M4s 와 같고 손실만 다르다.

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py \
  --out runs/omen/2026-10-03-m4w-s$S --stage stage3b --variants 4 \
  --eval-stage stage1 --eval-stage stage3a --eval-stage stage3b \
  --init runs/omen/2026-10-01-m4o-s0/policy-best.pt --anchor-coef 3 --near-m 30 --near-weight 4 \
  --betas 1.0,0.5,0.5,0.5,0.5 --select best --select-variants 4 \
  --rounds 5 --seeds 2 --epochs 8 --eval-seeds 3 --workers 5 --train-seed $S
```

- 학습 시드 `S ∈ {0..4}` 다섯 개를 **동시에** 돌린다(갈래마다 수집 워커 5 개). M4s 는 하나씩 워커 24 개였다.
- `OMP_NUM_THREADS=1`: 수집 워커는 학생 정책을 CPU 로 돌린다. 스레드 경합을 막으려고 준다. M4u 에서 CPU 평가 결과가 스레드 수와 상관없이 GPU 와 걸음 수까지 같음을 확인했다.
- 각 실행이 끝나면 `policy-best.pt` 를 **보고 창 v4~v7** 로 잰다(`eval_unseen.py --device cpu`).

## 사후 측정 — 데이터가 나아졌나

각 실행의 라운드 0~4 데이터(D4′)로 M4v 승자 설정 그대로 EMA 그물을 하나 학습한다.

```
refit_round.py --run runs/omen/2026-10-03-m4w-s$S --upto 4 --init runs/omen/2026-10-01-m4o-s0/policy-best.pt \
  --anchor-coef 3 --near-m 30 --near-weight 4 --batch-seed $((100+S)) --ema-halflife 2500 --no-eval \
  --out runs/omen/2026-10-03-m4w-refit/s$S
```

EMA 그물을 선택 창 v8~v11 과 보고 창 v4~v7 로 잰다. 비교 대상은 M4v 의 `a3-w4-s{S}`(M4s 의 D4 로 같은 명령) EMA 다.

## 판정 기준 (미리 적는다)

**주 판정 — 새 학생의 데이터가 더 나은가.** 잣대는 EMA2500 ③b 완주율(선택 창)이다. M4w 데이터 시드 `S` 의 값을 M4v `a3-w4-s{S}` 와 짝지운다. M4v 값은 59.7 · 61.1 · 56.9 · 62.5 · 63.9, 평균 60.8% 다.

- **낫다** = 평균 ≥ 64.8%(+4 점) **그리고** 다섯 짝 중 4 개 이상에서 높다 **그리고** ① 평균 ≥ 90%.
- **나쁘다** = 평균 ≤ 56.8%(−4 점).
- 둘 다 아니면 **차이 없다**.

**보조(판정 아님):** 채택 그물(라운드 선택)의 보고 창 ③b 평균을 M4s 채택 그물(37.8%)·M4v 승자 EMA(55.5%)와 나란히 적는다. EMA 그물의 보고 창 숫자도 적는다.

## 이 계획이 미루는 것

- `run_dagger` 안에서 EMA 그물을 최종 그물로 쓰기(지금은 사후 refit 으로 잰다).
- 회피 가중·거리 촘촘히, 단계 ③ 전체, ⑬ 지시등.

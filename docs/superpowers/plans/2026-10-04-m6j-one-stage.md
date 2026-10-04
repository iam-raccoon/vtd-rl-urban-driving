# M6j — 한 단계로: M4y 에서 바로 다섯 단계 커리큘럼 + 변종 4 로 앵커 PPO, 갈래 다섯 (코드 변경 없음)

**Goal:** 지금 최고 그물은 세 단계를 거쳤다. M4y(모방) → M6b(③b, 100 만 걸음) → M6c(+③a·③ 전체, 200 만) → M6e 설정(+변종 4, 200 만)이다. M6i 에서 출발 그물 갈래가 평균을 10~14 점 옮긴다는 것이 나왔지만, 갈래는 셋(M4y 데이터 시드 0~2)뿐이었다. M6j 는 **M4y 에서 바로** 다섯 단계 커리큘럼과 변종 4 로 한 번에 학습해 두 가지를 본다.
- 단계 나누기가 필요한가.
- 처음 쓰는 갈래 둘(M4y 데이터 시드 3·4)을 더한 갈래 다섯에서 더 좋은 그물이 나오나.

**Spec:** 이 문서. 직전 증거 `docs/reports/m6i-lineage.md`("다음" 1).

## 본 실행 (OMEN)

갈래 `L ∈ {0, 1, 2, 3, 4}` = 5 실행:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-04-m6j-L$L --init runs/omen/2026-10-04-m4y/K2-a2-w3-s$L/policy-ema2500.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 \
  --steps 4000000 --eval-every 1000000 --eval-seeds 1 --final-eval-seeds 3 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $L \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20
```

- 앵커 참조는 M4y 그물이다(M6b 와 같다).
- 400 만 걸음이다. 세 단계 경로는 모두 500 만 걸음이었다(100 만 + 200 만 + 200 만).
- 실행 하나 RAM 약 5.5 GB. 사용 가능 메모리 ≥ 7 GB 일 때만 하나씩 띄운다.
- 평가: M6e 와 같다. `ac-best`·마지막을 ①②(원본), ③a·③b·③(선택 창 v8~v11·보고 창 v4~v7)로 잰다.

## 판정 기준 (미리 적는다)

실행마다 `ac-best` 와 마지막 중 선택 창 ③ 전체가 높은 쪽을 고른다.

- **세 단계와 견주기(갈래 0~2, 같은 M4y 출발):** 세 단계 결과는 M6e V 다. 선택 창 ③ 전체 80.6 · 86.1 · 62.5(평균 76.4), 보고 창 73.6 · 80.6 · 61.1(평균 71.8).
  - "한 단계가 더 낫다": 갈래 0~2 평균이 선택 창 ≥ 81.4 **그리고** 보고 창 ≥ 76.8.
  - "세 단계가 낫다": 선택 창 평균 < 71.4.
  - 아니면 "한 단계로 충분하다".
- **지킴선:** 고른 그물의 ①·② 완주율 평균 ≥ 90%, ③a·③b 선택 창 평균 ≥ 90%. 못 지키면 위 판정 앞에 적는다.
- **갈래 다섯:** 갈래마다 고른 그물의 선택 창·보고 창 ③ 전체를 적는다.
- **최고 그물:** 새 그물을 M6h 고르기 후보에 더해 같은 규칙으로 다시 고른다. 바뀌면 바뀐 그물의 보고 창 종합 성적을 적는다.

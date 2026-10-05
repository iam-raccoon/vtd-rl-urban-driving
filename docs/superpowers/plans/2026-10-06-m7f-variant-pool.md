# M7f — 학습 판을 변종 풀 1000 개에서: 경로 쪽에 선 장애물을 더 자주 보게 (코드 변경 없음)

**Goal:** 지금 최고 그물(M7d O10 시드 1 · 50 만)의 큰 선택 창 실패 20 판 중 8 판이 course_G 4 번째 정지차다. 정지차가 경로 쪽(횡 −0.35~−0.59)에 선 판에서, 장애물 곡선을 배운 뒤에도 13 m/s 그대로 다가가 부딪친다. 지금 학습 판은 코스마다 변종 4 개(v0~v3)뿐이다. course_G 4 번째 정지차의 횡 위치는 v0~v3 에서 −0.90 · −0.59 · −0.51 · −1.11 이고, 변종 풀 v12~v1011 에서는 평균 −0.89, 경로 쪽(> −0.65)이 31% 다. M7f 는 리셋마다 변종 풀 1000 개에서 판을 새로 지어 학습한다(`--train-variant-pool 1000`, M6f 에서 만든 기능, 평가 창 v4~v11 은 풀에서 빠진다).

**Spec:** 이 문서. 직전 증거 `docs/reports/m7e-continue-obs.md`("다음").

## 본 실행 (OMEN)

시드 `S ∈ {0, 1, 2, 3, 4, 5}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-06-m7f-s$S --init runs/omen/2026-10-06-m7d-O10-s1/policy-501760.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variant-pool 1000 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0
```

- **대조군은 M7e**(같은 출발·보상, `--train-variants 4`, 시드 0~5)다. 이미 쟀다.
- 풀은 v0~v3 + v12~v1007 이다. 큰 선택 창 v1100~v1131 과 시험 창(v2000 부터)은 풀에 없다.
- 체크포인트 48 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 최고 그물은 큰 선택 창 90.38(완주 364, course_G 52)이다.

- **(가) 풀이 course_G 를 고치나.** 체크포인트 48 개의 course_G 완주 평균이 대조(M7e, 39.5/64)보다 8 판 넘게 많으면 "풀이 course_G 를 고친다" 로 본다.
  - 곁들여 완주 평균·충돌 평균을 대조와 비교한다.
- **(나) 고르기.** 후보는 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v18000~v18031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

# M6z — 차로를 지키는 그물에서 이어 돌려 고르기: course_G 충돌을 줄인 체크포인트 거두기 (코드 변경 없음)

**Goal:** M6y 의 고른 그물(A5 시드 0 · 100 만)은 완주한 판의 운전이 크게 좋아졌다. 완주 판 점수가 93.4 로, 지금 최고 그물의 88.1 보다 높다. 하지만 course_G 마지막 정지차에서 64 판 중 24 판을 부딪쳐 완주 포함 점수는 같았다. M6r 에서는 좋은 그물에서 시드 여럿으로 이어 돌리고 큰 창으로 골라 course_A 를 고쳤다. M6z 는 같은 방법을 M6y 그물에 쓴다.

**Spec:** 이 문서. 직전 증거 `docs/reports/m6y-lateral-profile.md`("다음").

## 본 실행 (OMEN)

시드 `S ∈ {0, 1, 2, 3, 4, 5}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-05-m6z-s$S --init runs/omen/2026-10-05-m6y-A5-s0/policy-1000960.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5
```

- 보상은 M6y 칸 A5 와 같다. 출발·앵커 참조만 M6y 고른 그물로 바꾼다.
- 체크포인트 48 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 출발 그물(M6y A5 시드 0 · 100 만)은 큰 선택 창 89.13(완주 367), 지금 최고 그물(M6w W10 시드 0 · 75 만)은 84.61 이다.

- **(가) course_G 를 고치나.** 고른 그물의 큰 선택 창 course_G 완주(64 판 중)를 출발 그물과 비교한다. 출발보다 8 판 이상 많으면 "고쳤다" 로 본다.
- **(나) 고르기.** 후보는 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v12000~v12031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.
- 곁들여: 큰 선택 창에서 출발 그물(89.13)보다 높은 체크포인트의 수, 차로 유지 중대가 다시 늘지 않는지.

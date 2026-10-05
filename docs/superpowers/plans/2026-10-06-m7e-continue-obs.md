# M7e — 새 최고 그물에서 같은 보상으로 이어 돌려 고르기 (코드 변경 없음)

**Goal:** M7d 의 새 최고 그물(O10 시드 1 · 50 만)은 새 시험 창 완주 98.2%, 완주 포함 점수 93.58 이다. 같은 실행의 37.5 만~75 만 체크포인트 넷이 모두 강했다(큰 선택 창 87.3~91.0). M6r·M6w·M7b 에서 좋은 그물을 이어 돌리고 큰 창으로 고르는 방법이 최고 그물을 갱신했다. M7e 는 같은 방법을 새 최고 그물에 쓴다.

**Spec:** 이 문서. 직전 증거 `docs/reports/m7d-obstacle-profile.md`.

**미룬 것:** 보호구역 속도(완주 판당 중대 0.7, 선생님 0)는 관측에 "앞으로 바뀔 제한속도" 가 없어 미리 줄이는 법을 배울 수 없다(관측은 지금 자리의 제한속도만 준다). 관측을 바꾸면 그물 입력이 바뀌어 이어 돌릴 수 없으므로 이번에는 다루지 않는다.

## 본 실행 (OMEN)

시드 `S ∈ {0, 1, 2, 3, 4, 5}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-06-m7e-s$S --init runs/omen/2026-10-06-m7d-O10-s1/policy-501760.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0
```

- 보상은 M7d 칸 O10 과 같다. 출발·앵커만 새 최고 그물로 바꾼다.
- 체크포인트 48 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 새 최고 그물은 큰 선택 창 90.38(완주 364)이다.

- **(가) 이어 돌리면 오르나.** 큰 선택 창에서 출발(90.38)보다 높은 체크포인트의 수를 센다(곁들임, 문턱 없음).
- **(나) 고르기.** 후보는 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v17000~v17031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.
  - 지금 최고는 이미 93.6 안팎이다. 남은 여지가 작아 3 점 문턱을 못 넘는 것이 자연스러운 결과일 수 있다.

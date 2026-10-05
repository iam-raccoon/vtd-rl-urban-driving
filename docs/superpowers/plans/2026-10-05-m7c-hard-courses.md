# M7c — 새 최고 그물의 장애물 충돌 줄이기: 학습 판에 장애물 판(③c)·어려운 코스(③h) 더하기

**Goal:** M7b 의 새 최고 그물(Fz 시드 1 · 50 만)은 완주한 판의 운전이 좋다(완주 판 점수 94.4). 하지만 장애물 충돌로 완주가 96% 다. 큰 선택 창의 실패 26/384 판을 액터별로 나누면(`m6_diag3.py`):

| 판 | 코스 · 액터 |
|---:|---|
| 12 | course_G 4 번째 정지차(횡 −0.3~−0.6) |
| 5 | course_D 3·4 번째(정지차·라바콘, 횡 +0.3~+0.7) |
| 3 | course_E 3 번째 라바콘(횡 −0.6~−0.7) |
| 2 | course_A 2 번째 라바콘(횡 +0.7) |
| 4 | course_B 첫 라바콘 앞에서 이탈 |

M7c 는 학습 판 구성을 바꾼다. 장애물 장면을 더 자주 보게 해서 비키는 능력을 키운다.

**Spec:** 이 문서. 직전 증거 `docs/reports/m7b-lat-free-80.md`("다음").

## 새 커리큘럼(이 마일스톤에서 더함)

- `curricula/stage3h.json`: ③ 전체(`stage3.json`)에서 실패가 나는 코스 A·D·E·G 의 항목을 **그대로**(경로·차로·액터·흔들기) 담는다. 학습에서 이 코스들의 ③ 판을 두 배로 보게 된다.
  - 판 이름 뒤에 커리큘럼 파일 이름이 붙어(`#stage3h`) ③ 전체의 같은 판과 세계 캐시가 섞이지 않는다.
  - 평가 스크립트의 기본 단계에 없다(테스트로 잠근다, `tests/test_stage3h.py`).
- `curricula/stage3c.json`(M6g 에서 만듦): ③ 전체 액터 자리에서 150 m 이상 떨어진 곳에 정지차 뒤 라바콘(좌·우)·오른쪽 라바콘을 둔 학습 전용 판이다.

## 본 실행 (OMEN)

칸 `ARM ∈ {C, S3C, S3H}` × 시드 `S ∈ {0, 1, 2}` = 9 실행, 동시 3 개씩 세 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-05-m7c-$ARM-s$S --init runs/omen/2026-10-05-m7b-Fz-s1/policy-501760.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json $EXTRA \
  --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
  --lat-profile 0.5 --lat-free-range 80
```

- `$EXTRA`: C 는 없음, S3C 는 `curricula/stage3c.json`, S3H 는 `curricula/stage3h.json`.
- 출발·앵커는 새 최고 그물이다. 보상은 M7b 와 같다.
- 학습 판 수(변종 4): C 120 개, S3C 144 개(③c 24 더함), S3H 136 개(③h 16 더함).
- 체크포인트 72 개 모두 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 새 최고 그물은 큰 선택 창 87.98(완주 358)이다.

- **(가) 판 구성이 충돌을 줄이나.** 칸마다 체크포인트 24 개의 충돌 평균을 낸다. S3C 또는 S3H 가 C 보다 20 판(384 판 중) 넘게 적으면 "그 판 구성이 충돌을 줄인다" 로 본다.
- **(나) 고르기.** 후보는 세 칸 체크포인트 72 개 전부다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물(M7b Fz 시드 1 · 50 만)을 **새 시험 창 v15000~v15031 × 시드 2**(384 판)에서 채점표 진단으로 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

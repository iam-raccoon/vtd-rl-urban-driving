# M7k — 장애물 앞 장면에서만 선생님 행동을 조금 섞기: DAgger 행을 골라 작은 계수의 보조 모방

**Goal:** 남은 손실은 장애물 충돌이다(지금 배포 묶음은 384 판 중 15~20 판을 놓친다, 대부분 course_G·course_D 의 정지차). 선생님은 크게 줄인 뒤 옆 차로로 옮겨 지나간다. 지금까지의 즉시 벌(감속 곡선·머무는 벌·겹친 폭 벌)은 칸 평균을 끌어올렸지만 상위 그물은 큰 창 90.3~90.9 에 모였다. M7k 는 학생이 모은 상태에 선생님 라벨을 붙이고(DAgger), **통로 안에 멈춘 장애물이 있는 행만** 골라, **작은 일정 계수**의 보조 모방으로 PPO 에 섞는다.

- M6q 의 보조 모방은 좁은 데이터(course_A 만)·큰 계수(1.0)로 정책을 무너뜨렸다(앵커 KL 2 대). 이번에는 장애물 장면 행만 쓰고 계수를 0.02·0.05 로 둔다.
- 계수는 이미 있는 인자로 일정하게 둔다: `--imitation-half-life 1 --imitation-floor C`(첫 걸음 뒤 바로 바닥값 C).

**Spec:** 이 문서. 직전 증거 `docs/reports/m7j-combine.md`("다음").

## 새 도구(이 마일스톤에서 더함)

- `scripts/filter_obstacle_rows.py`: DAgger 조각에서 장애물 앞 장면 행만 골라 새 폴더에 쓴다. 기준은 환경의 머무는 벌과 같다(통로 안, 앞범퍼에서 40 m 안, 멈춤 < 0.5 m/s, 높이 ≥ 0.25 m, 사람류 아님). 테스트 `tests/policy/test_filter_obstacle_rows.py`.

## 본 실행 (OMEN)

1. **수집:** 단계 ③ 전체(`curricula/stage3.json`, 여섯 코스), 학습 변종 v0~v3, 시드 8, β 0.5(학생과 선생님을 걸음마다 섞어 몬다), 학생 = 지금 배포 그물(M7f 시드 2 · 12.5 만). 판 192 개.
   ```
   python scripts/collect_episodes.py --curriculum curricula/stage3.json --variants 4 --seeds 8 --beta 0.5 --round 0 \
     --policy runs/omen/2026-10-06-m7f-s2/policy-125440.pt --out runs/omen/2026-10-06-m7k/data --workers 24
   python scripts/filter_obstacle_rows.py --src runs/omen/2026-10-06-m7k/data --dst runs/omen/2026-10-06-m7k/data_obs
   ```
2. **학습:** 칸 `ARM ∈ {I2: 0.02, I5: 0.05}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시 3 개씩 두 물결:
   ```
   python scripts/train_ppo.py --out runs/omen/2026-10-06-m7k-$ARM-s$S --init runs/omen/2026-10-06-m7f-s2/policy-125440.pt \
     --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
     --train-variants 4 --steps 1000000 --eval-every 125000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
     --violation-mode once_per_section --comfort-on-intent --seed $S \
     --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=5 --item-scale 1=3 --red-profile 1.0 \
     --lat-profile 0.5 --lat-free-range 80 --obs-profile 1.0 --lim-anticipate \
     --dagger-data runs/omen/2026-10-06-m7k/data_obs --imitation-half-life 1 --imitation-floor $C
   ```
   - 보상·관측은 M7h 실험 B 와 같고, 보조 모방만 더한다. **대조군은 M7h 실험 B**(같은 출발·보상·앞당김, 모방 없음, 시드 0~2)다.
   - DAgger 행은 앞당김을 끈 관측으로 기록된다. 장애물 앞 장면에서는 제한속도 칸만 다를 수 있어 그대로 쓴다.
3. 체크포인트 48 개 모두 앞당김을 켠 채 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 채점표 진단으로 잰다.

## 판정 기준 (미리 적는다)

완주 포함 점수는 M6u 와 같다(미완주 0 점, 384 판 평균). 지금 배포 묶음(M7f 시드 2 · 12.5 만 + 앞당김)은 큰 선택 창 90.90 이다.

- **(가) 장애물 장면 모방이 충돌을 줄이나.** 칸마다 체크포인트 24 개의 충돌 평균을 대조(M7h B)와 비교한다. 20 판 넘게 적은 칸이 있거나, course_G 완주 평균이 8 판 넘게 많은 칸이 있으면 "줄인다" 로 본다.
  - 곁들여 학습 로그의 앵커 KL·모방 손실을 본다(무너지는지).
- **(나) 고르기.** 후보는 두 칸 체크포인트 48 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3, 앞당김 켬) ①② 100%, ③a·③b ≥ 95% 다. 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물이 큰 선택 창에서 지금 배포 묶음(90.90)보다 높을 때만, **새 시험 창 v23000~v23031 × 시드 2**(384 판, 앞당김 켬)에서 나란히 잰다. 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.

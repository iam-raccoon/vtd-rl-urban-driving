# M6u — 적색 위반 벌을 키워 적색 신호를 지키게 하기: 보상에 항목별 배율 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M6t 에서 학생 그물은 모두 완주 판 하나당 적색 중대 위반이 3.6 건이었다(선생님 0.09 건). M6u 는 보상에서 적색 위반(항목 7)의 벌만 5 배·10 배로 키워 최고 그물에서 PPO 를 이어 돌린다. 적색을 지키는지, 그리고 완주를 포함한 점수가 오르는지 잰다.

**Architecture:**
- `RewardConfig` 에 `item_scale: tuple = ()` 를 더한다. `(항목, 배율)` 짝의 튜플이다.
  - 기본 `()` 이면 예전과 비트 단위로 같다.
  - 짝이 있으면 그 항목의 히트는 `minor·major` 에 배율을 곱해 따로 센다. 나머지 항목은 예전처럼 센다.
  - `ViolationTracker` 의 상태는 `(항목, 구간)` 열쇠라서 항목끼리 나눠 불러도 결과가 같다.
- `train_ppo.py --item-scale 항목=배율`(여러 번 줄 수 있다)을 열고 `_build_reward_cfg` 로 잇는다.

**Tech Stack:** Python 3.10, gymnasium, PyTorch, numpy 1.26.4, pytest. 본 실행은 OMEN(32 코어, RAM 30 GB).

**Spec:** 이 문서가 스펙이다. 직전 증거는 `docs/reports/m6t-score-sheet.md`("다음").

## 사전 측정(끝남)

적색 접근 추적: 단계 ② 원본 판, 시드 0. 정지선 80 m 안에서 신호가 적색인 동안 최저 속도를 쟀다.

- 지금 최고 그물(M6r 시드 4 · 50 만)은 적색 대부분을 10 m/s 넘게 지나간다. 몇 번은 2~3 m/s 까지 줄이지만 서지 않는다.
- 모방 최고(M4y)는 적색의 절반쯤에서 1.5~3 m/s 까지 줄이지만 역시 서지 않는다.
- 채점기의 "섰다" 는 좁다(`score_fma`).
  - 앞범퍼가 정지선 앞 [0, 2 m) 안에 있어야 한다.
  - 속도가 1 km/h(0.28 m/s) 이하여야 하고, 0.5 초를 버텨야 한다.
  - 2 m 보다 앞에서 서면 경미, 안 서면 중대다.
- 신호 주기는 녹색 30 초, 황색 3 초, 적색 30 초이고 신호마다 위상이 무작위다. 적색에 닿으면 평균 15 초를 기다린다.

**M6t 추정 바로잡기.** M6t 는 "할인율 0.99 에서는 서는 쪽이 지나가는 쪽보다 훨씬 손해" 라고 적었다. 다시 계산하면 그렇게까지 크지 않다.

- 진행 항은 경로 전체에 100 점이다. 경로 약 3,000 m, 12 m/s 이면 한 걸음에 약 0.04 점이다.
- 그 뒤 진행의 할인 합은 약 0.04 / 0.01 = 4 점이다.
- 15 초(150 걸음)를 기다리면 그중 1 − 0.99^150 ≈ 78% 가 늦어진다(약 3.1 점). 시간 비용 약 0.8 점을 더해 약 4 점을 잃는다.
- 그래서 지금 보상(중대 −6)으로도 **정지선 앞에 정확히 서는** 쪽이 약 2 점 낫다.
  - 2 m 보다 앞에서 서면 경미 −3 에 기다림 약 −4 라 지나가는 쪽(−6)보다 못하다.
- 정리하면, 이득이 정확한 정지에만 있고 그 폭이 2 점으로 작다. 5 배(중대 −30, 경미 −15)면 어디서든 서는 쪽이 낫고, 그 폭도 10~25 점으로 커진다.

## 본 실행 (OMEN)

칸 `ARM ∈ {K5: 7=5, K10: 7=10}` × 시드 `S ∈ {0, 1, 2}` = 6 실행, 동시 3 개씩 두 물결:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python scripts/train_ppo.py \
  --out runs/omen/2026-10-05-m6u-$ARM-s$S --init runs/omen/2026-10-05-m6r-s4/policy-501760.pt \
  --curricula curricula/stage1.json curricula/stage2.json curricula/stage3a.json curricula/stage3b.json curricula/stage3.json \
  --train-variants 4 --steps 1000000 --eval-every 250000 --eval-seeds 1 --final-eval-seeds 1 --envs 10 \
  --violation-mode once_per_section --comfort-on-intent --seed $S \
  --anchor-coef 0.03 --lr 1e-4 --warmup-updates 20 --item-scale 7=$K
```

- M6s 와 **`--item-scale` 만** 다르다. 출발·앵커 참조, 판, 계수, 걸음, 학습률이 같다.
- **대조군은 M6s 시드 0~2** 다. 같은 출발에서 배율 없이 같은 설정으로 돌린 실행이다. 그 체크포인트 12 개를 채점표 진단으로 다시 잰다(학습은 새로 안 한다).
- 체크포인트(25 만 간격 넷)마다 큰 선택 창(v1100~v1131 × 시드 2, 384 판)을 **채점표 진단**(`m6_diag4.py`)으로 잰다. 판마다 종료 사유, 점수(구간 평균), 채점표를 적는다.

## 판정 기준 (미리 적는다)

**완주 포함 점수** = 384 판의 평균. 완주한 판은 그 판의 점수(구간 평균), 완주 못 한 판은 0 점이다. 지금 최고 그물은 큰 선택 창에서 82.36 이다(M6t 자료). 선생님은 94.42 다.

- **(가) 적색을 지키나.** 고른 그물의 큰 선택 창 적색 중대 위반이 완주 판 하나당 1.0 건 이하이면 "적색을 지킨다" 로 본다. 지금 최고는 3.63 건이다.
  - 곁들여 칸마다(K5·K10·대조) 체크포인트 12 개의 적색 중대 평균을 낸다.
- **(나) 고르기.** 후보는 K5·K10 체크포인트 24 개다. 큰 선택 창 완주 포함 점수 1 위를 고른다. 동점이면 완주가 많은 쪽이다.
  - 지킴선은 보고 창(v4~v7 × 시드 3) ①② 100%, ③a·③b ≥ 95% 다(`eval_unseen.py`). 못 지키면 다음 순위다.
- **(다) 교체.** 고른 그물과 지금 최고 그물을 **새 시험 창 v7000~v7031 × 시드 2**(384 판, 아직 아무도 안 씀)에서 채점표 진단으로 나란히 잰다.
  - 완주 포함 점수가 3 점 이상 높을 때만 바꾼다.
- 곁들여: 큰 선택 창에서 완주 포함 점수가 82.36 보다 높은 체크포인트의 수(칸마다).

## 이 실험이 답하지 않는 것

- 할인율(γ)을 바꾸는 쪽은 안 잰다. 위 계산으로는 할인보다 "정확히 서기" 의 이득이 작은 것이 문제다.
- 정지선 앞 속도에 매 걸음 벌을 주는 식의 촘촘한 보상은 안 쓴다. 대회 채점과 같은 항만 키운다.

---

### Task 1: 보상의 항목별 벌 배율(`item_scale`)과 `--item-scale`

**Files:**
- Modify: `vtd_rl/env/reward.py`(`RewardConfig.item_scale`, `RewardConfig.__post_init__`, `RewardShaper.step`)
- Modify: `scripts/train_ppo.py`(`--item-scale`, `_parse_item_scale`, `_build_reward_cfg`)
- Test: `tests/env/test_reward.py`, `tests/rl/test_train_ppo.py`

**Interfaces:**
- Produces:
  - `RewardConfig.item_scale: tuple = ()` — `((항목:int, 배율:float), ...)`. `__post_init__` 이 튜플로 바꾸고 검사한다.
  - CLI `--item-scale 항목=배율`(여러 번), `hparams["item_scale"]`(안 주면 `None`).

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/env/test_reward.py` 끝에:

```python
def test_item_scale_기본값은_비어_있고_예전과_같다():
    assert RewardConfig().item_scale == ()
    assert RewardConfig(item_scale=()) == RewardConfig()


def test_item_scale_은_그_항목의_벌만_키운다():
    b = h_board()
    cfg = RewardConfig(item_scale=((7, 5.0),), rule_scale=2.0)
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    out = sh.step([Hit(1.0, 0, 7, "major"), Hit(1.0, 0, 3, "major"), Hit(1.0, 0, 1, "minor")],
                  0.0, zero, zero, "running")
    assert out.terms["violation"] == pytest.approx((cfg.major * 5.0 + cfg.major + cfg.minor) * 2.0)
    assert out.counted == 3


def test_item_scale_은_구간당_한_번_심화_차액에도_곱한다():
    b = h_board()
    cfg = RewardConfig(item_scale=((7, 5.0),), violation_mode="once_per_section")
    sh = RewardShaper(b, cfg)
    sh.reset()
    zero = {"control": [0.0, 0.0], "turn": 0}
    p1 = sh.step([Hit(0.0, 0, 7, "minor")], 0.0, zero, zero, "running").terms["violation"]
    p2 = sh.step([Hit(1.0, 0, 7, "major")], 0.0, zero, zero, "running").terms["violation"]
    p3 = sh.step([Hit(2.0, 0, 7, "major")], 0.0, zero, zero, "running").terms["violation"]
    assert p1 == pytest.approx(cfg.minor * 5.0)
    assert p2 == pytest.approx((cfg.major - cfg.minor) * 5.0)
    assert p3 == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["repeat", "once_per_section"])
def test_배율_1_은_배율이_없는_것과_같다_속성(mode):
    """항목끼리 나눠 세도 `(항목, 구간)` 열쇠라 결과가 같아야 한다 — 무작위 히트 흐름 300 개."""
    b = h_board()
    rng = random.Random(7)
    zero = {"control": [0.0, 0.0], "turn": 0}
    for _ in range(300):
        stream = [[Hit(t * 0.05, rng.randrange(3), rng.choice((1, 3, 7, 13, 14)),
                       rng.choice(("minor", "major"))) for _k in range(rng.randrange(3))]
                  for t in range(40)]
        plain = RewardShaper(b, RewardConfig(violation_mode=mode))
        ones = RewardShaper(b, RewardConfig(violation_mode=mode, item_scale=((7, 1.0), (3, 1.0))))
        plain.reset()
        ones.reset()
        for hits in stream:
            a = plain.step(hits, 0.0, zero, zero, "running")
            c = ones.step(hits, 0.0, zero, zero, "running")
            assert c.terms["violation"] == pytest.approx(a.terms["violation"])
            assert c.counted == a.counted and c.collision == a.collision


def test_item_scale_목록을_줘도_튜플로_바뀌고_해시된다():
    cfg = RewardConfig(item_scale=[[7, 5]])
    assert cfg.item_scale == ((7, 5.0),)
    assert isinstance(cfg.item_scale[0][1], float)
    hash(cfg)


@pytest.mark.parametrize("bad", [((11, 5.0),), ((14, 2.0),), ((0, 2.0),), ((16, 2.0),),
                                 ((7, -1.0),), ((7, 2.0), (7, 3.0)), ((7,),)])
def test_잘못된_item_scale_은_거부한다(bad):
    with pytest.raises(ValueError):
        RewardConfig(item_scale=bad)
```

`tests/rl/test_train_ppo.py` 끝(마지막 `@pytest.mark.slow` 테스트들 앞, `test_violation_mode_인자가_RewardConfig에_반영된다` 바로 뒤)에:

```python
def test_item_scale_인자가_RewardConfig에_반영된다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args([
        "--out", "/tmp/불필요-존재안함", "--item-scale", "7=5", "--item-scale", "9=2.5"])
    cfg = module._build_reward_cfg(a)
    assert cfg.item_scale == ((7, 5.0), (9, 2.5))

    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert off.item_scale == ()


@pytest.mark.parametrize("bad", ["7", "x=1", "7=y", "7=5=1"])
def test_item_scale_인자_형식이_틀리면_거부한다(bad):
    module = _load_train_ppo_module()
    with pytest.raises(SystemExit):
        module._build_parser().parse_args(["--out", "x", "--item-scale", bad])


def test_item_scale_인자의_항목이_틀리면_거부한다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "x", "--item-scale", "11=5"])
    with pytest.raises(ValueError):
        module._build_reward_cfg(a)
```

- [ ] **Step 2: 실패를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m6u_t1.txt 2>&1; echo rc=$?`
Expected: rc≠0 — `item_scale` 이 없어 `TypeError`, `--item-scale` 이 없어 `SystemExit`.

- [ ] **Step 3: `RewardConfig` 를 고친다.**

`vtd_rl/env/reward.py` 의 `RewardConfig` 에서 `violation_mode` 필드(와 그 주석) 바로 뒤에 더한다:

```python
    # M6u — 항목별 벌 배율 `((항목, 배율), ...)`. 그 항목의 히트는 minor·major 에 배율을 곱해 센다.
    # 기본 `()` 이면 예전과 같다. 충돌 항목(⑪⑭)은 `collision` 항이 따로 물리므로 받지 않는다.
    item_scale: tuple = ()

    def __post_init__(self):
        pairs = []
        for pair in self.item_scale:
            if len(pair) != 2:
                raise ValueError(f"item_scale 은 (항목, 배율) 짝이어야 한다: {pair!r}")
            item, scale = int(pair[0]), float(pair[1])
            if not 1 <= item <= 15 or item in COLLISION_ITEMS:
                raise ValueError(f"item_scale 항목은 1~15 이고 충돌 항목 {COLLISION_ITEMS} 가 아니어야 한다: {item}")
            if scale < 0.0:
                raise ValueError(f"item_scale 배율은 0 이상이어야 한다: {scale}")
            pairs.append((item, scale))
        if len({item for item, _ in pairs}) != len(pairs):
            raise ValueError(f"item_scale 에 같은 항목이 두 번 있다: {pairs}")
        object.__setattr__(self, "item_scale", tuple(pairs))   # frozen — 목록으로 줘도 튜플로 둔다(해시)
```

- [ ] **Step 4: `RewardShaper.step` 을 고친다.**

```python
        counted, penalty = self.tracker.charge(rule_hits, cfg.minor, cfg.major)
```

를 다음으로 바꾼다:

```python
        # M6u — 배율을 건 항목은 따로 센다. 추적기 상태가 (항목, 구간) 열쇠라 나눠 불러도 결과가 같다.
        scaled = {item for item, _ in cfg.item_scale}
        counted, penalty = self.tracker.charge(
            [h for h in rule_hits if h.item not in scaled], cfg.minor, cfg.major)
        for item, scale in cfg.item_scale:
            c, p = self.tracker.charge([h for h in rule_hits if h.item == item],
                                       cfg.minor * scale, cfg.major * scale)
            counted = counted + c
            penalty += p
```

- [ ] **Step 5: `train_ppo.py` 에 `--item-scale` 을 연다.**

`_build_parser` 위(모듈 수준)에:

```python
def _parse_item_scale(text: str):
    """`--item-scale 7=5` → (7, 5.0). 형식이 틀리면 argparse 가 알아듣는 오류를 낸다."""
    parts = text.split("=")
    try:
        if len(parts) != 2:
            raise ValueError
        return int(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"항목=배율 꼴이어야 한다(예: 7=5): {text!r}") from None
```

`--violation-mode` 인자 바로 뒤에:

```python
    ap.add_argument("--item-scale", type=_parse_item_scale, action="append", default=None,
                    metavar="ITEM=SCALE",
                    help="항목별 벌 배율(여러 번 줄 수 있다). 예: --item-scale 7=5 는 적색 위반의"
                         " 경미·중대 벌을 5 배로 키운다(M6u). 안 주면 예전과 같다")
```

`_build_reward_cfg` 의 독스트링 첫 줄 "네 필드" 를 "다섯 필드" 로, 본문 나열에 `a.item_scale`(안 주면 `None` → `()`)을 더하고, 반환을 다음으로 바꾼다:

```python
    return dataclasses.replace(RewardConfig(), comfort_steer=a.comfort_steer,
                               comfort_accel=a.comfort_accel, comfort_on_intent=a.comfort_on_intent,
                               violation_mode=a.violation_mode,
                               item_scale=tuple(a.item_scale or ()))
```

- [ ] **Step 6: 통과를 확인한다.**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/env/test_reward.py tests/rl/test_train_ppo.py -q -m "not slow" > $TMP/m6u_t1.txt 2>&1; echo rc=$?`
Expected: rc=0.

그다음 전체: `env -u PYTHONPATH .venv/bin/python -m pytest tests -q -m "not slow" > $TMP/m6u_all.txt 2>&1; echo rc=$?` → rc=0(실패 목록은 `grep -E "^(FAILED|ERROR)" $TMP/m6u_all.txt`).

- [ ] **Step 7: 커밋한다.**

```bash
git add vtd_rl/env/reward.py scripts/train_ppo.py tests/env/test_reward.py tests/rl/test_train_ppo.py
git commit -m "M6u — 보상에 항목별 벌 배율(item_scale)과 train_ppo --item-scale"
```

## Global Constraints

- `third_party/rule_stack` 은 절대 고치지 않는다.
- `item_scale=()`(기본)이면 보상이 예전과 비트 단위로 같아야 한다.
- `runs/`, `*.pt`, `*.npz`, CSV 는 커밋하지 않는다.
- 커밋 메시지·주석·독스트링은 한국어로 쓴다. 커밋에 Co-Authored-By·Claude·생성 도구 표기를 넣지 않는다.
- 테스트 판정은 종료 코드(rc)로 한다. "N passed" 를 grep 하지 않는다.
- numpy 는 1.26.4 그대로다.

## Review Focus

- 배율을 건 항목과 안 건 항목이 같은 걸음에 섞여 와도 각각 한 번씩만 센다(속성 테스트).
- `once_per_section` 에서 경미→중대 심화 차액에도 배율이 곱해진다.
- 충돌 항목(⑪⑭)에 배율을 주면 거부한다 — 충돌은 `collision` 항이 따로 물리고 위반 합계에서 빠진다.
- 목록으로 준 `item_scale`(JSON 에서 읽은 꼴)도 튜플로 바뀌어 `RewardConfig` 가 해시된다(벡터 환경 피클).
- `--item-scale` 을 안 주면 `_build_reward_cfg(a) == RewardConfig()` 가 그대로다.

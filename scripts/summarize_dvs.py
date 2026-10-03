#!/usr/bin/env python3
"""`eval_det_vs_stoch.py` 가 낸 JSONL 을 설정별 표로 접는다.

성적표 숫자를 손으로 옮겨 적지 않으려고 만들었다. 체크포인트 경로에서 설정 이름과 시드를
뽑아(`<...>/<name>-s<seed>/ac-*.pt`) 설정마다 평균을 내고, `--pair` 를 주면 **시드로 짝지어**
기준선과의 차이와 부호검정 p 값까지 낸다.

★ 짝짓기가 왜 필요한가: 시드 간 분산이 개입 효과보다 크다. M4e 는 시드 0·1·2 로 짝지어
모방 앵커 바닥이 결정적 완주율을 44.4% → 66.7% 로 올린다고 봤는데(n=3, p=0.25), 시드를
3·4·5 로 늘리자 재현되지 않았다. **짝 안 지은 평균 비교는 시드 운을 개입 효과로 읽는다.**

    python scripts/summarize_dvs.py --jsonl runs/omen/2026-09-29-m4f/dvs-3m.jsonl
    python scripts/summarize_dvs.py --jsonl <경로> --pair base
"""
import argparse
import json
import os
import re
from collections import defaultdict
from math import comb

# 결정적 완주율이 이 마일스톤의 1 순위 지표다(배포되는 것은 평균 행동이므로).
METRICS = ("det_goal", "sto_goal", "det_score_completed", "sto_score_completed")


def parse_run(path: str):
    """`.../<name>-s<seed>/ac-3002880.pt` → `("name", 3)`. 못 읽으면 `(폴더명, None)`."""
    run_dir = os.path.basename(os.path.dirname(path))
    m = re.match(r"^(.*)-s(\d+)$", run_dir)
    if not m:
        return run_dir, None
    return m.group(1), int(m.group(2))


def load(jsonl: str, alias: dict | None = None):
    """`{(설정, 시드): {단계: {지표: 값}}}` 로 읽는다.

    `alias` 는 `{원래이름: 묶을이름}`. 같은 설정을 **다른 폴더 이름으로 나눠 돌렸을 때**
    한 팔로 묶는 데 쓴다 — M4f 에서 기준선 시드 0~2 는 M4e 의 `m4d-align`, 시드 3~5 는
    새로 돌린 `base` 라 이름이 다른데 **설정은 같다**(둘 다 개입 없음).

    같은 (묶을이름, 시드) 가 둘 이상이면 **터뜨린다.** 조용히 덮어쓰면 어느 실행의 숫자가
    성적표에 실렸는지 알 수 없게 된다.
    """
    alias = alias or {}
    out = {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            name, seed = parse_run(row["checkpoint"])
            key = (alias.get(name, name), seed)
            if key in out:
                raise SystemExit(f"같은 (설정, 시드) 가 두 번 나왔다: {key} — "
                                 f"별칭이 서로 다른 실행을 겹쳐 덮고 있다")
            out[key] = {st: v for st, v in row.items() if _is_stage(v)}
    return out


def _is_stage(v) -> bool:
    """JSONL 한 줄의 값이 **단계 결과**인가 — 지표 키(`METRICS`)를 하나라도 가진 dict.

    예전에는 단계 이름을 `("stage1", "stage2")` 로 박아 둬서, ③ 단계 커리큘럼으로 잰 줄의
    `stage3a`·`stage3b` 가 **조용히 버려졌다**(표에 안 나오고 경고도 없음). 이름이 아니라
    모양으로 고른다 — `checkpoint`(문자열) 같은 다른 키는 dict 가 아니라 걸리지 않는다.
    """
    return isinstance(v, dict) and any(m in v for m in METRICS)


def stages_of(data: dict) -> list:
    """`load()` 결과에 나온 단계 이름을 **처음 나온 순서대로** 모은다(설정마다 달라도 합집합)."""
    seen = []
    for per_stage in data.values():
        for st in per_stage:
            if st not in seen:
                seen.append(st)
    return seen


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(v, pct: bool):
    if v is None:
        return "—"
    return f"{100 * v:.1f}%" if pct else f"{v:.2f}"


def sign_test_p(diffs, eps: float = 1e-9) -> tuple:
    """부호검정(양측). 0 인 차이는 버린다 — 표준 관례이고 n 을 줄여 보수적이다.

    돌아오는 것: `(개선 수, 악화 수, p)`. 묶인 것만 있으면 p 는 `None`.
    """
    pos = sum(1 for d in diffs if d > eps)
    neg = sum(1 for d in diffs if d < -eps)
    n = pos + neg
    if n == 0:
        return pos, neg, None
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    return pos, neg, min(1.0, 2.0 * tail / (2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--pair", default=None,
                    help="이 설정을 기준선으로 삼아 **시드로 짝지어** 차이와 부호검정 p 를 낸다")
    ap.add_argument("--metric", default="det_goal", choices=METRICS,
                    help="--pair 가 쓰는 지표(기본 det_goal — 배포되는 것은 평균 행동이다)")
    ap.add_argument("--alias", action="append", default=[], metavar="원래이름=묶을이름",
                    help="같은 설정을 다른 폴더 이름으로 나눠 돌렸을 때 한 팔로 묶는다(반복 가능)")
    a = ap.parse_args()

    alias = {}
    for item in a.alias:
        if "=" not in item:
            raise SystemExit(f"--alias 는 '원래이름=묶을이름' 꼴이어야 한다: {item!r}")
        k, v = item.split("=", 1)
        alias[k] = v
    data = load(a.jsonl, alias)
    names = sorted({n for n, _ in data})
    stages = stages_of(data)

    print(f"# {os.path.basename(a.jsonl)} — 설정별 평균\n")
    for st in stages:
        print(f"## {st}\n")
        print("| 설정 | 시드 | " + " | ".join(METRICS) + " |")
        print("|---|---|" + "---|" * len(METRICS))
        for name in names:
            seeds = sorted(s for n, s in data if n == name)
            for s in seeds:
                r = data[(name, s)].get(st, {})
                cells = [_fmt(r.get(m), m.endswith("goal")) for m in METRICS]
                print(f"| {name} | {s} | " + " | ".join(cells) + " |")
            cells = [_fmt(_mean([data[(name, s)].get(st, {}).get(m) for s in seeds]),
                          m.endswith("goal")) for m in METRICS]
            print(f"| **{name} 평균** | n={len(seeds)} | " + " | ".join(cells) + " |")
        print()

    if not a.pair:
        return
    base = a.pair
    if base not in names:
        raise SystemExit(f"기준선 설정 '{base}' 가 JSONL 에 없다 — 있는 것: {names}")

    print(f"# 시드로 짝지은 비교 — 기준선 `{base}`, 지표 `{a.metric}`\n")
    for st in stages:
        print(f"## {st}\n")
        print(f"| 설정 | 짝지은 시드 | 기준선 | 개입 | 차이 | 개선/악화 | 부호검정 p |")
        print("|---|---|---|---|---|---|---|")
        for name in names:
            if name == base:
                continue
            shared = sorted({s for n, s in data if n == name}
                            & {s for n, s in data if n == base})
            if not shared:
                print(f"| {name} | **없음** | — | — | — | — | **짝지을 시드가 없다** |")
                continue
            b = [data[(base, s)].get(st, {}).get(a.metric) for s in shared]
            t = [data[(name, s)].get(st, {}).get(a.metric) for s in shared]
            diffs = [x - y for x, y in zip(t, b) if x is not None and y is not None]
            pos, neg, p = sign_test_p(diffs)
            pct = a.metric.endswith("goal")
            print(f"| {name} | {','.join(map(str, shared))} (n={len(diffs)}) | "
                  f"{_fmt(_mean(b), pct)} | {_fmt(_mean(t), pct)} | "
                  f"{_mean(diffs) * (100 if pct else 1):+.1f}{'pp' if pct else ''} | "
                  f"{pos}/{neg} | {'—' if p is None else f'{p:.3f}'} |")
        print()


if __name__ == "__main__":
    main()

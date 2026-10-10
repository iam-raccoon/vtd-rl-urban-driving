"""M4o 계측 규칙 — **고른 판으로 성적을 내지 않는다.**

`run_dagger.py --select best` 는 라운드 5 개 중 최선을 고른다. 그러면 그 평가는
**잡음의 상단**을 고른 값이고, 단계 ① 은 독립 판이 **6 개**(해상도 16.7 점)라 편향이
작지 않다. 그래서 채택된 `policy-best.pt` 를 **수집이 쓰지 않은 변종**으로 다시 재고,
성적표 헤드라인은 이 숫자를 쓴다.

    env -u PYTHONPATH .venv/bin/python scripts/eval_unseen.py \\
        --ckpt runs/omen/<실행>/policy-best.pt \\
        --out  runs/omen/<실행>/unseen.md

기본값이 M4o 수집 설정(`--variants 4`)과 짝이다 — 수집이 `v0~v3` 를 썼으므로 평가는
`v4~v7` 이다. 겹치는 창을 달라고 하면 **돌기 전에 거부**한다(`--include-seen` 으로만
일부러 겹칠 수 있다. 선택용 숫자와 보고용 숫자의 차이를 재려면 그쪽을 쓴다).

**출발점(0/18)과 비교하려면 같은 창으로 M3 학생을 한 번 더 재면 된다** — 비교 대상이
같은 판이라야 뜻이 있다:

    env -u PYTHONPATH .venv/bin/python scripts/eval_unseen.py \\
        --ckpt runs/omen/2026-09-26-m3-squash-warm/policy.pt \\
        --out  runs/omen/<실행>/unseen-baseline.md

**이 스크립트는 판정하지 않는다.** 합격·불합격을 안 찍고 숫자만 낸다 — 판정 기준은
계획서에 미리 적혀 있고 결론은 성적표가 낸다.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.env.drive_env import EnvConfig  # noqa: E402
from vtd_rl.world.dynamics import DynamicsParams  # noqa: E402
from vtd_rl.world.world import WorldConfig  # noqa: E402
from vtd_rl.env.observation import ObsConfig  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.world.board import curriculum_has_actors, load_window  # noqa: E402
from vtd_rl.world.board import window_label as _window_label  # noqa: E402

#: 기본 평가 단계 — ① 유지, ② 유지, ③ 학습. 순서가 성적표 표의 순서다.
DEFAULT_STAGES = ["stage1", "stage2", "stage3"]

#: M4o 수집이 쓴 변종 수(`run_dagger.py --variants 4`). 평가 창의 **기본 시작점**이
#: 이 값이다 — 즉 기본이 "수집이 안 본 첫 변종부터" 다. 0 으로 두면 안 된다.
DEFAULT_SEEN_VARIANTS = 4

#: 성적표에 쓸 사람 말 이름(`scripts/run_dagger.py` 와 같은 표기).
STAGE_LABELS = {"stage1": "단계 ①", "stage2": "단계 ②", "stage3": "단계 ③",
                "stage4": "단계 ④", "stage5": "단계 ⑤"}


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


def stage_path(name: str) -> str:
    """단계 이름 → 커리큘럼 경로. 모르는 이름은 조용히 넘기지 않는다.

    (`run_dagger.py`·`probe_anchor.py` 에 같은 규칙의 쌍둥이가 있다 — 스크립트끼리는
    서로를 임포트하지 않는 것이 이 레포의 관행이라 작게 다시 쓴다.)
    """
    if name.endswith(".json") or os.sep in name:
        path = name if os.path.isabs(name) else os.path.join(REPO, name)
    else:
        path = os.path.join(REPO, "curricula", f"{name}.json")
    if not os.path.exists(path):
        here = os.path.join(REPO, "curricula")
        avail = sorted(os.path.splitext(f)[0] for f in os.listdir(here) if f.endswith(".json"))
        raise ValueError(f"모르는 단계 '{name}' — 커리큘럼이 없다({path}). "
                         f"있는 단계: {', '.join(avail)}")
    return path


def has_actors(path: str) -> bool:
    """그 커리큘럼에 액터가 한 개라도 있는가(판을 짓지 않고 JSON 만 본다)."""
    return curriculum_has_actors(path)


def stage_boards(name: str, variants: int, offset: int):
    """`(판 목록, 변종을 걸었는가)` — **액터가 없는 단계는 변종을 접는다.**

    변종은 액터만 흔든다(`world/place.py` 의 `jitter_specs` 는 `specs` 가 비면 그대로
    돌려준다). 그래서 단계 ①② 처럼 액터가 0 개인 단계는 변종 N 벌이 **글자 그대로 같은
    판 N 벌**이다 — 일만 N 배로 늘고(둘 다 코스를 끝까지 모는 평가다) 숫자는 한 자리도
    안 바뀐다. 접고, 접었다는 사실을 표와 요약에 남긴다.

    단계 ③ 은 액터가 있으므로 접지 않는다 — 이 도구의 존재 이유가 거기다.

    접는 규칙은 `world/board.py::load_window` 하나에 있다 — `run_dagger.py --select-variants`
    (선택 창)도 같은 함수를 쓴다.
    """
    return load_window(stage_path(name), variants, offset)


def window_label(variants: int, offset: int) -> str:
    """`v4~v7` 꼴 — 성적표가 **어느 변종으로 쟀는지** 한눈에 보이게(`world/board.py` 와 같은 것)."""
    return _window_label(variants, offset)


def evaluate_stage(net, boards, eval_seeds: int, config=None) -> dict:
    """단계 하나. 단계마다 **따로** 부른다 — 판 이름이 단계끼리 같아서(신호·액터만 다르다)

    한 번에 넘기면 `VtdDriveEnv` 의 세계 캐시가 "다른 판을 준다" 로 죽는다
    (`probe_anchor.py`·`refit_m3.py` 에 같은 주석).
    """
    ev = evaluate_policy(net, boards, seeds=tuple(range(eval_seeds)), config=config)
    return {"goal_rate": ev["goal_rate"], "mean_score": ev["mean_score"],
            "mean_score_completed": ev["mean_score_completed"],
            "mean_reward": ev["mean_reward"],
            # ★ **합**이다(평균이 아니다). M4m·M4n 성적표가 "얼마나 멀리 갔나" 를 이 자로
            # 쟀고, 완주율 0% 가 이어질 때 학습을 가리지 않는 유일한 숫자다.
            "total_steps": sum(int(e.steps) for e in ev["episodes"]),
            "episodes": len(ev["episodes"])}


def report_lines(a, rows, window, dev) -> list:
    out = [f"# 안 쓴 변종 재평가 — {datetime.date.today().isoformat()} · `{platform.node()}`"
           f" · 장치 `{dev}` · 규칙 스택 `{rs.commit()[:7]}`", "",
           f"- 체크포인트 `{a.ckpt}`",
           f"- 변종 창 **{window}** — 수집이 쓴 변종은 `v0~v{a.seen_variants - 1}`"
           f"({a.seen_variants}벌)이고 여기는 그 **안 쓴 변종**이다."
           + ("  ⚠ `--include-seen` 으로 **일부러 겹치게** 쟀다 — 이 숫자는 선택 편향이"
              " 든 쪽이다." if a.include_seen else ""),
           f"- 평가 시드 {a.eval_seeds}개 · 결정적 행동",
           *(["- 관측: 앞당긴 제한속도"] if getattr(a, "lim_anticipate", False) else []),
           *([f"- 동역학: 언더스티어 K={a.understeer} · 측가속 한계 {a.lat_accel_max} m/s²"]
             if getattr(a, "understeer", 0.0) or getattr(a, "lat_accel_max", 0.0) else []), "",
           "| 단계 | 변종 | 판 수 | 판별 평가 | 완주율 | 점수 | 총걸음 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for name, row in rows:
        mark = window if row["varied"] else "원본(액터 0 — 변종이 뜻 없음)"
        out.append(f"| {stage_label(name)} | {mark} | {row['boards']} | {row['episodes']} |"
                   f" {row['goal_rate'] * 100:.1f}% | {row['mean_score']:.1f} |"
                   f" {row['total_steps']} |")
    out += ["", "이 표는 **판정하지 않는다** — 판정 기준은 계획서에 미리 적혀 있다.", "",
            "액터가 없는 단계(①②)는 변종이 액터만 흔들기 때문에 N 벌이 같은 판 N 벌이 된다 —",
            "원본 한 벌로 접었다. 단계 ③ 만 변종 창이 실제로 다른 판을 준다."]
    return out


def dry_run_lines(a, window) -> list:
    out = [f"체크포인트: {a.ckpt}" + ("" if os.path.exists(a.ckpt) else "  ← 없다!"),
           f"변종 창: {window} (수집이 쓴 것 v0~v{a.seen_variants - 1})"
           + ("  ← --include-seen: 수집 창과 겹친다" if a.include_seen else ""),
           f"평가 시드: {a.eval_seeds}개"]
    for name in a.stage:
        boards, varied = stage_boards(name, a.variants, a.variant_offset)
        tag = window if varied else "원본(액터 0 — 접음)"
        out.append(f"  {name}: 판 {len(boards)}개 [{tag}] → {len(boards) * a.eval_seeds} 판"
                   f" · 예: {', '.join(b.name for b in boards[:3])}")
    out.append(f"성적표: {a.out or '(표준출력)'}")
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="채택된 체크포인트를 안 쓴 변종으로 다시 잰다")
    ap.add_argument("--ckpt", required=True,
                    help="평가할 `DrivePolicy` 체크포인트(보통 `<실행>/policy-best.pt`)")
    ap.add_argument("--stage", action="append", default=[],
                    help=f"평가 단계(반복 가능, 기본 {' '.join(DEFAULT_STAGES)})")
    ap.add_argument("--seen-variants", type=int, default=DEFAULT_SEEN_VARIANTS,
                    help=f"**수집이 쓴** 변종 수(기본 {DEFAULT_SEEN_VARIANTS} ="
                         " run_dagger --variants 4). 평가 창의 기본 시작점이자, 겹침을"
                         " 거부하는 기준이다")
    ap.add_argument("--variants", type=int, default=None,
                    help="평가할 변종 수(기본 --seen-variants 와 같은 수)")
    ap.add_argument("--variant-offset", type=int, default=None,
                    help="평가 변종의 시작 번호(기본 --seen-variants — 즉 **수집이 안 본 첫"
                         " 변종**). 0 으로 두면 수집이 본 판으로 평가하게 되어 거부한다")
    ap.add_argument("--include-seen", action="store_true",
                    help="수집이 본 변종과 겹치는 창을 **일부러** 허용한다 — 선택용 숫자와"
                         " 보고용 숫자의 차이를 잴 때만 쓴다(성적표에 경고가 붙는다)")
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="성적표 경로(안 주면 표준출력)")
    ap.add_argument("--dry-run", action="store_true",
                    help="아무것도 안 돌리고 무엇을 어느 변종으로 잴지만 찍는다")
    ap.add_argument("--lim-anticipate", action="store_true",
                    help="평가 환경 관측의 제한속도를 앞당긴 값으로(M7h) — 그렇게 학습한 그물을 잴 때 켠다")
    ap.add_argument("--understeer", type=float, default=0.0, help="세계 동역학 언더스티어 K[s²/m](train_ppo 와 같은 뜻)")
    ap.add_argument("--lat-accel-max", type=float, default=0.0, help="세계 동역학 측가속 한계[m/s²](train_ppo 와 같은 뜻)")
    return ap


def main():
    ap = _build_parser()
    a = ap.parse_args()

    a.stage = list(a.stage) or list(DEFAULT_STAGES)
    if a.seen_variants < 0:
        ap.error("--seen-variants 는 0 이상이어야 한다")
    if a.variants is None:
        a.variants = max(1, a.seen_variants)
    if a.variant_offset is None:
        # ★ 기본이 **0 이 아니다.** 0 이면 수집이 본 바로 그 판으로 평가하게 되고,
        #   선택 편향을 재려고 만든 도구가 편향을 그대로 들고 온다.
        a.variant_offset = a.seen_variants
    if a.variants < 1:
        ap.error("--variants 는 1 이상이어야 한다")
    if a.variant_offset < 0:
        ap.error("--variant-offset 은 0 이상이어야 한다")
    if a.eval_seeds < 1:
        ap.error("--eval-seeds 는 1 이상이어야 한다")
    if a.variant_offset < a.seen_variants and not a.include_seen:
        ap.error(f"평가 창 v{a.variant_offset}~v{a.variant_offset + a.variants - 1} 가 수집이 쓴"
                 f" 변종 v0~v{a.seen_variants - 1} 과 겹친다 — 고른 판으로 성적을 내면 라운드"
                 " 5 개 중 최선을 고른 편향이 그대로 성적표에 들어간다. 안 쓴 변종으로 재거나"
                 "(--variant-offset 을 --seen-variants 이상으로) 일부러 겹치려면"
                 " --include-seen 을 줘라.")
    try:
        for name in a.stage:
            stage_path(name)
    except ValueError as exc:
        ap.error(str(exc))

    window = window_label(a.variants, a.variant_offset)
    if a.dry_run:
        print("\n".join(dry_run_lines(a, window)))
        return 0
    if not os.path.exists(a.ckpt):
        ap.error(f"--ckpt 체크포인트가 없다: {a.ckpt}")

    dev = pick_device(a.device)
    net = DrivePolicy.load(a.ckpt, device=dev)
    net.clamp_log_std()          # 옛 체크포인트의 log_std 가 지금 범위 밖일 수 있다

    config = None
    if a.lim_anticipate or a.understeer or a.lat_accel_max:
        config = EnvConfig(obs=ObsConfig(lim_anticipate=bool(a.lim_anticipate)),
                           world=WorldConfig(dynamics=DynamicsParams(understeer=a.understeer,
                                                                     lat_accel_max=a.lat_accel_max)))
    t0, rows = time.perf_counter(), []
    for name in a.stage:
        boards, varied = stage_boards(name, a.variants, a.variant_offset)
        print(f"[{name}] 판 {len(boards)}개 [{window if varied else '원본'}] …",
              file=sys.stderr, flush=True)
        row = evaluate_stage(net, boards, a.eval_seeds, config=config)
        row.update({"boards": len(boards), "varied": varied,
                    "window": window if varied else "original"})
        rows.append((name, row))
        print(f"    완주율 {row['goal_rate'] * 100:.1f}% · 점수 {row['mean_score']:.1f}"
              f" · 총걸음 {row['total_steps']}", file=sys.stderr, flush=True)

    lines = report_lines(a, rows, window, dev)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))

    summary = {"ckpt": a.ckpt, "stages": {n: r for n, r in rows},
               "seen_variants": a.seen_variants, "variants": a.variants,
               "variant_offset": a.variant_offset, "window": window,
               "include_seen": a.include_seen, "eval_seeds": a.eval_seeds,
               "report": a.out, "seconds": time.perf_counter() - t0}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""선생님(규칙 스택)이 **임의 커리큘럼 단계**를 달린 성적을 낸다.

    env -u PYTHONPATH .venv/bin/python scripts/measure_teacher.py \
        --stage stage1 --stage stage2 --stage stage3 --seeds 1 \
        --out docs/reports/m4h-stage3-teacher.md

`scripts/run_stage1_teacher.py` 가 단계 ① 전용으로 하던 일의 일반화다. 다른 점 둘:

1. **월드가 아니라 환경(`VtdDriveEnv`)으로 몬다.** `run_stage1_teacher.py` 는
   `rollout.run_teacher_episode`(월드 단독)를 쓰는데, 월드는 충돌을 모른다
   (`drive_env.py:142-143`: 충돌은 **심판**이 낸 항목이 종료 사유다). 액터가 있는 판에서
   월드만으로 재면 유령 충돌로 −50 을 먹고도 `outcome == "goal"` 이 나온다 — 단계 ③ 에서
   정확히 쓸모없는 측정이다. 그래서 `policy.evaluate.evaluate_teacher`(환경+심판)를 쓴다.
2. **단계를 여러 개 받는다.** 단계별로 `evaluate_teacher` 를 따로 부른다 —
   `stage1/2/3.json` 은 판 **이름**을 그대로 공유하는데 `VtdDriveEnv._worlds` 가 이름으로
   세계를 캐시하므로(`drive_env.py:89-96`), 한 환경에 섞어 넣으면 다른 단계의 세계를 준다.

종료 코드는 `run_stage1_teacher.py` 와 같은 뜻이다 — **모든 단계가 100% 완주해야 0**.
단계 ③ 에서 이 코드가 0 이 아니면 "규칙 스택이 못 푸는 판을 만들었다"는 뜻이다.
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
PULLOVER_ENV = os.environ.get("PULLOVER", "unset")   # drive.py 가 import 때 읽는다 — import 전에 기록
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_teacher, violation_counts  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

CURRICULA = os.path.join(REPO, "curricula")


def stage_path(stage: str) -> str:
    """단계 이름 → 커리큘럼 파일 경로.

    `"stage3"` 처럼 이름만 주면 `curricula/stage3.json`, 경로나 `.json` 을 주면 그대로 쓴다.
    """
    if stage.endswith(".json") or os.sep in stage:
        return stage
    return os.path.join(CURRICULA, f"{stage}.json")


def load_stage(stage: str, only=None):
    """(커리큘럼 이름, 판 목록). `only` 는 판 이름 목록으로 걸러낸다(배치 손보기용).

    오타로 아무 판도 안 남으면 **터뜨린다** — 판 0 개짜리 측정은 완주율 0/0 이라
    "전부 완주" 처럼 읽히는 빈 성적표를 만든다(`run_m2b_env.py` 가 같은 이유로 막는다).
    """
    label, boards = load_curriculum(stage_path(stage))
    if only:
        want = list(dict.fromkeys(only))
        have = {b.name for b in boards}
        missing = [n for n in want if n not in have]
        if missing:
            raise ValueError(f"단계 '{stage}' 에 없는 판: {', '.join(missing)} "
                             f"(있는 판: {', '.join(sorted(have))})")
        boards = [b for b in boards if b.name in set(want)]
    if not boards:
        raise ValueError(f"단계 '{stage}' 에 판이 하나도 없다")
    return label, boards


def completion(ev: dict):
    """(완주 판 수, 전체 판 수).

    완주는 `outcome == "goal"` **하나뿐**이다. `collision`·`offroad`·`timeout`·`stalled`
    는 전부 미완주다 — 특히 `collision` 은 단계 ③ 에서 처음 열리는 종료 사유라
    "goal 이 아닌 것" 을 느슨하게 세면 배치가 틀렸는데도 100% 로 보인다.
    """
    episodes = ev["episodes"]
    return sum(1 for e in episodes if e.outcome == "goal"), len(episodes)


def board_rows(ev: dict) -> list:
    """판마다 한 줄 — 어느 판이 못 달리는지는 평균만 봐서는 안 보인다."""
    order, by = [], {}
    for e in ev["episodes"]:
        if e.board not in by:
            order.append(e.board)
            by[e.board] = []
        by[e.board].append(e)
    rows = []
    for name in order:
        eps = by[name]
        rows.append({"board": name,
                     "goal": sum(1 for e in eps if e.outcome == "goal"),
                     "n": len(eps),
                     "mean_score": sum(e.score for e in eps) / len(eps),
                     "outcomes": ", ".join(f"{e.outcome}({e.steps}보)" for e in eps)})
    return rows


def measure(stage: str, seeds: int = 1, only=None, config=None) -> dict:
    """한 단계를 재고 성적표 한 절에 필요한 것을 모두 담아 돌려준다."""
    if seeds < 1:
        raise ValueError(f"--seeds 는 1 이상이어야 한다: {seeds}")
    label, boards = load_stage(stage, only)
    ev = evaluate_teacher(boards, seeds=tuple(range(seeds)), config=config)
    done, n = completion(ev)
    return {"stage": stage, "label": label, "seeds": seeds,
            "lengths": {b.name: b.route.total for b in boards},
            "actors": {b.name: len(b.scenario.actors) for b in boards},
            "signals": {b.name: b.signals for b in boards},
            "ev": ev, "completed": done, "episodes": n,
            "rows": board_rows(ev), "violations": violation_counts(ev)}


def describe_rule_stack() -> str:
    """서브모듈 작업트리까지 반영한 이름 — 고친 채로 돌렸으면 `-dirty` 가 붙는다."""
    return subprocess.run(["git", "-C", rs.ROOT, "describe", "--always", "--dirty"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _circled(n: int) -> str:
    return chr(0x2460 + n - 1)


def summary_lines(results: list) -> list:
    lines = ["## 단계별 요약", "",
             "| 단계 | 커리큘럼 | 판 | 액터 | 완주 | 완주율 | 평균 점수 | 완주 판만 평균 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ev = r["ev"]
        done, n = r["completed"], r["episodes"]
        comp = ev["mean_score_completed"]
        lines.append(f"| `{r['stage']}` | {r['label']} | {len(r['lengths'])} | "
                     f"{sum(r['actors'].values())} | {done}/{n} | {done / max(n, 1) * 100:.0f}% | "
                     f"{ev['mean_score']:.1f} | "
                     + (f"{comp:.1f} |" if comp is not None else "— |"))
    return lines


def board_lines(r: dict) -> list:
    lines = ["", f"### `{r['stage']}` — 판별 결과 (시드 {r['seeds']}개)", "",
             "| 판 | 경로 길이 [m] | 액터 | 신호 | 완주 | 평균 점수 | 결과(걸음수) |",
             "|---|---:|---:|---|---:|---:|---|"]
    for row in r["rows"]:
        name = row["board"]
        lines.append(f"| {name} | {r['lengths'][name]:.0f} | {r['actors'][name]} | "
                     f"{r['signals'][name]} | {row['goal']}/{row['n']} | "
                     f"{row['mean_score']:.1f} | {row['outcomes']} |")
    return lines


def violation_lines(results: list) -> list:
    """항목별 감점 — **선생님이 감점을 받는 지점이 곧 RL 의 표적이다.**"""
    items = sorted({i for r in results for i in r["violations"]})
    head = " | ".join(f"`{r['stage']}` 경미 | `{r['stage']}` 중대" for r in results)
    lines = ["", "## 항목별 감점(구간-슬롯 수)", ""]
    if not items:
        return lines + ["어느 단계에서도 감점이 없다."]
    lines += [f"| 항목 | {head} |", "|---|" + "---:|" * (2 * len(results))]
    for item in items:
        cells = []
        for r in results:
            c = r["violations"].get(item, {})
            cells += [str(c.get("minor", 0)), str(c.get("major", 0))]
        name = rs.score_fma.ITEMS.get(item, "?")
        lines.append(f"| {_circled(item)} {name} | " + " | ".join(cells) + " |")
    return lines


def report_lines(results: list, bench: str | None = None) -> list:
    lines = ["# 선생님(규칙 스택) 기준선 — 커리큘럼 단계별 성적", "",
             f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` "
             f"· Python {platform.python_version()}",
             f"- 규칙 스택 커밋 `{rs.commit()}` · 작업트리 `{describe_rule_stack()}`",
             f"- 환경변수 `PULLOVER` = `{PULLOVER_ENV}` "
             f"(규칙 스택 `PULLOVER_MODE` = `{rs.DrivingStack.PULLOVER_MODE}`)",
             f"- 잰 단계: {', '.join('`' + r['stage'] + '`' for r in results)}", ""]
    lines += summary_lines(results)
    for r in results:
        lines += board_lines(r)
    lines += violation_lines(results)
    lines += ["", "## 이 성적표가 주장하지 않는 것", "",
              "- **학생 성적이 아니다.** 선생님(규칙 스택)만 잰 기준선이다.",
              "- **추월을 잘하는지 말하지 않는다.** 채점 항목에 추월이 없다"
              "(`score_fma.ITEMS`) — ④중앙선·⑥실선 차로변경·⑬지시등으로만 간접적으로 찍힌다.",
              "- **액터의 `type` 은 판정에 안 쓰인다.** `world/actors.py:30-31` 이 `rs.Obj` 로"
              " 감쌀 때 버린다(실제 9910 패킷에 종류가 없다) — 분류는 전부 바운딩박스 치수다.",
              "- 시드를 늘려도 단계 ③ 의 액터는 고정이다. 시드가 바꾸는 것은 신호 위상뿐이다."]
    if bench:
        lines += ["", f"- 참고: {bench}"]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="선생님을 임의 커리큘럼 단계에서 잰다")
    ap.add_argument("--stage", action="append", required=True,
                    help="커리큘럼 이름(stage3) 또는 파일 경로. 여러 번 줄 수 있다")
    ap.add_argument("--seeds", type=int, default=1, help="시드 개수(0..N-1)")
    ap.add_argument("--boards", default="", help="쉼표로 구분한 판 이름 — 배치를 손볼 때만")
    ap.add_argument("--out", default="", help="성적표 마크다운 경로(안 주면 안 쓴다)")
    ap.add_argument("--note", default="", help="성적표 맨 뒤에 붙일 한 줄")
    a = ap.parse_args(argv)
    only = [s.strip() for s in a.boards.split(",") if s.strip()]

    results = []
    for stage in a.stage:
        r = measure(stage, seeds=a.seeds, only=only)
        results.append(r)
        for row in r["rows"]:
            print(f"{stage} {row['board']}: {row['goal']}/{row['n']} 완주 "
                  f"· 점수 {row['mean_score']:.1f} · {row['outcomes']}", flush=True)
        print(f"{stage}: 완주 {r['completed']}/{r['episodes']} "
              f"· 평균 점수 {r['ev']['mean_score']:.1f}", flush=True)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines(results, a.note or None)) + "\n")
        print(f"성적표: {a.out}")

    ok = all(r["completed"] == r["episodes"] for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

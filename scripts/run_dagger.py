"""M3 — DAgger 라운드 반복과 성적표.

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py \
        --out runs/lab-main/$(date +%F)-dagger --rounds 5 --seeds 2 --workers 12
    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py --smoke --out /tmp/smoke

단계 ③(액터 있는 판)에서 모을 때 — 판 변종 4 개, 평가는 ①②③ 전부, **출발점은 M3 학생**:

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py \
        --out runs/omen/$(date +%F)-dagger-s3 --stage stage3 --variants 4 \
        --eval-stage stage1 --eval-stage stage2 --eval-stage stage3 --workers 24 \
        --init runs/omen/2026-09-26-m3-squash-warm/policy.pt

`--init` 없이 단계 ③ 을 모으면 **운전을 처음부터 다시 배운다** — 2026-09-30 실측: 라운드 0
(β=1.0, 선생님만 몬 데이터)의 학생이 단계 ① 완주율 16.7% 로 나왔다(M3 학생은 100%/98.60).
5 라운드로는 M3 가 5 라운드에 걸쳐 배운 것을 더 어려운 판에서 다시 못 배운다.

무엇을 모으고 무엇을 평가할지만 먼저 확인할 때(아무것도 안 돌린다):

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py \
        --dry-run --stage stage3 --variants 4

성적표는 기본이 `<--out>/report.md` 다 — `docs/reports/` 아래 커밋된 성적표를 덮어쓰려면
그 경로를 **직접** 적어야 한다. 커밋된 M3 성적표를 다시 만드는 명령은 이것이다:

    env -u PYTHONPATH .venv/bin/python scripts/run_dagger.py --report-only \
        --out runs/lab-main/2026-09-17-dagger-fix2 \
        --report docs/reports/m3-dagger.md --history ... --history-note "..."
"""
import argparse
import datetime
import glob
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.collect import collect_episode  # noqa: E402
from vtd_rl.policy.dataset import load_dir, load_shard, save_shard  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher, violation_counts  # noqa: E402
from vtd_rl.policy.net import DrivePolicy, PolicyConfig  # noqa: E402
from vtd_rl.policy.train import (TrainConfig, make_reference, squash_aligned,  # noqa: E402
                                 train_epochs)
from vtd_rl.world.board import load_board, load_curriculum, slice_board  # noqa: E402

BETAS = [1.0, 0.5, 0.25, 0.1, 0.0]
H = {"name": "course_H", "route": "routes/HL_FMA_NEW_H.json", "lane": "routes/HL_FMA_NEW_H_lane.json"}
STOPPED_SPEED = 0.02   # vec[:,0] = ego 속도/25 클립값 — 0.02 는 약 0.5 m/s 이하, "거의 정지"

#: 아무것도 안 주면 예전 그대로 — 수집은 단계 ① 하나, 평가는 ①②.
DEFAULT_COLLECT_STAGES = ["stage1"]
DEFAULT_EVAL_STAGES = ["stage1", "stage2"]

#: 성적표 기본 파일 이름. **산출물 폴더 안**에 쓴다 — 예전 기본값은 `docs/reports/m3-dagger.md`
#: 였고, 그래서 `--report` 를 안 준 실행이 커밋된 M3 성적표를 조용히 덮어썼다(2026-09-30 실측:
#: 단계 ③ DAgger 가 M3 성적표 자리에 실패한 결과를 써 버렸다).
DEFAULT_REPORT_NAME = "report.md"

#: 성적표에 쓸 사람 말 이름. 없는 단계는 이름을 그대로 쓴다 — 기본 실행의 성적표 문구가
#: 예전과 한 글자도 안 달라지게 하려는 것이다(옛 성적표와 나란히 놓고 읽는다).
STAGE_LABELS = {"stage1": "단계 ①", "stage2": "단계 ②", "stage3": "단계 ③",
                "stage4": "단계 ④", "stage5": "단계 ⑤"}

#: 어느 라운드를 **채택**하는가(`--select`). 기본은 `last` — 예전 그대로다.
SELECT_CHOICES = ("last", "best")

#: 선택 기준(`--select-metric`).
#:  * `primary`            — **예전 기준**. 첫 평가 단계(`primary`)의 완주율만 본다.
#:  * `retain-then-learn`  — 첫 평가 단계(**유지**)가 `--select-floor` 이상인 라운드 중에서
#:                           마지막 평가 단계(**학습**) 완주율이 가장 높은 라운드.
SELECT_METRICS = ("primary", "retain-then-learn")

#: `--select-metric` 을 안 줬을 때 `--select` 가 정하는 기본 기준. 둘을 묶어 두는 이유:
#:  * `last`(기본 실행) → `primary`. 성적표의 "가장 좋았던 라운드" 줄이 **한 글자도 안 바뀐다**
#:    (커밋된 M3 성적표가 그 줄을 들고 있다 — `가장 좋았던 라운드: 4`).
#:  * `best` → `retain-then-learn`. 라운드를 진짜로 고르러 왔으면 이 마일스톤의 기준이 기본이다.
#:    `--select best` 를 주고 기준을 깜빡하면 **단계 ① 만 보는 옛 기준**으로 단계 ③ 실행을
#:    고르게 되는데, 그건 "아무것도 안 배운 라운드" 를 뽑는 길이다.
DEFAULT_SELECT_METRIC = {"last": "primary", "best": "retain-then-learn"}

#: `retain-then-learn` 의 유지 문턱(완주율). 단계 ① 은 **독립 판이 6 개**라 해상도가
#: 16.7 점이고, "한 판만 잃었다" 가 5/6 = 83.3% 다. 문턱을 5/6 으로 그대로 쓰면 실측값
#: 0.8333… 의 부동소수 표현 하나로 라운드가 떨어질 수 있어 **그 아래**로 둔다. 4/6=66.7%
#: 보다는 위라 "두 판 잃음" 은 확실히 떨어진다.
SELECT_FLOOR = 0.8

#: 채택된 라운드 체크포인트의 **복사본** 이름. 심링크가 아니라 복사다 — 산출물 폴더는
#: 머신 사이를 rsync·zip·scp 로 옮겨 다니는데 그 과정에서 심링크는 깨지거나 풀리고,
#: 깨진 심링크는 "조용히 없는 파일" 이 된다. 300 KB 한 벌이 그보다 싸다.
BEST_CKPT_NAME = "policy-best.pt"


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name, name)


def stage_path(name: str) -> str:
    """단계 이름 → 커리큘럼 경로. 모르는 이름은 **조용히 넘기지 않는다**.

    `.json` 으로 끝나거나 경로 구분자가 있으면 경로로 본다(테스트가 임시 커리큘럼을 쓴다).
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


def resolve_report(report, out) -> str:
    """성적표 경로 — 안 주면 `<out>/report.md` 다. **어떤 인자 조합에서도 레포 문서를 안 고른다.**

    `--stage` 를 줬든 안 줬든, `--report-only` 든 `--smoke` 든 마찬가지다. 커밋된 성적표
    (`docs/reports/m3-dagger.md`)는 실행마다 새로 나오는 산출물이 아니라 M3 의 증거물이라,
    덮어쓰려면 그 경로를 `--report` 로 **직접** 적어야 한다(한 번 더 타이핑하는 대신, 실수로
    덮어쓸 길이 없다). 예전 기본값이 바로 그 문서였고 실제로 한 번 덮어썼다.
    """
    return report if report else os.path.join(out, DEFAULT_REPORT_NAME)


def resolve_stages(stage, eval_stage):
    """(수집 단계, 평가 단계) 를 정한다 — **순서가 뜻이다**.

    앞이 **수집**(학생이 몰고 선생님이 라벨을 다는 판), 뒤가 **평가**다. 둘을 뒤바꾸면
    단계 ③ 을 모으라고 시켜 놓고 ①② 만 모으게 되는데, 성적표는 멀쩡해 보인다.
    """
    collect = list(stage) or list(DEFAULT_COLLECT_STAGES)
    evals = list(eval_stage) or list(DEFAULT_EVAL_STAGES)
    return collect, evals


def select_stages(eval_names):
    """`(유지 단계, 학습 단계)` — **앞이 유지, 뒤가 학습**이다.

    평가 단계를 준 순서가 그대로 뜻이 된다(`resolve_stages` 와 같은 규칙). 단계 ①②③ 을
    주면 유지는 ①, 학습은 ③ 이다. 단계가 하나뿐이면 둘이 같은 단계가 되고, 그러면 문턱이
    공허해져 `retain-then-learn` 은 사실상 `primary` 와 같아진다 — 그게 맞다.
    """
    names = list(eval_names or [])
    if not names:
        raise ValueError("평가 단계가 없다 — 선택 기준을 세울 수 없다")
    return names[0], names[-1]


def gated_rounds(rounds, retain: str, floor: float) -> list:
    """유지 단계 완주율이 문턱 이상인 라운드들(없으면 빈 목록)."""
    return [r for r in rounds if r[retain]["goal_rate"] >= floor]


def pick_round(rounds, select, metric=None, eval_names=None, floor: float = SELECT_FLOOR):
    """**채택할** 라운드 한 줄을 돌려준다. 이 함수 하나가 최종 평가·요약·산출 체크포인트를 정한다.

    ★ 왜 기준이 두 단계를 같이 보는가. 한 단계만 보면 어느 쪽이든 망한다 —
    학습 단계(③)만 보면 **유지(①)를 부순 라운드**가 뽑히고, 유지만 보면 **아무것도 안 배운
    라운드**가 뽑힌다. 이 마일스톤의 목표가 "① 을 지키면서 ③ 을 배운다" 이므로 기준도 그
    모양이어야 한다: **유지가 문턱 이상인 라운드 중에서 학습 최대**.

    문턱을 넘은 라운드가 **하나도 없으면** 학습 최대가 아니라 **유지 최대**로 되돌린다.
    전부 부서진 판에서 "단계 ③ 가 제일 높은 폐허" 를 뽑으면 완주율 0% 짜리 그물을 성적표
    대표로 내보내게 된다. 되돌렸다는 사실은 `select_note` 가 문장으로 말한다.

    동점이면 **더 뒤 라운드**다 — DAgger 는 라운드가 갈수록 데이터가 쌓이므로 같은 성적이면
    뒤가 낫고, 예전 `:594` 의 동률 규칙도 그것이었다(옛 성적표가 그 규칙으로 나왔다).
    """
    if not rounds:
        raise ValueError("라운드가 없다 — 고를 것이 없다")
    if select not in SELECT_CHOICES:
        raise ValueError(f"모르는 --select 값 {select!r} — {SELECT_CHOICES} 중 하나여야 한다")
    if select == "last":
        return rounds[-1]
    metric = metric or DEFAULT_SELECT_METRIC[select]
    if metric not in SELECT_METRICS:
        raise ValueError(f"모르는 --select-metric 값 {metric!r} — {SELECT_METRICS} 중 하나여야 한다")
    retain, learn = select_stages(eval_names)
    if metric == "primary":
        return max(rounds, key=lambda r: (r[retain]["goal_rate"], r["round"]))
    pool = gated_rounds(rounds, retain, floor)
    if pool:
        return max(pool, key=lambda r: (r[learn]["goal_rate"], r[retain]["goal_rate"],
                                        r["round"]))
    return max(rounds, key=lambda r: (r[retain]["goal_rate"], r[learn]["goal_rate"], r["round"]))


def select_note(select, metric, floor, rounds, eval_names) -> str:
    """선택이 **무엇을 보고** 골랐는지 한 문장 — 성적표와 요약에 글자 그대로 들어간다.

    기준이 코드 안에만 있으면 성적표를 읽는 사람이 "최선" 이 무슨 뜻인지 알 수 없다.
    """
    if select == "last":
        return "마지막 라운드를 그대로 채택(선택 안 함)"
    retain, learn = select_stages(eval_names)
    if metric == "primary":
        return f"`primary`: {stage_label(retain)} 완주율 최대(동점이면 더 뒤 라운드)"
    passed = len(gated_rounds(rounds, retain, floor))
    if not passed:
        return (f"`retain-then-learn`: {stage_label(retain)} 완주율 ≥ {floor * 100:.1f}% 인"
                f" 라운드가 **하나도 없어** {stage_label(retain)} 완주율 최대로 되돌렸다")
    return (f"`retain-then-learn`: {stage_label(retain)} 완주율 ≥ {floor * 100:.1f}% 인 라운드"
            f" {passed}/{len(rounds)} 개 중 {stage_label(learn)} 완주율 최대"
            f"(동점이면 {stage_label(retain)}, 그다음 더 뒤 라운드)")


def has_actors(stage_name: str) -> bool:
    """그 단계 커리큘럼에 액터가 한 개라도 있는가(판을 짓지 않고 JSON 만 본다)."""
    with open(stage_path(stage_name), encoding="utf-8") as f:
        d = json.load(f)
    return any(e.get("actors") for e in d["boards"])


def check_smoke(stages, smoke: bool):
    """★ `--smoke` 는 `slice_board` 로 코스 H 의 0~250 m 조각을 쓰는데,

    `slice_board` 는 **액터·신호등·구역을 통째로 버린다**(`world/board.py` 의
    `actors=[], lights=[], zones=[]`). 그래서 액터가 있는 단계에 `--smoke` 를 쓰면
    **액터 0 개인 판에서 아무것도 시험하지 않고 초록불로 통과한다** — 이 마일스톤이
    반드시 막아야 하는 조용한 실패다. 조각내기의 좌표계를 건드리는 대신 여기서 거부한다.
    """
    if not smoke:
        return
    bad = [s for s in stages if has_actors(s)]
    if bad:
        raise ValueError(
            f"--smoke 는 액터가 있는 단계에 못 쓴다: {', '.join(bad)} — `slice_board` 가 "
            "액터를 버려서(world/board.py) 액터 0 개인 판을 시험하게 된다. "
            "--smoke 를 빼고 돌리거나 액터 없는 단계(stage1·stage2)를 써라.")


def build_jobs(targets, rnd, beta, policy_path, data_dir, smoke, variants, seeds):
    """한 라운드의 수집 작업 묶음. **판 이름과 변종 수가 같이 실려야** 일꾼이 그 판을 다시 짓는다."""
    return [(path, name, 1000 * rnd + s, beta, policy_path, data_dir, rnd, smoke, variants)
            for path, name in targets for s in range(seeds)]


def job_board(job):
    """작업 묶음이 가리키는 판을 다시 짓는다 — 수집은 spawn 프로세스라 묶음만 보고 짓는다."""
    curriculum, name, _seed, _beta, _policy, _out, _rnd, smoke, variants = job
    boards = _boards(curriculum, smoke, name, variants)
    if not boards:
        # 변종 수가 안 실려 오면 `course_A@v1` 을 못 찾는다 — IndexError 말고 이름을 말한다.
        raise ValueError(f"작업이 가리키는 판이 없다: {name!r} (커리큘럼 {curriculum}, "
                         f"변종 {variants})")
    return boards[0]


def _collect_job(job):
    _curriculum, name, seed, beta, policy_path, out, rnd, _smoke, _variants = job
    board = job_board(job)
    policy = DrivePolicy.load(policy_path) if policy_path else None
    shard = collect_episode(board, policy=policy, beta=beta, seed=seed)
    save_shard(shard, os.path.join(out, f"r{rnd}-{name}-s{seed}.npz"))
    return shard.meta


def _boards(curriculum, smoke, only=None, variants=1):
    if smoke:
        boards = [slice_board(load_board(H), 0.0, 250.0, "H_0_250")]
    else:
        _n, boards = load_curriculum(curriculum, variants=variants)
    return [b for b in boards if only is None or b.name == only]


def collect_targets(paths, smoke, variants):
    """수집할 (커리큘럼 경로, 판 이름) 목록 — 단계가 여럿이면 순서대로 이어 붙인다."""
    out = [(p, b.name) for p in paths for b in _boards(p, smoke, variants=variants)]
    names = [n for _p, n in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        # 조각 파일명이 `r{라운드}-{판}-s{시드}.npz` 라 이름이 겹치면 **서로 덮어쓴다** —
        # 데이터가 조용히 절반으로 준다. 단계 ①③ 처럼 코스 이름이 같은 조합에서 실제로 난다.
        raise ValueError(f"수집 판 이름이 여러 단계에서 겹친다: {', '.join(dup)} — 조각 파일이"
                         " 서로 덮어쓴다. 한 번에 한 단계씩 모으거나 판 이름을 갈라라.")
    return out


def eval_boards(names, smoke):
    """평가에 쓸 `(단계 이름, 판 목록)` 짝 — **이름과 판을 한 곳에서 같이 만든다.**

    따로 만들어 `zip` 으로 붙이면 이름과 판이 어긋날 수 있고, 그러면 로그의 `"stage3"`
    열쇠에 단계 ① 결과가 들어앉는다 — 숫자는 멀쩡해 보이고 이름만 거짓말이 된다.

    ★ 평가는 **변종을 안 쓴다**(`variants=1`). 성적표의 "단계 ③ 완주율" 은 출발점
    0/18(판 6 개 × 시드 3 개)과 비교하는 숫자라, 판 수가 변종에 따라 달라지면 비교가
    깨진다. 변종은 **수집에만** 건다.
    """
    return [(n, _boards(stage_path(n), smoke)) for n in names]


def _new_net(smoke, dev):
    cfg = PolicyConfig(trunk=(64, 64)) if smoke else PolicyConfig()
    return DrivePolicy(cfg).to(dev)


def round_start_net(init_path, smoke, dev):
    """그 라운드의 학습이 **출발할** 그물.

    `--init` 이 없으면 예전 그대로 새 그물이다(기본값은 안 바뀐다 — 커밋된 M3 성적표가 그
    동작으로 나왔다). 있으면 그 `DrivePolicy` 체크포인트를 읽어 거기서 이어 간다. 체크포인트를
    읽는 길은 수집이 이전 라운드 그물을 집어 오는 길(`_collect_job` 의 `DrivePolicy.load`)과
    **같은 것 하나**다 — 웜스타트 수단을 따로 만들지 않는다.

    ★ 왜 라운드 0 만이 아니라 **매 라운드** 이 체크포인트에서 출발하는가:
    라운드마다 새로 짓는 규칙(아래 학습 루프 주석)이 막으려던 것은 *이전 라운드* 그물을
    이어 써서 **에폭이 쌓이는 것**이다(라운드 7 이면 64 에폭 → log_std 붕괴). 매 라운드
    **같은 고정 체크포인트**에서 출발하면 라운드마다 정확히 `--epochs` 에폭이고 에폭이 안
    쌓인다 — 그 규칙을 그대로 지키면서 M3 학생을 안 버린다. 라운드 0 만 웜스타트하면
    라운드 1 에서 도로 버려져 결함이 반쯤만 고쳐진다.

    `--smoke` 의 작은 몸통(64,64)은 `--init` 이 있으면 안 쓴다 — 모양은 체크포인트가 정한다.
    """
    if not init_path:
        return _new_net(smoke, dev)
    net = DrivePolicy.load(init_path, device=dev)
    # 옛 체크포인트의 log_std 가 지금 설정의 범위 밖일 수 있다(`refit_m3.py` 도 같은 일을 한다).
    net.clamp_log_std()
    return net


def _stall_start_count(data_dir, rnd, names, num_seeds) -> int:
    """이 라운드에 새로 모은 조각만 다시 읽어(재수집 없이) 센다 —

    자차가 거의 정지(`vec[:,0] < STOPPED_SPEED`)했는데 선생님 라벨은 출발하라고 한
    (`control[:,1] > 0`, 곧 양의 가속) 표본 수. 학생이 멈춰서 못 움직이는 실패 양상을
    교정하는 바로 그 신호이므로, 라운드가 갈수록 느는지 주는지가 M4 의 출발점이다.
    이 조각을 몬 것은 "이번 라운드에 새로 나온 그물"이 아니라 그 라운드가 시작할 때 있던
    이전 라운드 체크포인트(라운드 0 은 선생님 전용)다 — 보고서에도 그대로 밝힌다.
    """
    n = 0
    for name in names:
        for s in range(num_seeds):
            seed = 1000 * rnd + s          # _collect_job 이 파일명에 쓰는 것과 같은 시드 계산
            shard = load_shard(os.path.join(data_dir, f"r{rnd}-{name}-s{seed}.npz"))
            n += int(((shard.vec[:, 0] < STOPPED_SPEED) & (shard.control[:, 1] > 0.0)).sum())
    return n


def _distinct_episodes(ev: dict) -> int:
    """서로 다른(코스, 결과, 걸음수) 조합 수 — 정책이 결정적이면 액터·신호가 고정인 판은

    시드를 바꿔도 똑같은 판이 나온다(단계 ①). "완주율 X%" 가 실제로 몇 가지 서로 다른 판을
    본 것인지 밝혀 둔다.
    """
    return len({(e.board, e.outcome, e.steps) for e in ev["episodes"]})


def _violation_lines(label: str, student_ev: dict, teacher_ev: dict) -> list:
    """마지막 라운드 학생 대 선생님, 항목별 위반 — 완주율·평균 점수만 보면 안 보이는 규칙 위반

    차이를 남긴다(정지한 차는 대부분의 항목을 안 어겨서 점수만으로는 안 보인다).
    """
    sc, tc = violation_counts(student_ev), violation_counts(teacher_ev)
    items = sorted(set(sc) | set(tc))
    lines = ["", f"### {label} — 항목별 위반(구간-슬롯 수, 마지막 라운드)"]
    if not items:
        return lines + ["", "학생·선생님 모두 위반 없음."]
    lines += ["", "| 항목 | 학생 minor | 학생 major | 선생님 minor | 선생님 major |",
             "|---:|---:|---:|---:|---:|"]
    for item in items:
        s, t = sc.get(item, {}), tc.get(item, {})
        lines.append(f"| {item} | {s.get('minor', 0)} | {s.get('major', 0)} | "
                     f"{t.get('minor', 0)} | {t.get('major', 0)} |")
    return lines


def _teacher_caveat(teacher2: dict, label: str = "단계 ②(신호 주기)") -> str:
    """두 번째 평가 단계 선생님의 위반을 감점 시트에서 직접 뽑아 낸다(레포에 박아 넣지 않는다).

    Task 1 에서 이미 본 대로 신호 주기에서는 선생님도 무결하지 않다(빨간불 위반·차선 관련 경미 감점) —
    그래도 전부 완주한다. "선생님보다 10점 이내"를 비교할 때 그 선생님 점수 자체가 이미 이런 감점을
    포함한 값임을 밝혀 둔다.
    """
    by_board: dict = {}
    for e in teacher2["episodes"]:
        for section in e.sheet:
            for item, grade in section.items():
                by_board.setdefault(e.board, set()).add((item, grade))
    if not by_board:
        return f"선생님은 {label}에서도 감점 없이 전부 완주했다."
    parts = [f"{board} " + ", ".join(f"항목{i}({g})" for i, g in sorted(items))
             for board, items in sorted(by_board.items())]
    return (f"선생님도 {label}에서 무결하지는 않다 — " + "; ".join(parts) +
            f" — 그래도 여섯 코스 모두 완주했다(완주율 {teacher2['goal_rate']*100:.0f}%, "
            f"점수 {teacher2['mean_score']:.1f}).")


def _final_outcome_lines(ev1: dict, label: str = "단계 ①") -> list:
    by_board: dict = {}
    for e in ev1["episodes"]:
        by_board.setdefault(e.board, []).append(f"{e.outcome}({e.steps}보)")
    lines = ["", f"### 마지막 라운드 — {label} 코스별 결과(시드 순)", "",
             "| 코스 | 결과(걸음수) |", "|---|---|"]
    for board, outs in by_board.items():
        lines.append(f"| {board} | {' '.join(outs)} |")
    return lines


def _cell(row: dict, stage: str, key: str, fmt: str) -> str:
    """로그 행에서 단계별 숫자 한 칸 — 그 단계가 없는 옛 로그는 '—' 로 둔다."""
    v = (row.get(stage) or {}).get(key)
    return "—" if v is None else format(v * 100 if key == "goal_rate" else v, fmt)


def _history_lines(history_paths, history_notes, stages=None) -> list:
    """'이전 실행' 절 — 무엇을 보여 주는지는 호출자가 --history-note 로 직접 적는다.

    (이 함수가 그 내용을 대신 짐작해 박아 넣지 않는다 — 그러면 결함 없는 로그도 "결함 있던
    실행"으로 잘못 붙을 수 있다.)
    """
    if not history_paths and not history_notes:
        return []
    stages = list(stages or DEFAULT_EVAL_STAGES)
    head = ("| 라운드 | β | 수집 판(완주) | 누적 표본 | 손실 | "
            + " | ".join(f"{stage_label(s)} 완주율 | {stage_label(s)} 점수" for s in stages) + " |")
    sep = "|---:|---:|---|---:|---:|" + "---:|---:|" * len(stages)
    lines = ["", "## 이전 실행(참고)"]
    for note in history_notes:
        lines += ["", f"- {note}"]
    for path in history_paths:
        # 히스토리 파일 하나가 깨졌다고(없음·JSON 오류·행 형식 다름) 지금 막 나온 실행 성적표까지
        # 못 쓰게 되면 안 된다 — 넓게 잡아 그 파일만 건너뛴다.
        try:
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            table = ["", f"### `{path}`", "", head, sep]
            for r in rows:
                cells = "".join(f" {_cell(r, s, 'goal_rate', '.0f')}% |"
                                f" {_cell(r, s, 'mean_score', '.1f')} |" for s in stages)
                table.append(f"| {r['round']} | {r['beta']} | {r['episodes']}({r['collect_goal']}) | "
                             f"{r['samples']} | {r['train']['loss']:.3f} |" + cells)
            lines += table
        except Exception as exc:  # noqa: BLE001 — 여기서 다 삼켜서 성적표 생성 자체를 지킨다
            lines += ["", f"### `{path}` — 읽지 못함: {type(exc).__name__}: {exc}"]
    return lines


def _dry_run_lines(collect_names, eval_names, smoke, variants, seeds, rounds):
    """무엇을 모으고 무엇을 평가할지, 액터가 실제로 어디에 놓였는지만 찍는다.

    변종을 켰는데 배치가 안 흩어지는 것을 **모으기 전에** 보려고 만든 문이다
    (M4e 에서 아무 일도 안 하는 개입에 실험 한 팔을 날린 적이 있다).
    """
    collect_paths = [stage_path(n) for n in collect_names]
    targets = collect_targets(collect_paths, smoke, variants)
    jobs = build_jobs(targets, 0, 1.0, None, "(dry-run)", smoke, variants, seeds)
    out = [f"수집 단계: {collect_names} (변종 {variants}개)",
           f"평가 단계: {eval_names} (변종 안 씀)",
           f"수집 판 {len(targets)}개 × 시드 {seeds}개 × 라운드 {rounds} = "
           f"{len(jobs) * rounds} 판", ""]
    for path in collect_paths:
        for b in _boards(path, smoke, variants=variants):
            pos = ", ".join(f"{a.id}@({a.motion['pos'][0]:.1f},{a.motion['pos'][1]:.1f})"
                            for a in b.scenario.actors) or "액터 없음"
            out.append(f"  {b.name}: {pos}")
    # 첫 작업 묶음을 실제로 풀어 본다 — 작업에 변종 수가 안 실리면 여기서 드러난다.
    first = job_board(jobs[0])
    out += ["", f"첫 작업 r0-{jobs[0][1]}-s{jobs[0][2]} → 판 {first.name} "
                f"(액터 {len(first.scenario.actors)}개)"]
    for stage, boards_e in eval_boards(eval_names, smoke):
        out.append(f"평가 {stage}: 판 {len(boards_e)}개 "
                   f"(액터 {sum(len(b.scenario.actors) for b in boards_e)}개)")
    return out


def _preflight_lines(a, sel_metric) -> list:
    """`--dry-run` 이 **개입과 선택**도 같이 찍는다 — 띄우기 전에 눈으로 확인하는 자리다.

    M4e 에서 아무 일도 안 하는 개입(`--imitation-floor 0.3`)에 실험 한 팔을 날렸다.
    무엇을 켰는지 모으기 전에 보여 주는 것이 그 재발을 막는 가장 싼 문이다.
    """
    anchor = (f"앵커: anchor_coef={a.anchor_coef:g} · 참조 = `{a.init}` 로 고정(라운드마다 안 바꾼다)"
              if a.anchor_coef > 0.0 else "앵커: 꺼짐(anchor_coef=0, 과거 성적표와 같은 경로)")
    near = (f"회피 가중: near_m={a.near_m:g} m · near_weight={a.near_weight:g}"
            if a.near_m > 0.0 and a.near_weight != 1.0 else "회피 가중: 꺼짐")
    return [anchor, near,
            f"채택: --select {a.select} · 기준 {sel_metric} · 유지 문턱 {a.select_floor:g}",
            f"웜스타트(--init): {a.init or '(없음 — 새 그물에서 시작한다)'}",
            f"배치 순서 시드: 라운드 rnd 는 {a.train_seed} + rnd"
            + ("  ← 예전 그대로(rnd)" if a.train_seed == 0 else "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="산출물 폴더(--dry-run 이 아니면 필수)")
    ap.add_argument("--report", default=None,
                    help=f"성적표 경로(기본 `<--out>/{DEFAULT_REPORT_NAME}`). `docs/reports/` 아래"
                         " 커밋된 성적표를 다시 만들려면 그 경로를 **직접** 적어라 — 기본값이"
                         " 그 문서였을 때 실행 한 번이 M3 성적표를 조용히 덮어썼다")
    ap.add_argument("--init", default=None,
                    help="라운드 학습이 출발할 `DrivePolicy` 체크포인트(웜스타트). 없으면 예전"
                         " 그대로 새 그물에서 시작한다. 이미 운전할 줄 아는 학생(M3:"
                         " runs/omen/2026-09-26-m3-squash-warm/policy.pt) 위에 더 어려운 단계를"
                         " 얹을 때 쓴다 — 없으면 DAgger 가 그 학생을 버리고 처음부터 다시 배운다."
                         " 수집(β 혼합)에 쓰는 이전 라운드 체크포인트와는 별개다: 라운드 0 은"
                         " β=1.0 이라 어차피 선생님만 몬다")
    ap.add_argument("--stage", action="append", default=[],
                    help=f"**수집** 커리큘럼 이름(반복 가능, 기본 {' '.join(DEFAULT_COLLECT_STAGES)})")
    ap.add_argument("--eval-stage", action="append", default=[],
                    help=f"**평가** 커리큘럼 이름(반복 가능, 기본 {' '.join(DEFAULT_EVAL_STAGES)})")
    ap.add_argument("--variants", type=int, default=1,
                    help="판 변종 수 — 수집에만 건다(평가는 늘 원본 판). 액터가 판마다 다르게"
                        " 놓인 판 N 벌이 생긴다(world/place.py 의 jitter)")
    ap.add_argument("--dry-run", action="store_true",
                    help="아무것도 안 돌리고 수집·평가 단계와 액터 배치만 찍고 끝낸다")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="라운드마다 그물을 새로 짓기 전에 거는 torch 시드(라운드 rnd 에는 seed+rnd) —"
                        " 없으면 전역 RNG 상태에 따라 초기값이 실행마다 달라진다."
                        " ⚠ **`--init` 을 주면 가중치를 체크포인트에서 읽어 효과가 없다** —"
                        " '학습 시드' 를 흔들려면 --train-seed 를 써라")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="**배치 순서** 시드의 밑값 — 라운드 rnd 는 `train-seed + rnd` 를 쓴다"
                        "(기본 0 이면 예전 그대로 `rnd`). M4l 실측: 같은 데이터·같은 설정에서"
                        " 배치 순서만 바꿔도 단계 ① 완주율이 16.7%% ↔ 66.7%% 로 흔들리는데"
                        " 손실·MAE·Δθ 는 소수 셋째 자리까지 같다. '학습 시드 3 개' 로 돌리라는"
                        " 말은 이 인자를 0·1·2 로 세 번 돌리라는 뜻이다(--seed 가 아니다)")
    ap.add_argument("--resume", action="store_true",
                    help="--out 에 이미 log.jsonl·조각이 있어도 이어 붙인다(기본은 거부)")
    ap.add_argument("--history", action="append", default=[],
                    help="이전 실행의 log.jsonl 경로(반복 가능) — 성적표에 '이전 실행' 절로 남긴다")
    ap.add_argument("--history-note", action="append", default=[],
                    help="'이전 실행' 절에 그대로 적을 문장(반복 가능) — 그 절이 무엇을 보여 주는지는"
                        " 이 인자로 밝힌다(함수가 짐작하지 않는다)")
    ap.add_argument("--report-only", action="store_true",
                    help="라운드를 다시 돌리지 않고 --out 의 기존 log.jsonl·체크포인트만으로 성적표를 다시 만든다")
    ap.add_argument("--select", choices=SELECT_CHOICES, default="last",
                    help="어느 라운드를 **채택**할지(기본 last = 예전 그대로 마지막 라운드)."
                         " best 면 최종 평가·요약·산출 체크포인트가 전부 고른 라운드가 된다 —"
                         " M4k 에서 r3(83.3%%)를 만들어 놓고 r4(16.7%%)를 내보냈다")
    ap.add_argument("--select-metric", choices=SELECT_METRICS, default=None,
                    help=f"선택 기준(안 주면 --select 를 따라간다: {DEFAULT_SELECT_METRIC})."
                         " primary 는 첫 평가 단계 완주율만 보고, retain-then-learn 은 첫 단계가"
                         " --select-floor 이상인 라운드 중 마지막 평가 단계 완주율을 최대화한다")
    ap.add_argument("--select-floor", type=float, default=SELECT_FLOOR,
                    help=f"retain-then-learn 의 유지 문턱(기본 {SELECT_FLOOR}). 단계 ① 은 독립"
                         " 판이 6 개라 해상도가 16.7 점이고 '한 판만 잃음' 이 83.3%% 다")
    ap.add_argument("--anchor-coef", type=float, default=0.0,
                    help="BC 손실에 더할 **참조 정책 KL**(증류) 계수. 0(기본)이면 앵커 계산"
                         " 자체를 안 하므로 과거 성적표와 비트 단위로 같다. 참조는 --init"
                         " 체크포인트로 **고정**한다(라운드마다 안 바꾼다) — 그래서 >0 이면"
                         " --init 이 반드시 있어야 한다. M4m 최선값은 10 이다")
    ap.add_argument("--near-m", type=float, default=0.0,
                    help="회피 가중의 '가깝다' 문턱[m] — 최근접 물체가 전방 이 거리 안인 표본의"
                         " 손실에 --near-weight 를 곱한다(정규화한다). 0(기본)이면 가중 없음")
    ap.add_argument("--near-weight", type=float, default=1.0,
                    help="그 표본의 손실 배율(기본 1.0 = 가중 없음 경로). M4n 결론은 '기각'"
                         " 이므로 기본 실행에서는 켜지 않는다")
    a = ap.parse_args()
    if a.smoke:
        a.rounds, a.seeds, a.epochs, a.eval_seeds, a.workers = 1, 1, 2, 1, 1
    if a.rounds < 1:
        ap.error("--rounds 는 1 이상이어야 한다")
    if a.variants < 1:
        ap.error("--variants 는 1 이상이어야 한다")
    if not 0.0 <= a.select_floor <= 1.0:
        ap.error("--select-floor 는 완주율이라 0~1 이어야 한다")
    if a.anchor_coef < 0.0:
        ap.error("--anchor-coef 는 0 이상이어야 한다")
    if a.near_m < 0.0:
        ap.error("--near-m 은 0 이상이어야 한다")
    if a.near_weight <= 0.0:
        # 0 이면 그 표본이 사라지고, 전부 0 이면 가중치 합이 0 이라 손실이 NaN 이다.
        ap.error("--near-weight 는 0 보다 커야 한다")
    if a.anchor_coef > 0.0 and not a.init:
        # 앵커는 "알던 것을 유지하라" 다 — 유지할 '알던 것' 이 없으면 묶을 대상이 없다.
        # 조용히 꺼진 채로 5 라운드를 돌면 성적표에서야 드러난다.
        ap.error("--anchor-coef 를 켜려면 --init 이 있어야 한다 — 앵커가 묶을 참조 정책이 없다")
    # ★ `--init` 오타는 **여기서** 잡는다(dry-run 보다 앞). OMEN 에 띄우기 전 preflight 가
    #   `--dry-run` 한 번이라, 이 검사가 그 뒤에 있으면 경로 오타를 preflight 가 놓친다.
    if a.init and not os.path.exists(a.init):
        ap.error(f"--init 체크포인트가 없다: {a.init}")

    # 앞이 수집, 뒤가 평가다 — 뒤바꾸면 단계 ③ 을 모으라고 시켜 놓고 ①② 만 모은다.
    collect_names, eval_names = resolve_stages(a.stage, a.eval_stage)
    try:
        # `--smoke` 는 액터를 버리는 조각을 쓴다 — 액터 있는 단계와 섞이면 조용히 아무것도
        # 안 시험한다. 일을 시작하기 **전에** 막는다.
        check_smoke(sorted(set(collect_names) | set(eval_names)), a.smoke)
        collect_paths = [stage_path(n) for n in collect_names]
        for _n in eval_names:
            stage_path(_n)          # 평가 단계 이름도 여기서 검사한다
    except ValueError as exc:
        ap.error(str(exc))

    sel_metric = a.select_metric or DEFAULT_SELECT_METRIC[a.select]
    if a.dry_run:
        print("\n".join(_dry_run_lines(collect_names, eval_names, a.smoke, a.variants,
                                       a.seeds, a.rounds)
                        + _preflight_lines(a, sel_metric)))
        return 0
    if not a.out:
        ap.error("--out 이 필요하다")
    a.report = resolve_report(a.report, a.out)

    data_dir = os.path.join(a.out, "data")
    log_path = os.path.join(a.out, "log.jsonl")
    dev = pick_device(a.device)
    eval_seeds = tuple(range(a.eval_seeds))

    if a.report_only:
        if not os.path.exists(log_path):
            ap.error(f"--report-only 인데 {log_path} 가 없다")
        with open(log_path, encoding="utf-8") as f:
            rounds = [json.loads(line) for line in f if line.strip()]
        if not rounds:
            ap.error(f"{log_path} 에 라운드가 없다")
    else:
        if not a.resume:
            existing_shards = glob.glob(os.path.join(data_dir, "*.npz"))
            if os.path.exists(log_path) or existing_shards:
                ap.error(f"`{a.out}` 에 이미 log.jsonl 또는 조각이 있다 — 이어 붙이면 서로 다른 두"
                        " 실행이 섞인다. 지우거나 --resume 을 줘라.")
        os.makedirs(data_dir, exist_ok=True)
        rounds, t0 = [], time.perf_counter()

        # ★★ 앵커의 참조는 `--init` 체크포인트로 **고정**한다 — 루프 **밖**에서 한 번 만든다.
        # 라운드마다 다시 만들면(이전 라운드 정책으로든, 그 라운드 출발 그물로든) 참조가
        # 라운드를 따라 흘러가고, 그러면 "무엇을 유지하라고 묶고 있는가" 가 라운드마다
        # 달라져 계수 하나의 뜻이 사라진다. M4m 이 검증한 구성이 이 고정 참조다.
        # 계수가 0 이면 아예 안 만든다 — 체크포인트를 한 번 더 읽지도, 메모리를 더 쓰지도
        # 않고, `policy_loss` 가 앵커 블록을 통째로 건너뛰어 과거 성적표와 같은 경로다.
        ref = make_reference(round_start_net(a.init, a.smoke, dev)) if a.anchor_coef > 0.0 else None

        for rnd in range(a.rounds):
            beta = BETAS[rnd] if rnd < len(BETAS) else 0.0
            policy_path = os.path.join(a.out, f"policy-r{rnd-1}.pt") if rnd else None
            targets = collect_targets(collect_paths, a.smoke, a.variants)
            names = [name for _p, name in targets]
            jobs = build_jobs(targets, rnd, beta, policy_path, data_dir, a.smoke,
                              a.variants, a.seeds)
            t_collect = time.perf_counter()
            if a.workers > 1:
                with mp.get_context("spawn").Pool(a.workers) as pool:
                    metas = pool.map(_collect_job, jobs)
            else:
                metas = [_collect_job(j) for j in jobs]
            collect_s = time.perf_counter() - t_collect
            stall_start = _stall_start_count(data_dir, rnd, names, a.seeds)

            dataset = load_dir(data_dir)
            # DAgger 는 라운드마다 누적 데이터셋에 예측기를 "다시" 짓는다 — 이전 라운드 그물을 웜스타트로
            # 이어 쓰면 라운드마다 에폭이 쌓여(라운드 7 이면 8 라운드 * 8 에폭 = 64 에폭) 조향 log_std 가
            # 무너지고 가속 머리가 굶는다(수정 라운드 — 측정: 웜스타트 policy-r7 완주율 0%, 새로 지은 그물
            # 8 에폭 완주율 100%, 같은 데이터). 판을 몰 때 쓰는 이전 라운드 체크포인트(policy_path, β 혼합)는
            # 그대로 두고, 학습만 매 라운드 새 그물로 한다.
            torch.manual_seed(a.seed + rnd)   # 안 걸면 초기값이 전역 RNG 상태에 따라 실행마다 달라진다
            net_r = round_start_net(a.init, a.smoke, dev)
            # 손실의 `squash` 를 그물에 맞춘다 — `--init` 이 스쿼시 체크포인트(M4c 재적합 학생)면
            # 라벨을 atanh 로 옮겨 배워야 한다. 안 맞추면 tanh 정책을 clamp 가능도로 학습해
            # **웜스타트가 조용히 망가진다**. `--init` 없는 기본 경로는 둘 다 False 라 무동작이다.
            # 배치 순서 시드 = `--train-seed + rnd`. **덮어쓰지 않고 더한다** — 라운드마다
            # 달라야 하는 것은 그대로 두고(기본 0 이면 `rnd`, 예전과 비트 동일) 세 실행이
            # 서로 다른 배치 순서를 보게 만든다. 이것이 M4l 이 말한 "학습 시드" 다.
            tcfg = squash_aligned(TrainConfig(epochs=a.epochs, seed=a.train_seed + rnd,
                                              anchor_coef=a.anchor_coef, near_m=a.near_m,
                                              near_weight=a.near_weight), net_r)
            train = train_epochs(net_r, dataset, tcfg, device=dev, ref=ref)
            net_r.save(os.path.join(a.out, f"policy-r{rnd}.pt"))

            row = {"round": rnd, "beta": beta, "episodes": len(metas),
                   "collect_goal": sum(1 for m in metas if m["outcome"] == "goal"),
                   "samples": len(dataset), "collect_s": collect_s, "train": train,
                   "stall_start": stall_start,
                   # 설정만 켜고 아무 일도 안 일어나는 꼴(M4e)을 **숫자로** 본다. `anchor` 는
                   # 계수를 안 곱한 KL 이고 `near_frac` 은 가중이 실제로 걸린 비율이다 —
                   # 둘 다 **학습기가 돌려준 값**이지 설정값을 베낀 것이 아니다.
                   "anchor_coef": tcfg.anchor_coef, "anchor": train["anchor"],
                   "near_m": tcfg.near_m, "near_weight": tcfg.near_weight,
                   "near_frac": train["near_frac"],
                   # 실제로 쓴 배치 순서 시드 — 시드 3 개의 로그를 나란히 놓고 읽을 때
                   # 어느 줄이 어느 실행인지 이것으로 가른다.
                   "train_seed": tcfg.seed}
            # 평가는 단계마다 그 단계 이름을 열쇠로 넣는다 — 기본값이면 예전과 같은
            # `"stage1"`·`"stage2"` 가 그대로 나와 옛 로그·`--report-only` 와 호환된다.
            for stage, boards_e in eval_boards(eval_names, a.smoke):
                ev_r = evaluate_policy(net_r, boards_e, seeds=eval_seeds)
                row[stage] = {k: ev_r[k] for k in ("goal_rate", "mean_score", "mean_reward")}
            missing = [s for s in eval_names if s not in row]
            if missing:
                # 열쇠가 빠지면 성적표 칸이 조용히 대시가 된다 — 평가를 안 한 것과 구별이 안 된다.
                raise AssertionError(f"평가 결과에 단계가 빠졌다: {missing}")
            rounds.append(row)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ★★ 채택 — 고른 라운드를 **실제로 내보낸다.** 이 변수 하나가 (1) 아래 상세 평가가 읽는
    # 체크포인트, (2) 목표 판정, (3) 요약의 단계별 숫자, (4) `policy-best.pt` 를 전부 정한다.
    # 예전에는 `best` 를 아래 성적표 자리에서 **계산만** 하고 여기서는 `rounds[-1]` 을 읽었다 —
    # M4k 실측으로 r3(단계 ① 83.3%)를 만들어 놓고 r4(16.7%)를 내보냈다. 고른 라운드와
    # 내보내는 라운드가 갈라질 길을 아예 없앤다.
    chosen = pick_round(rounds, a.select, sel_metric, eval_names, a.select_floor)
    sel_note = select_note(a.select, sel_metric, a.select_floor, rounds, eval_names)
    chosen_ckpt = os.path.join(a.out, f"policy-r{chosen['round']}.pt")
    # 체크포인트를 다시 읽어(학습 도중의 net 객체에 기대지 않고) 상세 평가를 낸다 —
    # --report-only 에서도 그대로 쓸 수 있고, 정상 실행에서도 저장한 체크포인트가 로그의 숫자와
    # 같은 걸 낸다는 확인이 겸사겸사 된다.
    net = DrivePolicy.load(chosen_ckpt, device=dev)
    # 뒤 단계(안 쓴 변종으로 다시 재는 평가)가 **경로 하나만** 알면 되게 복사해 둔다.
    # 복사다 — 심링크는 rsync·zip·scp 를 지나며 깨지고, 깨진 심링크는 조용히 없는 파일이 된다.
    best_path = os.path.join(a.out, BEST_CKPT_NAME)
    shutil.copyfile(chosen_ckpt, best_path)
    pairs = eval_boards(eval_names, a.smoke)
    assert [n for n, _bs in pairs] == eval_names
    evs = [evaluate_policy(net, bs, seeds=eval_seeds) for _n, bs in pairs]
    # 선생님 기준도 학생과 같은 평가 시드로 잰다 — "선생님보다 10점 이내" 가 같은 조건끼리의 비교가 되도록.
    teachers = [evaluate_teacher(bs, seeds=eval_seeds) for _n, bs in pairs]
    # 목표 판정은 **첫 평가 단계**로 한다(기본값이면 예전과 같은 단계 ①).
    primary = eval_names[0]
    ev1, teacher1 = evs[0], teachers[0]
    ok = (chosen[primary]["goal_rate"] >= 0.9
          and chosen[primary]["mean_score"] >= teacher1["mean_score"] - 10.0)

    if a.report_only:
        summary_line = (f"- 라운드 {len(rounds)} · 라운드마다 판 {rounds[0]['episodes']}개 · 평가 시드"
                        f" {a.eval_seeds}개 · 이 성적표는 라운드를 다시 안 돌리고 `{a.out}` 의 기존"
                        f" 로그·체크포인트로 다시 만들었다(수집+학습 합계 "
                        f"{sum(r['collect_s'] + r['train']['seconds'] for r in rounds):.0f}초 — "
                        f"원래 실행의 전체 시간은 로그에 없다)")
    else:
        summary_line = (f"- 라운드 {len(rounds)} · 라운드마다 판 {rounds[0]['episodes']}개 · 평가 시드"
                        f" {a.eval_seeds}개 · 그물 초기화 시드 {a.seed} · 전체"
                        f" {time.perf_counter() - t0:.0f}초")

    collect_label = ", ".join(f"`{os.path.relpath(p, REPO)}`" for p in collect_paths)
    only_eval = [s for s in eval_names if s not in collect_names]
    zero_shot_line = (f"- 수집 루프는 {collect_label} 만 쓴다(판 변종 {a.variants}벌)."
                      + (f" {', '.join(stage_label(s) for s in only_eval)} 평가는 분포 밖(zero-shot) 시험이지 학습된"
                         " 능력이 아니다." if only_eval else
                         " 평가 단계가 모두 수집 단계 안에 있다."))

    lines = [
        "# M3 성적표 — DAgger 모방학습", "",
        f"- 날짜 {datetime.date.today().isoformat()} · 머신 `{platform.node()}` · 장치 `{dev}` "
        f"· 규칙 스택 `{rs.commit()[:7]}`",
        summary_line,
        f"- 목표: {stage_label(primary)} 완주율 ≥ 90%, 점수가 선생님보다 10점 넘게 낮지 않을 것 → "
        f"**{'달성' if ok else '미달'}**",
        "- 선생님 기준: " + " · ".join(
            f"{stage_label(s)} 완주율 {t['goal_rate']*100:.0f}% 점수 {t['mean_score']:.1f}"
            for s, t in zip(eval_names, teachers)),
    ]
    if len(teachers) > 1:
        lines.append(f"- {_teacher_caveat(teachers[1], stage_label(eval_names[1]) + '(신호 주기)')}")
    lines.append(
        f"- 서로 다른 판 수(시드 {a.eval_seeds}개 중 실제로 달랐던 것, 마지막 라운드) — " + " · ".join(
            f"{stage_label(s)}: 학생 {_distinct_episodes(e)}/{len(e['episodes'])}개, 선생님"
            f" {_distinct_episodes(t)}/{len(t['episodes'])}개"
            for s, e, t in zip(eval_names, evs, teachers))
        + " (액터·신호가 고정이면 시드가 달라도 같은 판이 되기 쉽다).")
    # `--init` 이 없으면 줄 자체를 안 넣는다 — 기본 실행의 성적표 문구가 예전과 한 글자도
    # 안 달라져야 커밋된 M3 성적표를 그대로 다시 만들 수 있다.
    if a.init:
        lines.append(f"- 출발점: `{a.init}` 에서 웜스타트 — 라운드마다 **이 체크포인트**에서 다시"
                     " 시작한다(이전 라운드 그물을 이어 쓰지 않아 에폭이 안 쌓인다). 이 성적표의"
                     " 완주율은 '처음부터 배운 결과' 가 아니라 '그 학생을 이 단계에 더 얹은 결과' 다.")
    # 기본값(0)이면 줄 자체를 안 넣는다 — 그때는 예전과 같은 `rnd` 라 밝힐 것이 없고,
    # 성적표 문구가 한 글자도 안 달라져야 커밋된 M3 성적표를 그대로 다시 만든다.
    if a.train_seed:
        lines.append(f"- 배치 순서 시드: 라운드 rnd 는 `{a.train_seed} + rnd` — M4l 실측으로 같은"
                     " 데이터·같은 설정이라도 **배치 순서만 바꾸면** 단계 ① 완주율이 16.7% ↔"
                     " 66.7% 로 흔들린다. 이 성적표는 **시드 하나**의 결과이지 그 설정의"
                     " 결과가 아니다.")
    lines += [
        zero_shot_line, "",
        "| 라운드 | β | 수집 판(완주) | 누적 표본 | 손실 | 정지-출발 표본¹ | "
        + " | ".join(f"{stage_label(s)} 완주율 | {stage_label(s)} 점수" for s in eval_names) + " |",
        "|---:|---:|---|---:|---:|---:|" + "---:|---:|" * len(eval_names),
    ]
    for r in rounds:
        cells = "".join(f" {_cell(r, s, 'goal_rate', '.0f')}% |"
                        f" {_cell(r, s, 'mean_score', '.1f')} |" for s in eval_names)
        lines.append(f"| {r['round']} | {r['beta']} | {r['episodes']}({r['collect_goal']}) | "
                     f"{r['samples']} | {r['train']['loss']:.3f} | "
                     f"{r.get('stall_start', '—')} |" + cells)
    lines += ["", "¹ 그 라운드에 새로 모은 판은 '이번 라운드에 새로 나온 그물'이 아니라 그 라운드가"
              " 시작할 때 있던 이전 라운드 체크포인트(라운드 0 은 선생님 전용)가 몰았다."]
    # 앵커·회피 가중을 켠 실행에만 붙는 절 — 기본 실행의 성적표 문구는 예전 그대로다.
    # `anchor`(계수 안 곱한 KL)와 `near_frac`(가중이 걸린 비율)이 **0 이면 설정만 켜고
    # 아무 일도 안 일어난 것**이다(M4e 의 `--imitation-floor 0.3` 이 그랬다).
    if a.anchor_coef > 0.0 or a.near_m > 0.0:
        lines += ["", "### 앵커·회피 가중이 실제로 걸렸는가(라운드별)", "",
                  f"- 참조는 `{a.init}` 로 **고정**이다 — 라운드마다 안 바꾼다(라운드를 따라"
                  " 흘러가면 계수 하나의 뜻이 라운드마다 달라진다).", "",
                  "| 라운드 | anchor_coef | 앵커 KL | near_m | near_weight | 가중 비율 |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r in rounds:
            # 옛 로그(`--report-only`)에는 이 열쇠들이 없다 — 그때는 대시로 둔다.
            kl, frac = r.get("anchor"), r.get("near_frac")
            lines.append(
                f"| {r['round']} | {r.get('anchor_coef', '—')} |"
                f" {'—' if kl is None else format(kl, '.3f')} | {r.get('near_m', '—')} |"
                f" {r.get('near_weight', '—')} |"
                f" {'—' if frac is None else format(frac * 100, '.1f') + '%'} |")

    # 가장 좋았던 라운드. **`--select last`(기본)이면 기준이 `primary` 라 예전 식과 글자
    # 하나까지 같다** — 커밋된 M3 성적표가 이 줄을 들고 있다(`가장 좋았던 라운드: 4`).
    # `--select best` 면 이 줄이 곧 채택된 라운드이고, 바로 아래 줄이 기준을 밝힌다.
    # 동률이면 더 뒤 라운드(round 값이 더 큰 쪽)를 고른다 — `pick_round` 안의 규칙이다.
    last = rounds[-1]          # 아래 '정지-출발' 비교는 **마지막 라운드**를 말한다(채택과 별개)
    best = pick_round(rounds, "best", sel_metric, eval_names, a.select_floor)
    best_ok = (best[primary]["goal_rate"] >= 0.9
              and best[primary]["mean_score"] >= teacher1["mean_score"] - 10.0)
    best_ckpt = os.path.join(a.out, f"policy-r{best['round']}.pt")
    lines += ["", f"- 가장 좋았던 라운드: {best['round']}(β={best['beta']}) — {stage_label(primary)} 완주율 "
              f"{best[primary]['goal_rate']*100:.0f}% 점수 {best[primary]['mean_score']:.1f} → 목표 두 "
              f"조건 **{'달성' if best_ok else '미달'}** · 체크포인트 `{best_ckpt}`"]
    # 기본 실행(`--select last`)에는 이 줄이 **없다** — 성적표 문구가 예전과 한 글자도 안
    # 달라져야 커밋된 M3 성적표를 `--report-only` 로 그대로 다시 만들 수 있다.
    if a.select != "last":
        lines += [f"- **채택: 라운드 {chosen['round']}** (`--select {a.select}`) — 기준 {sel_note}"
                  f" · 이 성적표의 완주율·위반표·요약은 전부 **이 라운드**의 것이다"
                  f" · 체크포인트 `{best_path}`(= `{chosen_ckpt}` 복사본)"]
    # 라운드가 하나뿐이면(스모크) "베스트 대 마지막" 비교가 자기 자신과의 비교라 공허하니 건너뛴다.
    # stall_start 필드가 없는 옛 로그(--report-only)도 조용히 건너뛴다.
    if len(rounds) > 1 and last.get("stall_start") is not None:
        if best["round"] == last["round"]:
            baseline, baseline_label = rounds[0], f"라운드 {rounds[0]['round']}(첫 라운드)"
        else:
            baseline, baseline_label = best, f"라운드 {best['round']}(가장 좋음)"
        if baseline.get("stall_start") is not None:
            trend = ("늘었다" if last["stall_start"] > baseline["stall_start"]
                     else "줄었다" if last["stall_start"] < baseline["stall_start"] else "변하지 않았다")
            lines += [f"- {baseline_label}의 새로 모은 판에서 '정지 상태인데 선생님은 출발하라고 한' 표본이"
                      f" {baseline['stall_start']}건, 라운드 {last['round']}(마지막)에서는 "
                      f"{last['stall_start']}건이다 — {trend}."]
    lines += _final_outcome_lines(ev1, stage_label(primary))
    for stage, e, t in zip(eval_names, evs, teachers):
        lines += _violation_lines(stage_label(stage), e, t)
    lines += ["", "학생은 결정적 행동으로 혼자 몬다. 점수는 환경이 낸 구간 점수의 평균이고,",
              "환경의 감점표가 대회 채점기와 같다는 것은 M2a·M2b 성적표에 있다.",
              f"산출물(데이터·체크포인트·로그)은 `{a.out}` 아래에 있고 레포에는 넣지 않는다.", "",
              "log_std 바닥은 clamp 로 구현했다 — 바닥에 붙은 파라미터는 그래디언트가 0 이라 스스로 다시",
              "못 올라온다. 이 체크포인트로 M4 의 PPO 를 웜스타트하면 조향 탐색 폭(log_std)이 얼어붙은",
              "채로 시작한다 — softplus 매개변수화나, 매 최적화 스텝 뒤 in-place clamp 로 바꾸는 편이 낫다.",
              "", "목표 문턱의 점수 쪽(선생님보다 10점 이내)은 사실상 못 떨어진다 — 멈춰서 아무것도",
              "못 끝낸 차도 대부분의 항목을 안 어겨서 97~100점이 나온다(이 성적표의 정지-출발 표본이 그",
              "증거다). M4 에서 이 점수를 커리큘럼 승급 문에 쓰려면 완주 여부로 조건을 걸어야 한다",
              "(스펙 §6.4).",
              "", "`DrivePolicy.load` 는 state_dict 를 엄격하게(strict) 맞춘다 — PPO 가 가치 머리를",
              "더한 체크포인트를 이 함수로 불러오려면 `strict=False` 로 풀거나 머리만 따로 합쳐야 한다.",
              "", "`act(deterministic=False)` 는 표본을 [-1,1] 로 자르기만 하고 확률밀도는 그대로",
              "둔다 — PPO 는 자른 값이 아니라 원래 표본과 그 로그확률을 저장하고, 환경에 넣을 때만",
              "잘라야 한다(안 그러면 경계에서 로그확률이 실제 분포와 안 맞는다)."]
    lines += _history_lines(a.history, a.history_note, eval_names)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    summary = {"rounds": len(rounds), "samples": chosen["samples"],
               "collect_stages": collect_names, "eval_stages": eval_names,
               "variants": a.variants, "init": a.init,
               "teacher1": {k: teacher1[k] for k in ("goal_rate", "mean_score")},
               "target_met": ok, "report": a.report,
               # 선택이 **암묵적이면 안 된다** — 무엇을 어떤 기준으로 골랐는지 요약이 들고 있다.
               "select": a.select, "select_metric": sel_metric,
               "select_floor": a.select_floor, "select_note": sel_note,
               "selected_round": chosen["round"], "best_ckpt": best_path,
               # 개입이 켜졌는지는 요약만 봐도 알아야 한다(라운드별 실측은 log.jsonl 에 있다).
               "anchor_coef": a.anchor_coef, "near_m": a.near_m, "near_weight": a.near_weight,
               "train_seed": a.train_seed}
    for stage in eval_names:
        summary[stage] = chosen[stage]
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

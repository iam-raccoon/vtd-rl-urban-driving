"""M4a 완료 증거 — PPO 학습 곡선 표 + M3/선생님 대비 표 + 목표 판정 + 여러 실행 비교.

    env -u PYTHONPATH .venv/bin/python scripts/report_m4a.py \
        --run runs/omen/2026-09-21-ppo --compare runs/omen/2026-09-22-ppo-entropy \
        --m3 runs/lab-main/2026-09-17-dagger-fix2/policy-r4.pt \
        --out docs/reports/m4a-ppo.md --eval-seeds 3 --notes docs/reports/m4a-ppo-notes.md

`--skip-eval` 은 정책을 실제로 몰아 보는 절(비교 표·항목별 위반 표·목표 판정)을 건너뛰고
`log.jsonl` 만으로 학습 곡선 표·실행 비교 표만 채운 뼈대를 만든다(짧은 스모크 실행을 빠르게
확인하는 용도, 테스트가 이걸로 돈다).

M4a 학생 체크포인트(`ac-best.pt`)는 가치 머리가 붙은 `ActorCritic` 이라 `ActorCritic.load(path)`
로 읽고 `.policy` 를 꺼낸다 — `DrivePolicy.load` 로는 못 읽는다(cfg 모양이 다르다). M3 체크포인트는
순수 정책이라 `DrivePolicy.load` 가 맞다.

평가는 단계(커리큘럼 파일)마다 따로 부른다 — `stage1.json`·`stage2.json` 은 판 이름을 그대로
공유하고 `signals` 만 다르므로, 합쳐 넘기면 `VtdDriveEnv` 의 세계 캐시가 죽는다(`train_ppo.py` 의
같은 주의사항 참고). 판은 `load_curriculum` 으로 단계별로 직접 읽는다.

`--compare` 로 준 다른 실행들은 요약 비교 표(실행마다 한 행)와 그 실행 자신의 학습 곡선 표에만
나온다 — M3/선생님 대비 표·항목별 위반 표·목표 네 줄 판정은 `--run`(이번에 채점하는 실행) 하나만
싣는다(그게 "M4a 가 끝났는가"를 묻는 절이라 여러 실행을 합쳐 물을 수 없다).
"""
import argparse
import dataclasses
import datetime
import glob
import json
import os
import platform
import re
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from vtd_rl import rule_stack as rs  # noqa: E402
from vtd_rl.env.drive_env import EnvConfig  # noqa: E402
from vtd_rl.env.reward import RewardConfig  # noqa: E402
from vtd_rl.eval.verdict import completed_only, judge, major_total  # noqa: E402
from vtd_rl.policy import device as pick_device  # noqa: E402
from vtd_rl.policy.evaluate import evaluate_policy, evaluate_teacher, violation_counts  # noqa: E402
from vtd_rl.policy.net import DrivePolicy  # noqa: E402
from vtd_rl.rl.actor_critic import ActorCritic  # noqa: E402
from vtd_rl.rl.ppo import PPOConfig  # noqa: E402
from vtd_rl.rl.reward_cfg import reward_config_from_hparams  # noqa: E402
from vtd_rl.world.board import load_curriculum  # noqa: E402

STAGE1_LABEL, STAGE2_LABEL = "stage1", "stage2"

# 진단 곡선 표(신규, Task 6) — 롤아웃 줄이 3M 실행에 약 391줄이라 마크다운 표에 다 못 넣는다.
# 이 개수로 표본을 뽑는다(간격은 `max(1, len(rows)//DIAGNOSTIC_SAMPLE_TARGET)`).
DIAGNOSTIC_SAMPLE_TARGET = 20

# `train_ppo.py` 의 `hparams`(=`vars(argparse 결과)`) 에는 `--envs`·`--seed`·`--out` 처럼 실행마다
# 자연히 다른 값도 섞여 있다. `--envs` 는 기본값 자체가 `os.cpu_count() - 2` 라 이 스크립트를 돌리는
# 머신에서 다시 계산하면 실행 당시(OMEN)와 달라 보여 "바뀐 값"으로 잘못 잡힌다. 그래서 비교는
# `PPOConfig` 필드와 이름이 같은, 실제로 CLI 로 연 하이퍼파라미터로만 좁힌다.
# `imitation_sigma`(`--imitation-sigma`) 는 M4b 본 실험이 실제로 시험한 처치(learn vs detach)
# 인데 여기 빠져 있었다 — 그래서 `sigma-detach-s0` 실행이 "(기본값)" 으로 찍혀, 하이퍼파라미터가
# "같다"고 적힌 두 행이 항목⑦ 68 vs 39 로 갈린 걸 읽는 사람이 시드 잡음으로 오독했다
# (2026-09-22 최종 리뷰 Critical). 반드시 여기 추가해야 한다.
#
# 보상 필드(`comfort_steer` 등)도 같은 이유로 여기 있어야 한다 — 없으면 두 실행이 서로 다른
# 보상 설정으로 학습됐어도 이 열이 "(기본값)" 이라 찍어 정확히 그 차이를 숨긴다(M4d Task 1
# 재리뷰 Important #3). 손으로 나열하지 않고 `dataclasses.fields(RewardConfig)` 에서 뽑는다 —
# `vtd_rl/rl/reward_cfg.py` 와 같은 방식이라, 다음 작업이 새 필드(`violation_mode`)를 더해도
# 자동으로 실린다.
CLI_HPARAM_KEYS = (("lr", "entropy_coef", "target_kl", "imitation_half_life", "imitation_sigma")
                   + tuple(f.name for f in dataclasses.fields(RewardConfig)))


def _circled(n: int) -> str:
    return chr(0x2460 + n - 1)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _fmt_score(x: float) -> str:
    return f"{x:.1f}"


def _distinct_episodes(ev: dict) -> int:
    """서로 다른(코스, 결과, 걸음수) 조합 수 — `run_dagger.py._distinct_episodes` 와 같은 정의

    (정책이 결정적이면 액터·신호가 고정인 판은 시드를 바꿔도 같은 판이 나온다).
    """
    return len({(e.board, e.outcome, e.steps) for e in ev["episodes"]})


def _read_rows(run_dir: str) -> list:
    """`<run_dir>/log.jsonl` 을 읽는다 — 없으면(학습이 아직 시작 전이거나 첫 평가 전이면) 빈 리스트.

    `--compare` 로 진행 중인 실행을 넘길 수 있어서(2026-09-22 실측 — 두 번째 OMEN 실행이 도는 동안
    성적표를 다시 만들어야 했다) 파일이 없는 것 자체는 에러가 아니다. `--run`(채점 대상)만 main() 에서
    따로 필수로 확인한다.
    """
    log_path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _last_eval_row(rows: list):
    """`rows` 중 평가 줄(`"stages"` 키가 있는 줄) 마지막 것 — 없으면 `None`.

    Task 4 가 로그 해상도를 올려 대부분의 줄(롤아웃 줄)엔 `stages`·`outcome_*` 가 없다 —
    `rows[-1]` 을 그냥 쓰면 그게 롤아웃 줄일 때 `KeyError: 'stages'` 로 죽는다(이 작업이 고치는
    바로 그 버그). 평가 줄만 걸러야 한다(`"stages" in r`).
    """
    evals = [r for r in rows if "stages" in r]
    return evals[-1] if evals else None


def _curricula_from_log(rows: list) -> list:
    """`log.jsonl` 의 `stages` 키에서 커리큘럼 파일 경로를 되찾는다.

    `train_ppo.py` 는 `--curricula` 인자 자체를 파일로 남기지 않는다 — `_stage_boards()` 가
    라벨(요약 딕셔너리의 키)을 커리큘럼 파일 이름(확장자 없이)으로 쓰는 규칙을 거꾸로 쓴다.
    기본 커리큘럼(`stage1.json`, `stage2.json`) 밖의 이름을 쓴 실행이면 그 파일이 실제로
    `curricula/` 아래 그 이름으로 있어야 한다. 평가 줄이 아직 하나도 없으면(학습이 첫 평가
    전이면) 빈 리스트를 낸다.
    """
    last_eval = _last_eval_row(rows)
    if last_eval is None:
        return []
    return [(label, os.path.join(REPO, "curricula", f"{label}.json")) for label in last_eval["stages"]]


def _hparam_diff(hparams) -> str:
    """`CLI_HPARAM_KEYS` 만 기본값과 비교해 바뀐 것만 "key=value" 로 나열한다.

    `CLI_HPARAM_KEYS` 가 `PPOConfig` 필드(`lr` 등)와 `RewardConfig` 필드(`comfort_steer` 등)를
    섞어 담으므로, 비교 기준 `defaults` 도 둘 다 합쳐야 한다 — `PPOConfig()` 만 쓰면 보상 필드
    키에서 `KeyError` 가 난다(두 dataclass 필드 이름이 겹치지 않음을 계획 단계에서 확인했다).
    """
    if not hparams:
        return "기록 안 됨(이 실행은 hparams 로깅 이전 버전으로 돌았다)"
    defaults = {**dataclasses.asdict(PPOConfig()), **dataclasses.asdict(RewardConfig())}
    # 기본값이 빈 튜플인 필드(`item_scale`)는 안 준 실행이 hparams 에 `None`(JSON `null`)으로 남는다 —
    # `None != ()` 라 그대로 비교하면 기본 실행마다 "item_scale=None" 이 바뀐 것처럼 찍힌다.
    def _norm(k):
        v = hparams[k]
        return () if defaults[k] == () and not v else v

    diffs = [f"{k}={hparams[k]}" for k in CLI_HPARAM_KEYS if k in hparams and _norm(k) != defaults[k]]
    return ", ".join(diffs) if diffs else "(기본값)"


def _detect_best_step(run_dir: str):
    """`ac-best.pt` 가 `ac-<step>.pt` 중 어느 것과 같은 체크포인트인지 state_dict 를 비교해 찾는다.

    학습이 끝난 뒤 뽑은 `best_step` 은 표준출력에 찍는 요약 JSON 에만 있고 파일 어디에도 안 남는다
    (`train_ppo.py` 의 `summary` 는 stdout 전용). 그 숫자를 손으로 옮겨 적지 않으려면 체크포인트
    자체를 비교해 되찾아야 한다 — CPU 로만 읽는다(환경 시뮬레이션이 없어 device 를 안 가린다).
    """
    best_path = os.path.join(run_dir, "ac-best.pt")
    if not os.path.exists(best_path):
        return None
    best_state = ActorCritic.load(best_path, device="cpu").state_dict()
    for path in sorted(glob.glob(os.path.join(run_dir, "ac-*.pt"))):
        m = re.fullmatch(r"ac-(\d+)\.pt", os.path.basename(path))
        if not m:
            continue
        cand_state = ActorCritic.load(path, device="cpu").state_dict()
        if set(cand_state) == set(best_state) and all(
                torch.equal(cand_state[k], best_state[k]) for k in best_state):
            return int(m.group(1))
    return None


def _train_seed_line(hparams, train_seed_arg) -> str:
    """학습 시드 — 최우선 출처는 `hparams["seed"]`(`train_ppo.py` 가 실제로 `torch.manual_seed`

    에 준 값)다. 이게 있는 실행(hparams 로깅 이후)에서 "기록 안 됨"이라 말하는 건 산출물에
    이미 있는 값을 없다고 말하고 사람에게 `--train-seed` 로 다시 손으로 넣으라고 시키는
    것과 같다(2026-09-22 리뷰) — README 의 "숫자를 손으로 옮겨 적지 않는다" 원칙에 정면으로
    어긋난다. `hparams` 자체가 없는 실행(로깅 이전, 예: `runs/omen/2026-09-21-ppo`)에서만
    `--train-seed` 폴백을 쓴다. 어느 경로로 얻었는지 항상 괄호로 밝힌다.
    """
    if hparams and hparams.get("seed") is not None:
        return f"{hparams['seed']}(log.jsonl 의 hparams 에서 읽음)"
    if train_seed_arg is not None:
        return (f"{train_seed_arg}(직접 입력값 — 이 실행은 hparams 로깅 이전이라 log.jsonl 에 "
                "시드가 없다)")
    return "기록 안 됨(log.jsonl 에 hparams 가 없고 --train-seed 도 안 줬다)"


def _best_step_caveat(rows: list, best_step) -> str:
    """`best_step` 이 전체 스텝의 절반 미만이면, 그 시점의 모방 계수와 함께 사실을 그대로 적는다.

    이건 해석이 아니라 사실이다 — 이게 없으면 완주율·점수 줄의 "달성"이 PPO 자체의 성과처럼
    읽힌다(2026-09-22 지시: 507k/3M, 모방 계수 0.839 인 체크포인트가 뽑혔다).
    """
    if best_step is None or not rows:
        return ""
    total_steps = rows[-1]["step"]
    if total_steps <= 0 or best_step >= total_steps / 2:
        return ""
    match = next((r for r in rows if r.get("step") == best_step), None)
    coef = f"{match['imitation_coef']:.3f}" if match and "imitation_coef" in match else "알 수 없음"
    pct = best_step / total_steps * 100
    return (f"선정된 체크포인트가 {best_step} 스텝(전체 {total_steps} 의 {pct:.0f}%) 지점이고, 그때"
            f" 모방 계수가 {coef} 였다 — 위 달성이 PPO 자체의 성과라기보다 M3 워밍스타트에 가까울"
            " 수 있다는 뜻이다.")


def _curve_table(rows: list, stage_labels: list) -> list:
    """평균 보상(`mean_reward`) 도 완주율·점수와 나란히 단계마다 한 열 — 계획서가 명시한 열이고,

    관찰 노트("단계① 평균 보상 102.5 → 22.5")가 바로 이 숫자를 인용하므로 성적표에서도 그
    숫자를 확인할 수 있어야 한다(2026-09-22 리뷰). `mean_reward` 는 `train_ppo.py` 의
    `PUBLIC_KEYS` 에 있어 `log.jsonl` 각 줄의 `stages.<라벨>` 에 이미 들어 있다.

    Task 4 가 로그 해상도를 올린 뒤로 `rows` 대부분은 `stages` 가 없는 롤아웃 줄이다(3M
    실행에 약 391줄 중 평가 줄은 약 6줄뿐) — `"stages" in r` 로 평가 줄만 골라야 한다. 안
    그러면 `r["stages"][label]` 가 롤아웃 줄에서 `KeyError` 로 죽는다.
    """
    eval_rows = [r for r in rows if "stages" in r]
    header = ["스텝", "iter"]
    for i, _label in enumerate(stage_labels, start=1):
        header += [f"단계{_circled(i)} 완주율", f"단계{_circled(i)} 점수(완주 0점 기준)",
                   f"단계{_circled(i)} 평균 보상"]
    header += ["explained_variance", "log_std", "모방 계수", "approx_kl"]
    lines = ["", f"(평가 줄만 — 전체 로그 {len(rows)}줄 중 {len(eval_rows)}줄)",
             "| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    for r in eval_rows:
        cells = [str(r["step"]), str(r["iter"])]
        for label in stage_labels:
            st = r["stages"][label]
            cells += [_fmt_pct(st["goal_rate"]), _fmt_score(st["mean_score"]), _fmt_score(st["mean_reward"])]
        ev = f"{r['explained_variance']:.3f}" if r.get("explained_variance") is not None else "—"
        log_std = ", ".join(f"{x:.3f}" for x in r["log_std"])
        cells += [ev, log_std, f"{r['imitation_coef']:.3f}", f"{r['approx_kl']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _diagnostic_table(rows: list) -> list:
    """진단 곡선 표(신규, Task 6) — 롤아웃 줄(대부분) 을 일정 간격으로 표본해 학습 곡선 표(평가

    줄만, 3M 실행에 약 6줄) 가 못 보는 붕괴 구간을 보여준다. M4a 는 6줄로만 봐서 1M~1.5M 구간을
    점 2개로만 봤다 — 이 표가 M4b 의 존재 이유다.

    `drift_log_std` 가 특히 중요하다 — 전체망 `drift_rel` 은 trunk 수만 파라미터에 희석돼
    log_std 움직임을 10배 넘게 묻는다(실측: log_std +1.5 일 때 drift_log_std=2.121 vs
    drift_rel=0.143). `rollout_return_mean` 은 그 구간에 끝난 판이 없으면 `None` 일 수 있다
    (`—` 로 찍는다). `outcome_*` 는 0인 사유는 키 자체가 없어 이 표엔 안 싣는다(값싼 진단
    용도로 매 줄 있는 계측만 고른다).
    """
    if not rows:
        return ["", "### 진단 곡선(전 롤아웃 표본)", "", "로그가 없다."]
    step = max(1, len(rows) // DIAGNOSTIC_SAMPLE_TARGET)
    sampled = rows[::step]
    header = ["step", "rollout_return_mean", "rollout_return_n", "drift_log_std", "drift_rel",
              "entropy", "approx_kl", "imitation_coef"]

    def _num(r, key, fmt):
        v = r.get(key)
        return fmt.format(v) if v is not None else "—"

    lines = ["", f"### 진단 곡선(롤아웃·평가 통틀어 {len(rows)}줄 중 {step}줄 간격 표본"
                f" — {len(sampled)}줄)", "",
             "| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    for r in sampled:
        cells = [str(r["step"]),
                 _num(r, "rollout_return_mean", "{:.1f}"),
                 str(r.get("rollout_return_n", "—")),
                 _num(r, "drift_log_std", "{:.3f}"),
                 _num(r, "drift_rel", "{:.3f}"),
                 _num(r, "entropy", "{:.4f}"),
                 _num(r, "approx_kl", "{:.4f}"),
                 _num(r, "imitation_coef", "{:.3f}")]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _summary_table(run_dirs: list, rows_by_run: dict, best_steps: dict, extra_evs: dict,
                   skip_eval: bool) -> list:
    """실행마다 한 행 — 학습 중 주기 평가(시드 1개, log.jsonl)를 기본으로 쓰고, `extra_evs` 에

    그 실행의 라이브 평가(`--skip-eval` 이 아닐 때만 채워진다)가 있으면 그 값으로 덮는다. 항목⑦·②
    중대 건수는 구간-슬롯(sheet) 정보가 필요해 라이브 평가 없이는 절대 못 낸다 — 그때는 "—".

    `stages`(완주율·점수)는 평가 줄에만 있어 `_last_eval_row` 로 찾는다. `log_std`·
    `explained_variance`·`hparams` 는 롤아웃 줄에도 있으므로(값싼 진단은 매 줄에 실린다)
    절대적으로 마지막 줄(`rows[-1]`)에서 읽는 게 더 최신값이다 — 둘을 갈라 쓴다.

    항목⑦·② 중대 건수는 `completed_only` 를 거쳐야 한다(I12, 2026-09-22 최종 리뷰) — 안 그러면
    이 표만 조기 종료 판까지 섞어 세어, 같은 성적표 안의 항목별 표(`_violation_table_lines`,
    이미 `completed_only` 를 쓴다)와 같은 항목이 다른 숫자로 찍힐 수 있다.
    """
    header = ["실행", "바뀐 하이퍼파라미터", "best_step", "단계① 완주율", "단계① 점수",
              "단계② 완주율", "단계② 점수", "항목⑦ 중대", "항목② 중대", "log_std", "EV"]
    lines = ["", "## 실행 비교", "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for rd in run_dirs:
        rows = rows_by_run.get(rd, [])
        name = os.path.basename(os.path.normpath(rd))
        best_step = best_steps.get(rd)
        raw_last = rows[-1] if rows else None
        eval_last = _last_eval_row(rows)
        hp = (_hparam_diff(raw_last.get("hparams")) if raw_last
              else "로그 없음(학습 진행 중이거나 시작 전)")
        stage_vals = dict(eval_last["stages"]) if eval_last else {}
        item7 = item2 = "—"
        live = extra_evs.get(rd)
        if live:
            stage_vals.update(live)
            if STAGE2_LABEL in live:
                item7 = violation_counts(completed_only(live[STAGE2_LABEL])).get(
                    7, {"major": 0})["major"]
            if STAGE1_LABEL in live:
                item2 = violation_counts(completed_only(live[STAGE1_LABEL])).get(
                    2, {"major": 0})["major"]
        s1, s2 = stage_vals.get(STAGE1_LABEL), stage_vals.get(STAGE2_LABEL)
        log_std = ", ".join(f"{x:.3f}" for x in raw_last["log_std"]) if raw_last else "—"
        ev_val = (f"{raw_last['explained_variance']:.3f}"
                  if raw_last and raw_last.get("explained_variance") is not None else "—")
        row = [name, hp, str(best_step) if best_step is not None else "미완료(ac-best.pt 없음)",
               _fmt_pct(s1["goal_rate"]) if s1 else "—", _fmt_score(s1["mean_score"]) if s1 else "—",
               _fmt_pct(s2["goal_rate"]) if s2 else "—", _fmt_score(s2["mean_score"]) if s2 else "—",
               str(item7), str(item2), log_std, ev_val]
        lines.append("| " + " | ".join(row) + " |")
    if skip_eval:
        lines += ["", "(`--skip-eval` 이라 항목⑦·② 열은 log.jsonl 만으로 못 채워 `—` 다.)"]
    return lines


def _comparison_lines(stage_labels: list, m3_ev: dict, m4a_ev: dict, teacher_ev: dict, eval_seeds: int,
                      student_label: str = "학생") -> list:
    """I8(2026-09-22 최종 리뷰): 이 절 제목·표 헤더·"서로 다른 판 수" 줄이 전부 `"M4a"` 로

    하드코딩돼 있어 M4b(또는 그 뒤) 성적표에도 "M4a 학생" 이 찍혔다. `student_label` 을 받아
    채점 대상 실행을 가리키는 라벨을 호출부(`--student-label`)가 정하게 한다.
    """
    lines = ["", f"## 비교 — M3 학생 · {student_label} · 선생님", "",
             f"| 단계 | 지표 | M3 학생 | {student_label} | 선생님 |", "|---|---|---:|---:|---:|"]
    for i, label in enumerate(stage_labels, start=1):
        tag = f"단계{_circled(i)}"
        lines.append(f"| {tag} | 완주율 | {_fmt_pct(m3_ev[label]['goal_rate'])} | "
                     f"{_fmt_pct(m4a_ev[label]['goal_rate'])} | {_fmt_pct(teacher_ev[label]['goal_rate'])} |")
        lines.append(f"| {tag} | 점수(완주 0점 기준) | {_fmt_score(m3_ev[label]['mean_score'])} | "
                     f"{_fmt_score(m4a_ev[label]['mean_score'])} | {_fmt_score(teacher_ev[label]['mean_score'])} |")

    lines += ["", f"### 서로 다른 판 수(시드 {eval_seeds}개 중 실제로 달랐던 것)", ""]
    for i, label in enumerate(stage_labels, start=1):
        n = len(m4a_ev[label]["episodes"])
        note = ("(액터·신호가 고정이라 시드가 달라도 같은 판이 되기 쉽다)" if label == STAGE1_LABEL
                else "(신호 주기 위상이 시드마다 달라 실제로 다른 판이 나올 수 있다)")
        lines.append(f"- 단계{_circled(i)}: M3 {_distinct_episodes(m3_ev[label])}/{n}, "
                     f"{student_label} {_distinct_episodes(m4a_ev[label])}/{n}, "
                     f"선생님 {_distinct_episodes(teacher_ev[label])}/{n} {note}")
    return lines


def _violation_table_lines(stage_labels: list, m3_ev: dict, m4a_ev: dict, teacher_ev: dict,
                           student_label: str = "학생") -> list:
    """항목별 표 — 마지막에 **중대 합계** 행을 더한다(M3·학생·선생님 각각).

    항목별 행은 진단용이라(어느 항목이 문제인지 보려는 것) 판을 전부 쓰지만, 합계 행은
    `judge()` 의 목표 4번(전 항목 중대 위반)이 실제로 세는 값과 일치해야 하므로 반드시
    `major_total(violation_counts(completed_only(ev)))` 순서를 쓴다 — `completed_only` 를
    빼면 조기 종료한 판이 중대를 덜(또는, 도달한 구간에 중대가 더 있으면 많이) 잡혀 합계
    행이 목표 판정과 다른 숫자를 보이게 된다.

    `student_label` 은 `_comparison_lines` 와 같은 이유(I8) — 열 이름이 `"M4a minor"`/
    `"M4a major"` 로 하드코딩돼 있었다.
    """
    lines = []
    for i, label in enumerate(stage_labels, start=1):
        sc_m3 = violation_counts(m3_ev[label])
        sc_m4a = violation_counts(m4a_ev[label])
        sc_t = violation_counts(teacher_ev[label])
        items = sorted(set(sc_m3) | set(sc_m4a) | set(sc_t))
        lines += ["", f"### 단계{_circled(i)} — 항목별 위반(구간-슬롯 수)", "",
                  f"| 항목 | M3 minor | M3 major | {student_label} minor | {student_label} major |"
                  " 선생님 minor | 선생님 major |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        if not items:
            lines.append("| (위반 없음) | 0 | 0 | 0 | 0 | 0 | 0 |")
        else:
            for item in items:
                a3, a4, at = sc_m3.get(item, {}), sc_m4a.get(item, {}), sc_t.get(item, {})
                name = rs.score_fma.ITEMS.get(item, "?")
                lines.append(f"| {item}({name}) | {a3.get('minor', 0)} | {a3.get('major', 0)} | "
                            f"{a4.get('minor', 0)} | {a4.get('major', 0)} | "
                            f"{at.get('minor', 0)} | {at.get('major', 0)} |")
        m3_major = major_total(violation_counts(completed_only(m3_ev[label])))
        m4a_major = major_total(violation_counts(completed_only(m4a_ev[label])))
        t_major = major_total(violation_counts(completed_only(teacher_ev[label])))
        lines.append(f"| **중대 합계(완주 판 기준)** | — | {m3_major} | — | {m4a_major} | — | {t_major} |")
    return lines


def _goal_lines(m4a_ev: dict, teacher_ev: dict, major_totals: dict, rows: list, best_step,
                eval_seeds: int) -> list:
    """목표 판정을 `verdict.judge()` 에 위임한다 — 학습 스크립트(`train_ppo.py`)와 성적표가

    같은 규칙을 쓰게 하려는 것이다. `judge()` 가 낸 네 줄 중 `precondition=True` 인 두 줄
    (완주율·선생님 대비 점수)은 3번(출발점 대비 점수)에 수학적으로 포함된 전제다
    (`mean_score <= 100 * goal_rate` 이고 `선생님-10 <= 90 < 94.2 <= M3` 이므로 3번이 참이면
    1·2번은 항상 참이다) — 헤드라인 카운트에서 빼야 M4a 를 망친 "형식상 N 줄 달성" 부풀림을
    재현하지 않는다. 실질 판정은 항상 2줄(출발점 대비 점수·전 항목 중대 위반)이다.
    """
    s1, s2 = m4a_ev.get(STAGE1_LABEL), m4a_ev.get(STAGE2_LABEL)
    t1, t2 = teacher_ev.get(STAGE1_LABEL), teacher_ev.get(STAGE2_LABEL)
    missing = s1 is None or s2 is None or t1 is None or t2 is None
    if missing:
        return ["", "## 목표 판정", "",
                f"단계 라벨이 `{STAGE1_LABEL}`/`{STAGE2_LABEL}` 이 아니라 목표를 판정할 수 없다"
                " (log.jsonl 의 stages 키를 확인해라)."]

    verdicts = judge(m4a_ev, teacher_ev, major_totals, eval_seeds)
    real = [v for v in verdicts if not v.precondition]
    preconditions = [v for v in verdicts if v.precondition]
    achieved = sum(1 for v in real if v.ok)
    held = sum(1 for v in real if "보류" in v.line)

    head = f"- 실질 판정 {len(real)} 줄 중 **{achieved} 줄 달성**"
    head += f"({held} 줄 보류 — 시드 수가 안 맞아 M3 대비 비교가 성립하지 않는다)." if held else "."

    lines = ["", "## 목표 판정", "", head, ""]
    lines += [f"- **{v.name}** — {v.line}" for v in real]
    # I9(2026-09-22 최종 리뷰): `verdicts` 전체로 걸면 precondition=True 인 두 줄만 통과해도
    # caveat 가 뜬다 — "실질 판정 2 줄 중 0 줄 달성" 바로 밑에 "위 **달성**이..." 가 찍히는
    # 자기모순이 실제 성적표에 나왔다. `real`(precondition=False) 만 보고 판단해야 한다.
    if any(v.ok for v in real):
        caveat = _best_step_caveat(rows, best_step)
        if caveat:
            lines.append(f"  - ⚠️ {caveat}")

    lines += ["", "### 전제(3번 '출발점 대비 점수' 에 이미 포함됨 — 헤드라인에서 제외)", ""]
    lines += [f"- **{v.name}** — {v.line}" for v in preconditions]
    return lines


def _sweep_lines(sweep_paths: list) -> list:
    """`--sweep <sweep.json>`(여러 번) — 시드 편차 표.

    M4a 는 "결과 보고 가장 좋은 실행을 고르는" 함정에 빠졌다 — 시드 하나의 좋은 숫자가 운인지
    실력인지 시드 여러 개를 같이 보지 않으면 알 수 없다. `sweep.json` 의 `complete: false` 는
    부분 결과(일부 시드가 아직 안 돌았거나 실패했다)라는 뜻이라, 조용히 완성본처럼 보이면
    안 된다 — 그대로 드러낸다. 목표 달성 여부는 각 시드의 `summary["verdict"]` 에서
    `precondition=False` 인 줄(실질 판정)이 전부 `ok` 인 시드만 센다 — 전제 두 줄까지 세면
    같은 부풀림 함정을 여기서도 반복한다.
    """
    lines = ["", "## 시드 편차(--sweep)", ""]
    if not sweep_paths:
        return lines + ["`--sweep` 을 안 줬다 — 시드 편차 표가 없다."]
    for path in sweep_paths:
        with open(path, encoding="utf-8") as f:
            sw = json.load(f)
        name = sw.get("name", os.path.basename(path))
        seeds = sw.get("seeds", [])
        runs = sw.get("runs", [])
        complete = sw.get("complete", True)
        spread = sw.get("spread", {})
        lines += ["", f"### {name}(`{path}`)", ""]
        if not complete:
            lines.append("**부분 결과 — `complete: false`(일부 시드가 아직 안 돌았거나 실패했다).**")
        lines.append(f"- 시드 {len(seeds)}개({', '.join(str(s) for s in seeds)}) 중 "
                     f"실제로 끝난 실행 {len(runs)}개")
        goal_hits = 0
        for r in runs:
            verdict = r.get("summary", {}).get("verdict", [])
            real = [v for v in verdict if not v.get("precondition")]
            if real and all(v.get("ok") for v in real):
                goal_hits += 1
        lines.append(f"- **{len(seeds)} 개 시드 중 {goal_hits} 개**에서 목표 달성"
                     "(실질 판정 줄이 전부 통과한 시드 수 — 전제 두 줄은 안 센다)")
        if spread:
            lines += ["", "| 단계 | 완주율 min~max | 점수 min~max |", "|---|---:|---:|"]
            for stage in ("stage1", "stage2"):
                if stage not in spread:
                    continue
                gr, sc = spread[stage]["goal_rate"], spread[stage]["mean_score"]
                lines.append(f"| {stage} | {_fmt_pct(gr['min'])}~{_fmt_pct(gr['max'])} | "
                             f"{_fmt_score(sc['min'])}~{_fmt_score(sc['max'])} |")
        else:
            lines.append("(`spread` 없음 — 끝난 시드가 없다.)")
    return lines


def _notes_lines(notes_path) -> list:
    """"관찰과 해석" 절 — 표의 숫자는 전부 이 스크립트가 생성하지만, 무엇이 일어났고 왜인지에

    대한 해석은 생성할 수 없다(2026-09-22 지시). `--notes` 파일 내용을 그대로 삽입만 한다 —
    검증도 가공도 안 한다. 안 주면 정확히 이 한 줄만 남긴다(자리표시자를 알아서 지어내지 않는다).
    """
    lines = ["", "## 관찰과 해석", ""]
    if not notes_path:
        return lines + ["관찰과 해석 — `--notes` 로 주지 않았다"]
    with open(notes_path, encoding="utf-8") as f:
        content = f.read().rstrip("\n")
    return lines + [content]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train_ppo.py 가 남긴 실행 폴더(log.jsonl, ac-best.pt)")
    ap.add_argument("--compare", action="append", default=[],
                    help="비교할 다른 실행 폴더(반복 가능) — 요약 비교 표와 자기 학습 곡선 표에만 나온다"
                        "(M3/선생님 대비·항목별 위반·목표 판정은 --run 하나만 싣는다)")
    ap.add_argument("--notes", default=None,
                    help="'관찰과 해석' 절에 그대로 삽입할 마크다운 파일(가공 없이 그대로 붙인다)")
    ap.add_argument("--m3", default=None, help="비교할 M3 DrivePolicy 체크포인트(policy-rN.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="목표 판정에 쓰는 마지막 전체 평가 시드 수 — 학습 중 주기 평가(log.jsonl, 시드"
                        " 1개)와는 별개로 이 스크립트가 다시 돈다")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="hparams 로깅 이전 실행(log.jsonl 에 hparams.seed 가 없는 실행)일 때만 쓰는"
                        " 폴백 — hparams 가 있으면 거기 적힌 seed 를 그대로 쓰고 이 값은 무시한다")
    ap.add_argument("--skip-eval", action="store_true",
                    help="정책을 실제로 몰아 보는 비교·위반·목표 판정을 건너뛰고 학습 곡선·실행 비교"
                        " 표만 만든다(테스트·빠른 확인용)")
    ap.add_argument("--sweep", action="append", default=[],
                    help="`sweep_ppo.py` 가 낸 sweep.json(반복 가능) — 시드 편차 표를 만든다"
                        "(M4a 의 '결과 보고 가장 좋은 실행을 고르는' 함정을 막는다)")
    ap.add_argument("--title", default="M4a 성적표 — PPO",
                    help="성적표 맨 위 제목(h1) — 기본값은 지금까지의 'M4a 성적표 — PPO' 그대로다"
                        "(리뷰 Minor: M4b 성적표를 만들 땐 --title 'M4b 성적표 — PPO' 로 넘겨라)")
    ap.add_argument("--student-label", default="학생",
                    help="비교 표·항목별 표에서 채점 대상 실행을 가리키는 열 이름(기본 '학생')"
                        "(2026-09-22 최종 리뷰 I8: 'M4a 학생'/'M4a minor' 가 하드코딩돼 있어 M4b"
                        " 이후 성적표에도 M4a 라벨이 찍혔다)")
    a = ap.parse_args()
    if not a.skip_eval and not a.m3:
        ap.error("--skip-eval 이 아니면 --m3(비교할 M3 체크포인트)가 필요하다")

    primary_rows = _read_rows(a.run)
    if not primary_rows:
        ap.error(f"{os.path.join(a.run, 'log.jsonl')} 가 없거나 비어 있다")
    last = primary_rows[-1]
    last_eval = _last_eval_row(primary_rows)
    stage_labels = list(last_eval["stages"]) if last_eval else []
    curricula = _curricula_from_log(primary_rows)

    run_dirs = [a.run] + list(a.compare)
    rows_by_run = {rd: (primary_rows if rd == a.run else _read_rows(rd)) for rd in run_dirs}
    best_steps = {rd: _detect_best_step(rd) for rd in run_dirs}

    # 라이브 평가(비교 표·항목별 위반·목표 판정·실행 비교 표의 항목⑦·② 열이 쓴다) — `--run` 은
    # 한 번만 평가해 두 곳(깊은 비교 절 + 실행 비교 표)에 그대로 재사용한다(중복 평가로 몇 분씩
    # 낭비하지 않는다). `--compare` 실행은 체크포인트가 있을 때만(학습이 끝났을 때만) 평가한다.
    m4a_ev = m3_ev = teacher_ev = None
    extra_evs = {}
    dev = None
    if not a.skip_eval:
        dev = pick_device(a.device)
        eval_seeds = tuple(range(a.eval_seeds))
        stage_paths = dict(curricula)
        for label in stage_labels:
            if not os.path.exists(stage_paths[label]):
                ap.error(f"커리큘럼 파일을 못 찾았다: {stage_paths[label]}(라벨 '{label}')")
        boards_by_stage = {label: load_curriculum(stage_paths[label])[1] for label in stage_labels}

        ac = ActorCritic.load(os.path.join(a.run, "ac-best.pt"), device=dev)
        ac.eval()
        m3_policy = DrivePolicy.load(a.m3, device=dev)

        # 채점 대상 실행(`--run`)이 실제로 쓴 보상 설정을 log.jsonl 의 hparams 에서 되살린다 —
        # 안 그러면 평가가 `EnvConfig()` 기본값으로 돌아 '평균 보상' 이 학습 목적함수와 다른
        # 자가 된다(M4c 최종 리뷰 I6). M3 학생도 같은 설정으로 재야 비교가 성립한다.
        run_env_cfg = EnvConfig(reward=reward_config_from_hparams(last.get("hparams")))

        m4a_ev = {label: evaluate_policy(ac.policy, boards_by_stage[label], seeds=eval_seeds,
                                         config=run_env_cfg)
                  for label in stage_labels}
        m3_ev = {label: evaluate_policy(m3_policy, boards_by_stage[label], seeds=eval_seeds,
                                        config=run_env_cfg)
                 for label in stage_labels}
        teacher_ev = {label: evaluate_teacher(boards_by_stage[label], seeds=eval_seeds)
                      for label in stage_labels}
        extra_evs[a.run] = m4a_ev

        for rd in a.compare:
            rows = rows_by_run[rd]
            if not rows or best_steps.get(rd) is None:
                continue   # 학습이 아직 안 끝났다(ac-best.pt 없음) — 요약 표에서 "미완료" 로 남는다
            cand = ActorCritic.load(os.path.join(rd, "ac-best.pt"), device=dev)
            cand.eval()
            # 비교 실행은 각자 자신의 보상 설정으로 잰다 — 여러 실행이 서로 다른 목적함수로
            # 학습됐을 수 있으므로(예: --comfort-steer 스윕) 그 실행 자신의 log.jsonl 이 자다.
            cand_env_cfg = EnvConfig(reward=reward_config_from_hparams(rows[-1].get("hparams")))
            evs = {}
            for label, path in _curricula_from_log(rows):
                if os.path.exists(path):
                    _n, boards = load_curriculum(path)
                    evs[label] = evaluate_policy(cand.policy, boards, seeds=eval_seeds,
                                                 config=cand_env_cfg)
            extra_evs[rd] = evs

    lines = [f"# {a.title}", "",
             f"- 날짜 {datetime.date.today().isoformat()} · 성적표 생성 머신 `{platform.node()}` · "
             f"규칙 스택 `{rs.commit()[:7]}`",
             f"- 채점 대상 실행 `{a.run}`(어느 머신에서 돌렸는지는 이 경로로 안다, 예: `runs/omen/...`)"
             f" · 총 스텝 {last['step']} · 걸린 시간(마지막 로그 시점 기준) {last['elapsed_s']:.0f}초 · "
             f"반복(iter) {last['iter']}회",
             "- 학습 시드: " + _train_seed_line(last.get("hparams"), a.train_seed),
             "- 학습에 쓴 커리큘럼: " + (", ".join(f"`{os.path.relpath(p, REPO)}`" for _, p in curricula)
                                    or "(log.jsonl 이 비어 있어 알 수 없다)"),
             "- 학습 곡선 표의 평가는 학습 스크립트가 값싸게 자주 도는 주기 평가로 **시드 1개**다. "
             f"아래 목표 판정에 쓰는 수치는 그와 별개로 이 성적표 생성 스크립트가 시드 **{a.eval_seeds}개**"
             "로 다시 돌린 마지막 전체 평가다."]

    # 학습 곡선 표 + 진단 곡선 표 — 실행마다(비교 대상 포함) 나란히
    for rd in run_dirs:
        rows = rows_by_run[rd]
        name = os.path.basename(os.path.normpath(rd))
        lines += ["", f"### 학습 곡선 — 실행 `{name}`(`{rd}`)"]
        if not rows:
            lines += ["", "로그가 없다(학습이 아직 시작 전이거나 첫 평가 전이다)."]
            continue
        rd_last_eval = _last_eval_row(rows)
        lines += _curve_table(rows, list(rd_last_eval["stages"]) if rd_last_eval else [])
        lines += _diagnostic_table(rows)

    # M4d Task 1 재리뷰(Important #3): `--compare` 실행은 각자 자신의 보상 설정으로 평가된다
    # (위 `run_env_cfg`/`cand_env_cfg` 참고) — 그래서 위 학습 곡선 표들과 아래 실행 비교 표의
    # "평균 보상" 열은 실행마다 서로 다른 자로 잰 값일 수 있다. 가로로 비교하면 안 된다는 것을
    # 캡션으로 명시하지 않으면 읽는 사람이 `repeat` 대 `once_per_section` 처럼 정의상 스케일이
    # 다른 값을 같은 잣대로 오독한다.
    lines += ["", "> 평균 보상 열은 각 실행 **자신의** 보상 설정으로 잰 값이다 — 설정이 다른"
                  " 실행끼리 이 열을 가로로 비교하면 안 된다(완주율·점수는 심판이 따로 매겨"
                  " 설정과 무관하니 비교해도 된다)."]

    lines += _summary_table(run_dirs, rows_by_run, best_steps, extra_evs, a.skip_eval)

    if a.skip_eval:
        lines += ["", "## 목표 판정", "",
                  "`--skip-eval` 로 만들어 정책을 실제로 몰아 보지 않았다 — 완주율·점수·항목별 위반"
                  " 목표 판정은 보류다(학습 곡선 표·실행 비교 표만 유효하다).",
                  "", "### 항목별 위반 — 평가 생략", "",
                  "`--skip-eval` 이라 위반 집계가 없다 — 항목별 표의 **중대 합계** 행도 낼 수 없다"
                  "(그 값은 실제로 정책을 몰아 봐야 나온다)."]
    else:
        lines += _comparison_lines(stage_labels, m3_ev, m4a_ev, teacher_ev, a.eval_seeds,
                                   student_label=a.student_label)
        lines += _violation_table_lines(stage_labels, m3_ev, m4a_ev, teacher_ev,
                                        student_label=a.student_label)
        major_totals = {label: major_total(violation_counts(completed_only(ev)))
                        for label, ev in m4a_ev.items()}
        lines += _goal_lines(m4a_ev, teacher_ev, major_totals, primary_rows, best_steps[a.run],
                             a.eval_seeds)

    lines += _sweep_lines(a.sweep)

    lines += _notes_lines(a.notes)

    # I8(2026-09-22 최종 리뷰): 다음 마일스톤으로 넘긴 항목을 알리는 고정 절을 여기 박아
    # 뒀었다 — 마일스톤마다 그 목록이 달라지는데 생성기에 고정하면 다음 성적표에서 거짓말이
    # 된다(실제로 이 절과 `--notes` 가 서로 다른 다음 마일스톤을 가리켜 정면으로 부딪혔다).
    # 그 내용은 `--notes` 가 말할 몫이지 생성기가 고정할 몫이 아니다 — 지웠다.
    lines += ["", "산출물(로그·체크포인트)은 `runs/` 아래에 있고 레포에는 넣지 않는다."]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"report": a.out, "rows": len(primary_rows), "skip_eval": a.skip_eval,
                      "runs": run_dirs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

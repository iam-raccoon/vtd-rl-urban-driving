import importlib.util
import json
import math
import os
import subprocess

import pytest
import torch

from vtd_rl.env.reward import RewardConfig
from vtd_rl.rl.actor_critic import ActorCritic
from vtd_rl.rl.buffer import RolloutBuffer
from vtd_rl.rl.ppo import PPOConfig

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_train_ppo_module():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다

    (`tests/policy/test_dagger_loop.py` 와 같은 패턴).
    """
    path = os.path.join(REPO, "scripts", "train_ppo.py")
    spec = importlib.util.spec_from_file_location("train_ppo_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_인자를_안_주면_PPOConfig_기본값과_같다():
    """`--entropy-coef`/`--lr`/`--target-kl`/`--imitation-half-life` 를 하나도 안 주면

    `_build_cfg` 가 만드는 `cfg` 는 `PPOConfig()` 와 완전히 같아야 한다 — CLI 로 여는 것
    자체가 기본 동작을 바꾸면 안 된다(2026-09-21 지시: "아무 인자도 안 주면 동작이 지금과
    100% 같아야 한다").
    """
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "/tmp/불필요-존재안함"])
    cfg = module._build_cfg(a)
    assert cfg == PPOConfig()


def test_하이퍼파라미터_인자가_cfg와_옵티마이저에_실제로_반영된다():
    """"인자가 파싱된다"만 보는 공허한 테스트가 되지 않도록, 실제 `torch.optim.Adam` 을 만들어

    `opt.param_groups[0]["lr"]` 을 직접 본다 — 이 프로젝트에서 `PPOConfig.gamma`/`lam` 이
    `ppo.update()` 에 안 쓰여 조용히 무시된 적이 있어(`compute_gae` 호출부), `--lr` 도 같은
    함정(옵티마이저가 하드코딩된 값을 쓰는)에 빠질 수 있다는 게 코디네이터의 우려였다.
    """
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args([
        "--out", "/tmp/불필요-존재안함", "--lr", "0.00013", "--entropy-coef", "0.2",
        "--target-kl", "0.5", "--imitation-half-life", "12345"])
    cfg = module._build_cfg(a)
    assert cfg.lr == 0.00013
    assert cfg.entropy_coef == 0.2
    assert cfg.target_kl == 0.5
    assert cfg.imitation_half_life == 12345
    # 안 건드린 필드는 그대로(네 필드만 골라 바꿨다는 확인).
    default_cfg = PPOConfig()
    assert cfg.clip == default_cfg.clip and cfg.gamma == default_cfg.gamma
    assert cfg.lam == default_cfg.lam and cfg.value_coef == default_cfg.value_coef

    # main() 이 실제로 쓰는 그 함수로 옵티마이저를 만들어 lr 이 진짜로 닿는지 본다.
    net = module.ActorCritic()
    opt = module._build_optimizer(net, cfg)
    assert opt.param_groups[0]["lr"] == 0.00013


def test_자동_리셋_더미_행이_GAE_사슬을_끊는다():
    """2026-09-21 리뷰 재현 — truncated(시간 초과·정체)로 끝난 판 뒤 자동 리셋 더미 행의

    `done` 에 `term` 만(거짓) 넣으면 다음 판의 큰 가치가 `(γλ)^k` 로 새어 들어와 이전 진짜
    걸음들의 이점을 체계적으로 부풀린다. `_bootstrap_reward_done` 이 더미 행의 reward 를
    제 가치로, done 을 1 로 바꿔야 그 자리에서 사슬이 끊긴다.

    시나리오(리뷰가 직접 재현한 수치와 동일): 4 걸음 — 0,1 은 truncated 로 끝나는 판의 진짜
    전이(마지막 실제 관측의 가치 V(final)=1 은 더미 행 자신의 값으로 나타난다), 2 는 그 뒤에
    오는 자동 리셋 더미 행, 3 은 새 판의 진짜 첫 전이(V(new)=50). γ=0.99, λ=0.95.
    고치기 전: [3.642, 2.82, 1.945, -49.5]. 고친 뒤: [1.921, 0.99, 0.0, -49.5]
    (마지막 값 -49.5 는 더미 행보다 뒤(시간순으로는 앞서 처리되는) 행이라 그대로다 — 이점이
    거꾸로만 샌다는 것 자체도 이 값으로 확인된다).
    """
    module = _load_train_ppo_module()

    def make_buf(reward2, done2):
        buf = RolloutBuffer(4, 1, torch.device("cpu"))

        def row(reward, value, done):
            n = 1
            buf.add(vec=torch.zeros(n, 1), objs=torch.zeros(n, 1, 1), mask=torch.zeros(n, 1),
                    raw=torch.zeros(n, 1), turn=torch.zeros(n, dtype=torch.long),
                    log_prob=torch.zeros(n), value=torch.full((n,), float(value)),
                    reward=torch.full((n,), float(reward)), done=torch.full((n,), float(done)),
                    valid=torch.ones(n))

        row(0.99, 0.0, False)     # 0: 진짜 전이
        row(0.0, 0.0, False)      # 1: truncated 로 끝나는 마지막 진짜 전이(term=False)
        row(reward2, 1.0, done2)  # 2: 자동 리셋 더미 행 — V(final)=1
        row(0.5, 50.0, False)     # 3: 새 판의 진짜 첫 전이 — V(new)=50
        buf.compute_gae(last_value=torch.zeros(1), gamma=0.99, lam=0.95)
        return [round(x, 3) for x in buf.advantages_raw.reshape(-1).tolist()]

    # RED 증거: done 에 term 만(거짓) 넣은 옛 방식 — 사슬이 안 끊긴다.
    buggy = make_buf(reward2=0.0, done2=False)
    assert buggy == [3.642, 2.82, 1.945, -49.5]

    # 실제 프로덕션 함수로 더미 행의 reward·done 을 만든다 — 이 함수를 빼거나 원래대로
    # 되돌리면(reward=term 그대로) 위 buggy 값이 나와 아래 단언이 깨진다.
    reward_t, done_t = module._bootstrap_reward_done(
        reward=[0.0], term=[False], prev_done=[True], value=torch.tensor([1.0]))
    fixed = make_buf(reward2=float(reward_t.item()), done2=float(done_t.item()))
    assert fixed == [1.921, 0.99, 0.0, -49.5]


@pytest.mark.slow
def test_연습_모드가_한_바퀴를_끝낸다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["steps"] > 0 and "stages" in summary and "stage1" in summary["stages"]
    log_path = tmp_path / "run" / "log.jsonl"
    best_path = tmp_path / "run" / "ac-best.pt"
    assert os.path.exists(log_path)
    assert os.path.exists(best_path)

    with open(log_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    last_row = json.loads(lines[-1])
    assert len(last_row["log_std"]) == 2
    assert math.isfinite(last_row["explained_variance"])
    assert "stage1" in last_row["stages"]
    # 이번 실행에 쓴 하이퍼파라미터가 산출물(log.jsonl·요약 JSON)에 실제로 남는지.
    assert last_row["hparams"]["lr"] == pytest.approx(0.0003)
    assert "hparams" in summary and summary["hparams"]["entropy_coef"] == pytest.approx(0.005)

    ActorCritic.load(str(best_path))   # ac-best.pt 가 실제로 ActorCritic 으로 읽혀야 한다


@pytest.mark.slow
def test_같은_폴더에_두_번_쓰지_않는다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    args = [os.path.join(REPO, ".venv", "bin", "python"),
            os.path.join(REPO, "scripts", "train_ppo.py"), "--smoke",
            "--out", str(tmp_path / "run")]
    first = subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert first.returncode == 0
    second = subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    assert second.returncode != 0 and "이미" in (second.stderr + second.stdout)


@pytest.mark.slow
def test_연습_모드가_계측과_판정을_남긴다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0",
                          "--imitation-sigma", "detach"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])

    assert summary["hparams"]["imitation_sigma"] == "detach"
    assert len(summary["verdict"]) == 4
    assert {v["name"] for v in summary["verdict"]} == {
        "완주율", "선생님 대비 점수", "출발점 대비 점수", "전 항목 중대 위반"}
    assert all(isinstance(v["ok"], bool) and v["line"] for v in summary["verdict"])

    rows = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl", encoding="utf-8")]
    assert rows, "log.jsonl 이 비어 있다"
    for r in rows:
        for k in ("rollout_return_mean", "rollout_return_n", "rollout_len_mean",
                  "drift_l2", "drift_rel", "drift_log_std", "drift_rest_rel"):
            assert k in r, k
    assert any(r["rollout_return_n"] > 0 for r in rows), "끝난 판이 한 번도 안 잡혔다"
    # 드리프트는 처음엔 0 에 가깝고 학습이 돌수록 커진다. `drift_log_std` 를 `drift_l2` 와
    # 따로 확인한다 — `drift_l2`(전체망 노름)만 보면 `drift_log_std` 를 로그에서 빼거나 항상
    # 0 으로 찍어도 못 잡는다(trunk 수만 파라미터에 묻힌다, diagnostics.py `policy_drift` 참고
    # — M4b 설계에서 "가장 중요하다"고 지목한 값이라 별도 단언을 둔다).
    assert rows[0]["drift_l2"] < rows[-1]["drift_l2"]
    assert rows[0]["drift_log_std"] < rows[-1]["drift_log_std"]
    # 평가 줄만 "stages" 로 걸러진다 — Task 6 성적표가 이 자리로 평가 줄을 고른다.
    evals = [r for r in rows if "stages" in r]
    assert evals and any(k.startswith("outcome_") for k in evals[-1])
    # 롤아웃마다 한 줄(가벼운 진단)이 남아 평가 줄보다 로그 해상도가 높아야 한다 — 2026-09-22
    # 리뷰 지적(3M 실행에서 평가만으로는 약 6줄뿐이라 붕괴 구간을 점 2개로만 본다).
    assert len(rows) > len(evals)
    # ReturnTracker 와 OutcomeCounter, 서로 다른 두 계측기의 교차 일관성 — 실행 내내 완주한
    # 판 총수는 어느 쪽으로 세도 같아야 한다(M4a 의 갱신 횟수 8배 오류와 같은 종류의 버그를
    # 잡는 자리). 단, 이 등식은 이동창(30)이 아직 안 찼을 때만 성립한다 — 완주 판이 30개를
    # 넘으면 tracker 쪽 rollout_return_n 은 30에서 잘리고 outcomes 쪽 누적 합만 계속 커진다
    # (스모크는 판 수가 적어 항상 30 미만이라 성립한다).
    total_outcomes = sum(v for r in rows for k, v in r.items() if k.startswith("outcome_"))
    assert total_outcomes == rows[-1]["rollout_return_n"]


def test_중대_집계가_완주_판만_센다():
    """`_major_totals` 가 `completed_only` 를 실제로 거치는지 — 서브프로세스도, 완주하는

    정책도 필요 없다. `completed_only` 는 `ev["episodes"]` 만 보고 `violation_counts` 는
    `e.sheet` 만 보므로 `EpisodeOutcome` 의 나머지 필드는 아무 값이나 된다(2026-09-22 리뷰).
    `completed_only` 가 빠지면(=미완주 판까지 센다) 결과가 1 이 아니라 2 가 된다.
    """
    mod = _load_train_ppo_module()
    from vtd_rl.policy.evaluate import EpisodeOutcome
    ev = {"episodes": [EpisodeOutcome("b", 0, "goal",    10, 0.0, 100.0, [{"②": "major"}]),
                       EpisodeOutcome("b", 1, "timeout",  5, 0.0,   0.0, [{"②": "major"}])]}
    assert mod._major_totals({"stage1": ev}) == {"stage1": 1}   # 2 면 completed_only 가 빠진 것


def test_reward_terms_를_환경별_리스트로_바꾼다():
    """gymnasium 의 정보 벡터화(`VectorEnv._add_info`)는 값이 dict 면 재귀적으로 파고들어

    {항목명: 환경별 배열} 로 만든다 — 브리프가 가정한 "환경 수만큼의 dict 배열"과 정반대
    모양이다(2026-09-26 실측: 판이 실제로 끝나는 스텝에서도 `info["reward_terms"]` 는
    이 전치된 모양으로 나온다). `_reward_terms_per_env` 가 `RewardTermTracker.add()` 의
    계약(환경별 dict 리스트)에 맞춰 되돌린다. `_` 로 시작하는 마스크 키는 걸러낸다.
    """
    import numpy as np
    mod = _load_train_ppo_module()
    info = {"reward_terms": {"progress": np.array([1.0, 2.0]),
                             "_progress": np.array([True, True]),
                             "comfort": np.array([-0.5, -0.1])}}
    out = mod._reward_terms_per_env(info, 2)
    assert out == [{"progress": 1.0, "comfort": -0.5}, {"progress": 2.0, "comfort": -0.1}]


def test_reward_terms가_비어있으면_환경_수만큼_빈_dict를_낸다():
    """리셋 직후(모든 환경이 빈 dict)면 gymnasium 이 재귀할 것이 없어 `{}` 그대로 나온다

    (`VtdDriveEnv.reset()`/`_frozen_step()` 이 내는 모양) — `reward_terms` 키 자체가
    없을 때도 같이 견뎌야 한다.
    """
    mod = _load_train_ppo_module()
    assert mod._reward_terms_per_env({"reward_terms": {}}, 2) == [{}, {}]
    assert mod._reward_terms_per_env({}, 2) == [{}, {}]


def test_reward_terms_혼합_리셋을_실제_gymnasium_add_info로_검증한다():
    """병렬 학습의 정상 상태 — 일부 환경은 방금 리셋(빈 dict), 나머지는 주행 중(실값) —

    을 실제 `gymnasium.vector.vector_env.VectorEnv._add_info` 에 태워 만든다(가짜 dict 를
    손으로 짜지 않는다 — gymnasium 이 채움 방식을 바꾸면 이 테스트가 그걸 잡아야 한다).

    `_add_info` 는 `reward_terms`(dict) 처럼 값이 dict 인 키를 재귀적으로 파고들어
    `info["reward_terms"]` 자체를 `{항목명: 환경별 배열, "_항목명": 환경별 bool 마스크}` 로
    만든다(2026-09-26 실측, `.venv/.../gymnasium/vector/vector_env.py` 소스로 직접 확인 —
    이 태스크의 브리프가 "마스크 없으면 모든 환경이 그 항목을 낸 것"이라 적었지만, 실측으로는
    **마스크는 항목 키가 하나라도 등장하면 항상 함께 생기고**(전원이 냈으면 전원 True인 마스크가
    생긴다), 항목 키 자체가 아예 없을 때만(=이 걸음에 그 항목을 낸 환경이 하나도 없을 때만)
    마스크도 같이 없다 — `_reward_terms_per_env` 의 "마스크 없으면 전원 유효" 처리는 그 경우를
    덮는 방어 코드다). 리셋 직후 환경(env0)의 자리는 gymnasium 이 0.0 으로 채우고 그 자리의
    마스크(`_progress` 등)를 거짓으로 낸다 — `_reward_terms_per_env` 는 이 마스크를 직접 읽어
    거짓인 자리를 dict 에서 아예 빼야 한다(채움값 0.0 에 암묵적으로 기대면 안 된다).
    """
    import types

    import numpy as np
    from gymnasium.vector.vector_env import VectorEnv

    from vtd_rl.rl.diagnostics import TERM_KEYS, RewardTermTracker

    class _Dummy:            # VectorEnv._add_info 는 self.num_envs 만 본다
        num_envs = 2

    d = _Dummy()
    d._add_info = types.MethodType(VectorEnv._add_info, d)

    real = {"progress": 1.0, "time": -0.2, "violation": -3.0, "comfort": -0.5}
    info: dict = {}
    info = d._add_info(info, {"reward_terms": {}}, 0)      # env0: 방금 리셋
    info = d._add_info(info, {"reward_terms": real}, 1)    # env1: 주행 중(실값)

    mod = _load_train_ppo_module()
    out = mod._reward_terms_per_env(info, 2)

    assert not (set(out[0]) & set(TERM_KEYS)), out[0]   # env0 → {} (또는 TERM_KEYS 가 없는 dict)
    assert out[1] == pytest.approx(real)                # env1 → 넣은 값 그대로

    # `RewardTermTracker.add()` 에 실제로 먹여 env0 누적이 정확히 0 으로 남는지까지 확인한다 —
    # "0 을 더한 것"과 "안 더한 것"이 지금은 결과가 같지만(모든 항목이 순가산량), 나머지
    # 코드가 이 어댑터를 신뢰해도 되는 근거는 이 단언이다.
    tracker = RewardTermTracker(2)
    tracker.add({"reward_terms": out})
    tracker.add_done(np.array([True, False]))   # env0 만 이 걸음에 끝난 것으로 표시해 바로 flush
    stats = tracker.stats()
    assert stats["term_n"] == 1
    for k in TERM_KEYS:
        assert stats[f"term_{k}_mean"] == 0.0, k


@pytest.mark.slow
def test_연습_모드가_항목별_보상을_남긴다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "train_ppo.py"),
                          "--smoke", "--out", str(tmp_path / "run"), "--seed", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = [json.loads(l) for l in open(tmp_path / "run" / "log.jsonl", encoding="utf-8")]
    for r in rows:
        for k in ("term_progress_mean", "term_violation_mean", "term_comfort_mean", "term_n"):
            assert k in r, k
    done = [r for r in rows if r["term_n"] > 0]
    assert done, "끝난 판이 한 번도 안 잡혔다"
    # 승차감은 벌점이므로 음수여야 한다 — 부호가 뒤집히면 배분 해석이 통째로 틀어진다
    assert done[-1]["term_comfort_mean"] <= 0.0
    # M4c 행동 상자 진단 — tanh 는 `|control| ≤ 1` 을 자명하게 통과시키므로(리뷰 I3) 포화 비율과
    # 스쿼시 전 평균 크기를 따로 잰다. 매 롤아웃(가벼운 진단 줄, 평가 줄 아님)마다 실려야 한다.
    for r in rows:
        for k in ("act_abs_mean", "act_sat_frac", "mean_abs_mean"):
            assert k in r, k
            assert r[k] is None or r[k] >= 0.0, (k, r[k])
    with_values = [r for r in rows if r["act_abs_mean"] is not None]
    assert with_values, "행동 상자 진단이 한 번도 안 찍혔다"
    assert all(0.0 <= r["act_sat_frac"] <= 1.0 for r in with_values)


def test_인자를_안_주면_RewardConfig_기본값과_같다():
    """`--comfort-steer`/`--comfort-accel`/`--comfort-on-intent` 를 하나도 안 주면

    `_build_reward_cfg` 가 만드는 `cfg` 는 `RewardConfig()` 와 완전히 같아야 한다 —
    `test_인자를_안_주면_PPOConfig_기본값과_같다` 와 같은 취지(CLI 로 여는 것 자체가 기본
    동작을 바꾸면 안 된다).
    """
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args(["--out", "/tmp/불필요-존재안함"])
    cfg = module._build_reward_cfg(a)
    assert cfg == RewardConfig()


def test_승차감_인자가_RewardConfig에_반영된다():
    module = _load_train_ppo_module()
    a = module._build_parser().parse_args([
        "--out", "/tmp/불필요-존재안함", "--comfort-steer", "-0.01",
        "--comfort-accel", "-0.005", "--comfort-on-intent"])
    cfg = module._build_reward_cfg(a)
    assert cfg.comfort_steer == -0.01
    assert cfg.comfort_accel == -0.005
    assert cfg.comfort_on_intent is True
    # 안 건드린 필드는 그대로(세 필드만 골라 바꿨다는 확인).
    default_cfg = RewardConfig()
    assert cfg.progress_total == default_cfg.progress_total
    assert cfg.collision == default_cfg.collision

    off = module._build_reward_cfg(module._build_parser().parse_args(["--out", "x"]))
    assert off.comfort_on_intent is False


@pytest.mark.slow
def test_comfort_on_intent가_승차감의_표본_잡음을_줄인다(tmp_path):
    """`--comfort-on-intent` 를 켜면 실행 행동이 아니라 정책의 의도(결정적 평균)의 변화로

    승차감을 잰다 — PPO 가 매 걸음 뽑는 독립 표본(탐색 잡음)이 승차감에 덜 잡혀야 하므로
    끈 실행보다 `|term_comfort_mean|` 이 작아야 한다(M4b 실측: 실행 행동 기준으로는 판당
    -108.67). 이 배선은 `VtdDriveEnv.intent`(속성)를 거쳐 `venv.set_attr()` 로 별도 프로세스
    (비동기 벡터 환경) 워커까지 실제로 닿아야 하므로, `set_attr` 호출이나 `RewardShaper` 의
    의도 분기 중 하나라도 빠지면 이 테스트가 그 차이를 못 보고 실패해야 한다.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    def run(tag, extra):
        out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                              os.path.join(REPO, "scripts", "train_ppo.py"),
                              "--smoke", "--out", str(tmp_path / tag), "--seed", "0", *extra],
                             capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
        assert out.returncode == 0, out.stderr[-3000:]
        rows = [json.loads(l) for l in open(tmp_path / tag / "log.jsonl", encoding="utf-8")]
        done = [r for r in rows if r["term_n"] > 0]
        assert done, "끝난 판이 한 번도 안 잡혔다"
        return done[-1]["term_comfort_mean"]

    off = run("off", [])
    on = run("on", ["--comfort-on-intent"])
    assert abs(on) < abs(off), (on, off)


def test_시그마_모드가_cfg에_닿는다():
    mod = _load_train_ppo_module()   # 이 파일이 이미 쓰는 importlib 헬퍼(이름 그대로)
    a = mod._build_parser().parse_args(["--out", "x", "--imitation-sigma", "detach"])
    assert mod._build_cfg(a).imitation_sigma == "detach"
    b = mod._build_parser().parse_args(["--out", "x"])
    assert mod._build_cfg(b).imitation_sigma == "learn"


def test_smoke가_eval_every_0_을_가리지_않는다():
    """`--smoke` 가 `--eval-every` 검증보다 먼저 스모크 기본값으로 덮어쓰면 `--eval-every 0`

    같은 잘못된 값이 검증 없이 통과한다(2026-09-22 리뷰 지적) — `--steps 0` 도 같은 함정이라
    같이 확인한다. `main()` 은 이 두 검증을 통과 못 하면 무거운 준비(venv·모델) 전에
    `SystemExit`(argparse `error()`)로 죽으므로 서브프로세스 없이 빠르게 돈다.
    """
    import sys
    mod = _load_train_ppo_module()
    old_argv = sys.argv
    try:
        for bad_args in (["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함",
                          "--eval-every", "0"],
                         ["train_ppo.py", "--smoke", "--out", "/tmp/불필요-존재안함",
                          "--steps", "0"]):
            sys.argv = bad_args
            with pytest.raises(SystemExit):
                mod.main()
    finally:
        sys.argv = old_argv

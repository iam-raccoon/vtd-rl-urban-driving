import numpy as np
import torch

from vtd_rl.rl.diagnostics import (OutcomeCounter, ReturnTracker, policy_drift,
                                   snapshot_policy)


def test_끝난_판의_리턴만_담는다():
    t = ReturnTracker(2)
    t.add(np.array([1.0, 2.0]), np.array([False, False]))
    t.add(np.array([3.0, 4.0]), np.array([True, False]))
    s = t.stats()
    assert s["rollout_return_n"] == 1
    assert abs(s["rollout_return_mean"] - 4.0) < 1e-9      # 0번 환경: 1+3
    assert abs(s["rollout_len_mean"] - 2.0) < 1e-9


def test_끝난_판이_없으면_평균은_None():
    t = ReturnTracker(2)
    t.add(np.array([1.0, 1.0]), np.array([False, False]))
    s = t.stats()
    assert s["rollout_return_n"] == 0 and s["rollout_return_mean"] is None
    assert s["rollout_len_mean"] is None


def test_판이_끝나면_누적이_0에서_다시_시작한다():
    t = ReturnTracker(1)
    t.add(np.array([5.0]), np.array([True]))
    t.add(np.array([2.0]), np.array([True]))
    s = t.stats()
    assert s["rollout_return_n"] == 2
    assert abs(s["rollout_return_mean"] - 3.5) < 1e-9      # (5 + 2) / 2


def test_reset_은_담은_것만_비우고_진행중_누적은_남긴다():
    t = ReturnTracker(1)
    t.add(np.array([5.0]), np.array([True]))
    t.add(np.array([1.0]), np.array([False]))              # 진행 중 누적 1.0
    t.reset()
    assert t.stats()["rollout_return_n"] == 0
    t.add(np.array([2.0]), np.array([True]))
    assert abs(t.stats()["rollout_return_mean"] - 3.0) < 1e-9   # 1 + 2 — 경계를 넘어 이어졌다


def test_종료_사유를_센다():
    c = OutcomeCounter()
    c.add({"outcome": np.array(["goal", "collision", "running"], dtype=object)})
    c.add({"outcome": np.array(["goal", "stalled", "running"], dtype=object)})
    s = c.stats()
    assert s["outcome_goal"] == 2 and s["outcome_collision"] == 1 and s["outcome_stalled"] == 1
    assert "outcome_running" not in s                      # 진행 중은 종료 사유가 아니다


def test_outcome_키가_없으면_아무것도_안_센다():
    c = OutcomeCounter()
    c.add({})
    assert c.stats() == {}


def test_NEXT_STEP_자동리셋_다음_스텝의_running을_또_세지_않는다():
    """gymnasium AutoresetMode.NEXT_STEP(기본값) 전제를 손으로 재현한다: 종료 스텝에서
    top-level outcome 에 진짜 사유가 실리고, 자동 리셋이 낸 다음 더미 스텝은 'running' 이다.
    collision 은 정확히 1번만 세어져야 한다 — SAME_STEP 모드로 바뀌면 이 전제가 깨진다."""
    c = OutcomeCounter()
    c.add({"outcome": np.array(["collision"], dtype=object)})   # 종료 스텝 — 진짜 사유
    c.add({"outcome": np.array(["running"], dtype=object)})     # 자동 리셋 다음 더미 스텝
    assert c.stats() == {"outcome_collision": 1}


def test_윈도우가_있으면_최근_것만_남는다():
    t = ReturnTracker(1, window=2)
    for r in (5.0, 2.0, 7.0):
        t.add(np.array([r]), np.array([True]))
    s = t.stats()
    assert s["rollout_return_n"] == 2
    assert abs(s["rollout_return_mean"] - 4.5) < 1e-9      # 마지막 두 개(2, 7)만 남는다


def test_윈도우가_None이면_기존과_같다():
    t = ReturnTracker(1, window=None)
    for r in (5.0, 2.0, 7.0):
        t.add(np.array([r]), np.array([True]))
    s = t.stats()
    assert s["rollout_return_n"] == 3
    assert abs(s["rollout_return_mean"] - (5.0 + 2.0 + 7.0) / 3) < 1e-9


def test_드리프트는_움직인_만큼_커진다():
    from vtd_rl.policy.net import PolicyConfig
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    net = ActorCritic(ActorCriticConfig(policy=PolicyConfig(trunk=(16, 16)), value_hidden=(16,)))
    ref = snapshot_policy(net)
    assert policy_drift(net, ref)["drift_l2"] == 0.0
    with torch.no_grad():
        net.policy.log_std.add_(1.0)
    d = policy_drift(net, ref)
    assert abs(d["drift_l2"] - 2.0 ** 0.5) < 1e-5          # 두 차원에 각각 1.0
    assert d["drift_rel"] > 0.0


def test_스냅샷은_이후_변경에_영향받지_않는다():
    from vtd_rl.policy.net import PolicyConfig
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    net = ActorCritic(ActorCriticConfig(policy=PolicyConfig(trunk=(16, 16)), value_hidden=(16,)))
    ref = snapshot_policy(net)
    with torch.no_grad():
        net.policy.log_std.add_(3.0)
    assert policy_drift(net, ref)["drift_l2"] > 1.0        # clone 이 아니면 0 이 나온다


def test_실사이즈에서_drift_rel은_log_std_붕괴를_묻고_drift_log_std는_뚫는다():
    """토이넷(16,16)에서는 이 마스킹이 안 보인다 — 실사이즈(trunk=(256,256), 기본값)라야
    trunk 수만 파라미터가 log_std(2개)를 분모·분자에서 묻는다. M4a 는 정확히 log_std 에서
    무너졌으므로, 전체망 기준 drift_rel 만 보면 그 사건을 놓친다는 것을 이 테스트로 남긴다."""
    from vtd_rl.rl.actor_critic import ActorCritic, ActorCriticConfig
    net = ActorCritic(ActorCriticConfig())          # 실사이즈 기본값 — trunk=(256, 256)
    ref = snapshot_policy(net)
    with torch.no_grad():
        net.policy.log_std.add_(1.5)                # 두 차원 모두 +1.5 (clamp 폭 2.5 에 근접)
    d = policy_drift(net, ref)

    expected_log_std_l2 = (2 * 1.5 ** 2) ** 0.5      # ≈2.121 — log_std 두 차원 각각 +1.5
    assert abs(d["drift_log_std"] - expected_log_std_l2) < 1e-4
    assert d["drift_rest_rel"] == 0.0                # log_std 말고는 아무것도 안 바뀌었다

    # log_std 만 따로 봤을 때 "제 몫"의 상대 이동(초기값이 각 -1.0 이라 노름은 sqrt(2))과 비교하면
    # 전체망 기준 drift_rel 은 trunk 라는 분모에 묻혀 그 1/5 미만으로 깎여 나온다 — 이게 마스킹이
    # 실재한다는 증거다. drift_log_std 는 나누지 않으므로 이 마스킹을 그대로 뚫고 나온다.
    own_ref_l2 = (2 * 1.0 ** 2) ** 0.5               # log_std_init=-1.0, 두 차원
    own_rel = expected_log_std_l2 / own_ref_l2       # = 1.5 (150% 이동)
    assert d["drift_rel"] < own_rel / 5
    assert d["drift_log_std"] > 2.0                  # 절대량은 묻히지 않는다


def test_항목별_보상을_판마다_누적한다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(2)
    t.add({"reward_terms": [{"progress": 1.0, "comfort": -0.5, "violation": 0.0},
                            {"progress": 2.0, "comfort": -0.1, "violation": -3.0}]})
    t.add_done(np.array([False, False]))
    t.add({"reward_terms": [{"progress": 1.0, "comfort": -0.5, "violation": -6.0},
                            {"progress": 2.0, "comfort": -0.1, "violation": 0.0}]})
    t.add_done(np.array([True, False]))
    s = t.stats()
    assert s["term_n"] == 1
    assert abs(s["term_progress_mean"] - 2.0) < 1e-9      # 0번 환경: 1+1
    assert abs(s["term_comfort_mean"] - (-1.0)) < 1e-9    # -0.5 + -0.5
    assert abs(s["term_violation_mean"] - (-6.0)) < 1e-9


def test_끝난_판이_없으면_None():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1)
    t.add({"reward_terms": [{"progress": 1.0}]})
    t.add_done(np.array([False]))
    s = t.stats()
    assert s["term_n"] == 0 and s["term_progress_mean"] is None


def test_리셋_직후_빈_reward_terms_를_견딘다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1)
    t.add({"reward_terms": [{}]})          # VtdDriveEnv.reset() 이 내는 모양
    t.add({"reward_terms": [{"progress": 3.0}]})
    t.add_done(np.array([True]))
    assert abs(t.stats()["term_progress_mean"] - 3.0) < 1e-9


def test_reward_terms_키가_없으면_아무것도_안_센다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1)
    t.add({})
    t.add_done(np.array([True]))
    assert t.stats()["term_n"] == 1 and t.stats()["term_progress_mean"] == 0.0


def test_이동창이_최근_것만_남긴다():
    from vtd_rl.rl.diagnostics import RewardTermTracker
    t = RewardTermTracker(1, window=2)
    for v in (1.0, 2.0, 3.0):
        t.add({"reward_terms": [{"progress": v}]})
        t.add_done(np.array([True]))
    assert t.stats()["term_n"] == 2
    assert abs(t.stats()["term_progress_mean"] - 2.5) < 1e-9   # (2+3)/2

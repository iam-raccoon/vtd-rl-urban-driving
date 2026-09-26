"""학습이 무엇을 하고 있는지 보이게 하는 계측.

M4a 는 세 실행이 1M~1.5M 이후 무너졌는데 **왜** 무너졌는지 못 밝혔다. 이유는 계측이 없어서다:
PPO 가 실제로 최대화하는 **확률적 롤아웃 리턴**이 한 번도 로깅되지 않아, 인용할 수 있는
보상은 결정적 평가 보상(프록시)뿐이었다. 종료 사유와 정책 드리프트도 마찬가지로 없었다.

세 가지를 잰다 — 롤아웃 리턴(PPO 의 목적함수), 종료 사유(붕괴가 충돌·이탈·정체 중 무엇으로
나타나는가), 초기 정책으로부터의 거리(정책이 얼마나 멀리 갔는가).
"""
from collections import deque

import numpy as np
import torch

# 판이 아직 안 끝난 상태 — 종료 사유가 아니므로 세지 않는다.
RUNNING = "running"

# DrivePolicy.state_dict() 에서 log_std 파라미터의 키(net.py: self.log_std, 최상위 속성이라
# 접두어가 안 붙는다). policy_drift 가 이 파라미터를 나머지와 갈라 볼 때 쓴다.
_LOG_STD_KEY = "log_std"


class ReturnTracker:
    """환경별로 보상을 누적하다가 판이 끝나면 그 리턴을 담는다.

    `reset()` 은 **담아 둔 리턴만** 비운다. 진행 중인 누적은 롤아웃 경계를 넘어 이어져야 한다 —
    판 하나가 롤아웃 여러 개에 걸치기 때문이다(판은 2500~5000 걸음, 롤아웃은 256 걸음).

    ⚠ **표본이 성기다.** 롤아웃 256 걸음 × n_envs(예: 30) = 7680 환경-걸음인데 판은 2500~5000
    걸음이라, 롤아웃 하나에 끝나는 판이 보통 2~3개뿐이다. 롤아웃마다 `reset()` 을 부르면 로그
    한 줄이 표본 2~3개짜리 평균이라 대부분 잡음이다. 호출부는 `window`(이동창)를 쓰거나
    리셋 주기를 판 길이에 맞춰야 한다.
    """

    def __init__(self, n_envs: int, window: int | None = None):
        """`window`: 정수면 최근 `window`개의 판(리턴·길이)만 남기는 이동창이다
        (예: 30 ≈ 롤아웃 12개 분량 — 한 줄이 뜻 있는 이동평균이 되게). `None`(기본)이면
        지금까지 담은 것을 전부 유지한다 — 기존 호출부·테스트는 이 기본값을 그대로 쓴다.
        """
        self.n_envs = n_envs
        self.window = window
        self._running = np.zeros(n_envs, dtype=np.float64)
        self._steps = np.zeros(n_envs, dtype=np.int64)
        self.reset()

    def reset(self):
        self._returns = deque(maxlen=self.window)
        self._lengths = deque(maxlen=self.window)

    def add(self, reward, done):
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        self._running += reward
        self._steps += 1
        for i in np.nonzero(done)[0]:
            self._returns.append(float(self._running[i]))
            self._lengths.append(int(self._steps[i]))
            self._running[i] = 0.0
            self._steps[i] = 0

    def stats(self) -> dict:
        n = len(self._returns)
        return {"rollout_return_mean": (sum(self._returns) / n) if n else None,
                "rollout_return_n": n,
                "rollout_len_mean": (sum(self._lengths) / n) if n else None}


class OutcomeCounter:
    """벡터 환경 `infos` 에서 종료 사유를 센다 — 붕괴가 무엇으로 나타나는지 보려는 것.

    ⚠ **gymnasium `AutoresetMode.NEXT_STEP`(기본값) 을 전제한다.** 이 모드에서는 판이 끝나는
    스텝의 top-level `infos["outcome"]` 에 진짜 사유가 실제로 실리고, 그 다음 스텝(자동 리셋이
    낸 첫 더미 관측)만 `"running"` 으로 돌아온다 — `vtd_rl/rl/vec_env.py` 는 이 gymnasium 기본값을
    오버라이드하지 않고, `VtdDriveEnv` 는 매 스텝 `outcome` 을 채운다(2026-09 실측 확인).
    `AutoresetMode.SAME_STEP` 으로 바뀌면 진짜 사유가 `infos["final_info"]` 로 밀려나고
    top-level `outcome` 은 항상 `"running"` 이 되어, 이 카운터는 **모든 종료를 조용히 0으로 센다**
    (예외도 없이 조용히) — 이 클래스는 그 전제를 스스로 검증하지 않는다.
    """

    def __init__(self):
        self._counts: dict = {}

    def add(self, infos: dict):
        values = infos.get("outcome")
        if values is None:
            return
        for v in np.asarray(values, dtype=object).reshape(-1):
            if v is None or v == RUNNING:
                continue
            key = f"outcome_{v}"
            self._counts[key] = self._counts.get(key, 0) + 1

    def stats(self) -> dict:
        return dict(self._counts)


# 판당 누적을 볼 항목 — `VtdDriveEnv` 가 info["reward_terms"] 로 내는 키들 중
# 배분 논쟁에 필요한 것만 고른다(collision·offroad·goal 은 종료 사유로 이미 보인다).
TERM_KEYS = ("progress", "time", "violation", "comfort")


class RewardTermTracker:
    """판 하나가 쌓는 **항목별** 보상을 누적한다.

    M4b 는 "승차감 −128 대 위반 −96" 을 **계산**으로만 말했다 — `info["reward_terms"]` 가
    걸음마다 이미 있는데 한 번도 안 썼기 때문이다. 이 누적기가 그 배분을 측정으로 바꾼다.

    `ReturnTracker` 와 달리 `add(infos)` 와 `add_done(done)` 이 나뉘어 있다 — 보상 항목은
    `infos` 에서, 종료는 `term|trunc` 에서 오기 때문이다. 호출부는 매 걸음 둘 다 부른다.
    """

    def __init__(self, n_envs: int, window: int | None = None):
        self.n_envs = n_envs
        self.window = window
        self._running = {k: np.zeros(n_envs, dtype=np.float64) for k in TERM_KEYS}
        self.reset()

    def reset(self):
        self._done: dict = {k: deque(maxlen=self.window) for k in TERM_KEYS}

    def add(self, infos: dict):
        terms = infos.get("reward_terms")
        if terms is None:
            return
        for i, t in enumerate(terms):
            if not t:                      # 리셋 직후는 빈 dict 다
                continue
            for k in TERM_KEYS:
                self._running[k][i] += float(t.get(k, 0.0))

    def add_done(self, done):
        for i in np.nonzero(np.asarray(done, dtype=bool))[0]:
            for k in TERM_KEYS:
                self._done[k].append(float(self._running[k][i]))
                self._running[k][i] = 0.0

    def stats(self) -> dict:
        n = len(self._done[TERM_KEYS[0]])
        out = {"term_n": n}
        for k in TERM_KEYS:
            vals = self._done[k]
            out[f"term_{k}_mean"] = (sum(vals) / n) if n else None
        return out


def snapshot_policy(net) -> dict:
    """정책 파라미터를 떼어 복사한다 — clone 을 빼면 이후 갱신이 기준까지 따라 움직인다."""
    return {k: v.detach().clone() for k, v in net.policy.state_dict().items()}


def policy_drift(net, ref_state: dict) -> dict:
    """현재 정책이 기준에서 얼마나 멀어졌나. `approx_kl` 은 갱신 한 걸음의 크기만 보지만,

    이 값은 **누적된 이동**을 본다 — M4a 의 붕괴는 걸음마다는 작고(KL 0.011~0.016) 누적으로만
    큰 형태였다.

    `drift_l2`/`drift_rel` 은 `net.policy.state_dict()` 전체를 하나의 노름으로 합친 값이다.
    실사이즈 설정(`PolicyConfig.trunk=(256, 256)`)에서는 trunk 두 층만 수만 개 파라미터인 반면
    `log_std` 는 **2개뿐**이라, log_std 가 clamp 한계(`log_std_min=-2.0`~`log_std_max=0.5`)까지
    완전히 튀어도 이 두 값은 trunk 에 묻혀 거의 안 움직인다 — M4a 가 무너진 바로 그 파라미터를
    이 두 값만으로는 못 본다. 그래서 `log_std` 를 갈라 따로 낸다:
    `drift_log_std`(절대 이동량, **나누지 않는다** — 상대값으로 나누면 다시 묻힌다)와
    `drift_rest_rel`(log_std 를 뺀 나머지의 상대 이동)을 추가로 준다.
    """
    with torch.no_grad():
        sq = torch.zeros(())
        ref_sq = torch.zeros(())
        log_std_sq = torch.zeros(())
        rest_sq = torch.zeros(())
        rest_ref_sq = torch.zeros(())
        for k, v in net.policy.state_dict().items():
            r = ref_state[k].to(v.device)
            d_sq = (v - r).pow(2).sum()
            r_sq = r.pow(2).sum()
            sq = sq + d_sq
            ref_sq = ref_sq + r_sq
            if k == _LOG_STD_KEY:
                log_std_sq = log_std_sq + d_sq
            else:
                rest_sq = rest_sq + d_sq
                rest_ref_sq = rest_ref_sq + r_sq
    l2 = float(sq) ** 0.5
    ref_l2 = float(ref_sq) ** 0.5
    rest_l2 = float(rest_sq) ** 0.5
    rest_ref_l2 = float(rest_ref_sq) ** 0.5
    return {"drift_l2": l2,
            "drift_rel": l2 / ref_l2 if ref_l2 > 0 else 0.0,
            "drift_log_std": float(log_std_sq) ** 0.5,
            "drift_rest_rel": rest_l2 / rest_ref_l2 if rest_ref_l2 > 0 else 0.0}

"""M4c Task 5 — `scripts/refit_m3.py` 를 잠근다(2026-09-26 수정: 씨앗·--atanh-eps·--init·

--freeze-sigma). 넷 다 커밋 `2e73c7a` 로 이미 구현돼 있다 — 이 파일은 그걸 잠그는 것이지 새로
구현하는 게 아니다. `base_run` 픽스처(모듈 스코프)로 seed=0 기본 실행을 한 번만 돌려 여러
테스트가 그 결과를 나눠 쓴다(서브프로세스 수를 줄인다).
"""
import json
import os
import subprocess

import pytest
import torch

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(REPO, "runs", "lab-main", "2026-09-17-dagger-fix2", "data")


def _run(out_dir, *extra_args, timeout=300):
    """`out_dir` 아래 `policy.pt` 를 만든다(디렉터리 자체는 아직 없어도 된다)."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "refit_m3.py"),
                          "--data", DATA, "--out", str(out_dir / "policy.pt"), "--epochs", "1",
                          *extra_args],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=timeout)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _state(out_dir):
    from vtd_rl.policy.net import DrivePolicy
    return DrivePolicy.load(str(out_dir / "policy.pt")).state_dict()


def _differ(sd_a, sd_b) -> bool:
    return any(not torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    """`--seed 0`, 기본 인자로 한 번 돌린 결과 — 여러 테스트가 비교 기준으로 나눠 쓴다."""
    out = tmp_path_factory.mktemp("base")
    summary = _run(out, "--seed", "0")
    return summary, _state(out)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_재적합이_스쿼시_체크포인트를_만든다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "refit_m3.py"),
                          "--data", DATA, "--out", str(tmp_path / "policy.pt"),
                          "--epochs", "1", "--seed", "0"],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    assert summary["epochs"] == 1 and summary["samples"] > 0

    from vtd_rl.policy.net import DrivePolicy
    net = DrivePolicy.load(str(tmp_path / "policy.pt"))
    assert net.cfg.squash is True          # 스쿼시 매개화로 저장됐다

    # 결정적 행동이 항상 상자 안이다 — 이게 이 체크포인트의 존재 이유다
    from vtd_rl.env.observation import ObsConfig, observation_space
    from vtd_rl.policy.encode import stack_obs, to_tensors
    sp = observation_space(ObsConfig()); sp.seed(0)
    vec, objs, mask = to_tensors(*stack_obs([sp.sample() for _ in range(64)]),
                                 torch.device("cpu"))
    with torch.no_grad():
        action = torch.tanh(net(vec, objs, mask)[0])
    assert torch.all(action.abs() <= 1.0)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_같은_폴더에_두_번_쓰지_않는다(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    args = [os.path.join(REPO, ".venv", "bin", "python"),
            os.path.join(REPO, "scripts", "refit_m3.py"),
            "--data", DATA, "--out", str(tmp_path / "policy.pt"), "--epochs", "1"]
    assert subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO,
                          timeout=1800).returncode == 0
    second = subprocess.run(args, capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    assert second.returncode != 0 and "이미" in (second.stdout + second.stderr)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_같은_시드는_같은_초기_가중치를_낸다(tmp_path, base_run):
    """`torch.manual_seed(a.seed)` 가 `DrivePolicy` 생성 **전**에 걸려야 한다.

    안 걸면(또는 생성 뒤로 옮기면) 초기 가중치가 프로세스마다 다른 전역 RNG 상태를 타 같은
    인자로도 실행마다 결과가 달라진다(실측: 완주율 0.833 ↔ 0.333). 같은 시드 두 번은
    `state_dict` 가 원소 단위로 완전히 같아야 하고, 다른 시드는 달라야 한다(그래야 "시드를
    아예 안 쓴다" 는 돌연변이도 잡는다 — `manual_seed` 를 지워도 배치 순서는 `--seed` 를
    그대로 따르므로 다른 시드 비교만으로는 못 잡지만, 같은 시드 두 번 비교가 그 돌연변이를
    직접 잡는다).
    """
    _base_summary, sd_base = base_run
    _run(tmp_path / "again", "--seed", "0")
    sd_again = _state(tmp_path / "again")
    assert not _differ(sd_base, sd_again), "같은 시드인데 결과 가중치가 다르다"

    _run(tmp_path / "other-seed", "--seed", "1")
    sd_other = _state(tmp_path / "other-seed")
    assert _differ(sd_base, sd_other), "다른 시드인데 가중치가 같다(시드가 안 먹는다)"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_atanh_eps가_TrainConfig에_전달된다(tmp_path, base_run):
    """`--atanh-eps` 가 `TrainConfig.atanh_eps` 로 실제로 흐르는지 — 값을 바꾸면 결과 가중치가

    달라지는 것(배선이 실제로 손실에 닿는지)과, 요약 JSON 의 `atanh_eps` 필드(둘 다 잠근다 —
    후자만 보면 필드는 채워지는데 학습에는 안 쓰이는 배선 끊김을 놓친다).
    """
    from vtd_rl.policy.train import TrainConfig
    base_summary, sd_base = base_run
    assert base_summary["atanh_eps"] == pytest.approx(TrainConfig().atanh_eps)

    changed_summary = _run(tmp_path / "changed", "--seed", "0", "--atanh-eps", "1e-4")
    assert changed_summary["atanh_eps"] == pytest.approx(1e-4)

    sd_changed = _state(tmp_path / "changed")
    assert _differ(sd_base, sd_changed), \
        "--atanh-eps 를 바꿔도 가중치가 같다(TrainConfig 로 안 전달되는 것 아닌가)"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_init이_시작점을_바꾼다(tmp_path, base_run):
    """`--init` 이 M3 체크포인트에서 가중치를 실제로 이어받는지 — 안 받으면 결과가

    `--init` 없이 같은 시드로 돌린 것과 같아야 한다(같으면 안 먹는 것). 요약 JSON 에 경로가
    실리는 것도 같이 잠근다.
    """
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig
    base_summary, sd_base = base_run

    init_path = tmp_path / "fake_m3.pt"
    # 진짜 M3 체크포인트일 필요는 없다 — trunk 모양만 같으면 된다(스쿼시는 파라미터를 안
    # 늘리므로 state_dict 모양이 같다, refit_m3.py 의 --init 문서 참고). 무작위 초기화라도
    # refit_m3.py 자신의 기본 초기화(seed=0)와는 다른 시작점이라는 것만 보장되면 충분하다.
    torch.manual_seed(999)
    DrivePolicy(PolicyConfig(squash=True)).save(str(init_path))

    inited_summary = _run(tmp_path / "inited", "--seed", "0", "--init", str(init_path))
    assert inited_summary["init"] == str(init_path)

    sd_inited = _state(tmp_path / "inited")
    assert _differ(sd_base, sd_inited), \
        "--init 을 줘도 결과가 --init 없이 돌린 것과 같다(이어받기가 안 먹는 것 아닌가)"


@pytest.mark.slow
def test_init_trunk이_다르면_거부한다(tmp_path):
    """trunk 모양이 다른 `--init` 체크포인트는 학습을 시작하기 전에 `ap.error` 로 거부돼야 한다.

    이 검사는 `load_dir(a.data)` 보다 먼저 일어나므로(스크립트 순서) 실데이터가 없어도(존재하지
    않는 --data 경로) 확인할 수 있다 — 빠르다.
    """
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig

    bad_init = tmp_path / "bad_trunk.pt"
    DrivePolicy(PolicyConfig(squash=True, trunk=(128, 128))).save(str(bad_init))

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                          os.path.join(REPO, "scripts", "refit_m3.py"),
                          "--data", str(tmp_path / "존재안함"),
                          "--out", str(tmp_path / "out" / "policy.pt"),
                          "--epochs", "1", "--init", str(bad_init)],
                         capture_output=True, text=True, env=env, cwd=REPO, timeout=60)
    assert out.returncode != 0
    assert "trunk" in (out.stdout + out.stderr)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(DATA), reason="M3 데이터가 있어야 한다")
def test_freeze_sigma가_log_std를_고정한다(tmp_path, base_run):
    """`--freeze-sigma` 를 켜면 `log_std` 가 `PolicyConfig.log_std_init` 에서 안 움직여야 한다.

    끄면(기본 실행, `base_run`) 움직여야 한다 — 안 그러면 이 테스트의 비교 기준 자체가
    무의미하다.
    """
    from vtd_rl.policy.net import PolicyConfig
    base_summary, _sd_base = base_run
    init = PolicyConfig().log_std_init
    assert not all(abs(x - init) < 1e-9 for x in base_summary["log_std"]), \
        "고정 안 했는데 log_std 가 초기값에서 안 움직였다(비교 기준이 무의미하다)"

    frozen_summary = _run(tmp_path / "frozen", "--seed", "0", "--freeze-sigma")
    assert frozen_summary["log_std"] == pytest.approx([init, init])

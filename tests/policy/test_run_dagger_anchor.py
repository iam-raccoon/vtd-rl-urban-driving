"""`scripts/run_dagger.py` 에 **앵커**(참조 정책 KL)와 **회피 가중**을 배선한 자리.

M4m 이 앵커가 듣는 것을 확인했다(계수 10: Δθ 11.0→3.80, 단계 ① 0%→100%). 다만 거기서는
`scripts/probe_anchor.py` 로 **DAgger 없이** 쟀다. 여기서 잠그는 것은 그 설정이 DAgger
라운드 루프까지 **실제로 흘러가는가** 다.

★ 참조는 `--init` 체크포인트로 **고정**한다(라운드마다 안 바꾼다). M4m 이 검증한 구성이
그것이고, 라운드마다 이전 정책으로 바꾸면 참조가 누적으로 흘러가 앵커가 무엇을 묶는지
불분명해진다. 라운드마다 `--init` 에서 다시 출발하므로 "라운드마다 새로 만든 참조" 는
**가중치가 똑같다** — 그래서 가중치 비교로는 못 잡고 **같은 객체인지**로 잡는다.

★ 또 하나: 설정만 켜고 아무 일도 안 일어나는 꼴(M4e 의 `--imitation-floor 0.3`)을 막으려고
라운드 기록에 `anchor`(계수 안 곱한 KL)와 `near_frac`(가중이 실제로 걸린 비율)을 남긴다.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PY = os.path.join(REPO, ".venv", "bin", "python")
SCRIPT = os.path.join(REPO, "scripts", "run_dagger.py")


@pytest.fixture(scope="module")
def rd():
    spec = importlib.util.spec_from_file_location("run_dagger_anchor_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ 도구

def _init_ckpt(tmp_path, name="init.pt"):
    """알아볼 수 있는 가중치의 작은 **스쿼시** 체크포인트(M3 학생과 같은 종류)."""
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig
    net = DrivePolicy(PolicyConfig(trunk=(8, 8), squash=True))
    with torch.no_grad():
        for i, p in enumerate(net.parameters()):
            p.fill_(0.125 + 0.001 * i)
        net.log_std.copy_(torch.tensor([-1.5, -0.5]))
    path = tmp_path / name
    net.save(str(path))
    return path


def _ckpt_state(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)["state"]


def _fake_ev(goal_rate=1.0, mean_score=98.6):
    ep = types.SimpleNamespace(board="course_A", seed=0, outcome="goal", steps=100,
                               reward=0.0, score=mean_score, sheet=[])
    return {"goal_rate": goal_rate, "mean_score": mean_score, "mean_score_raw": mean_score,
            "mean_score_completed": mean_score, "mean_reward": 0.0, "episodes": [ep]}


class _FakeDataset:
    def __len__(self):
        return 8


#: 가짜 학습기가 돌려줄 "실제로 걸렸다" 숫자 — 설정만 켜고 0 이 나오는 꼴과 구별하려고
#: 0 이 아닌 값을 쓴다(실측 기대값은 M4m 의 11.4% 다).
FAKE_ANCHOR_KL = 0.0345
FAKE_NEAR_FRAC = 0.114

MIN = ["--rounds", "1", "--seeds", "1", "--epochs", "1", "--eval-seeds", "1",
       "--workers", "1", "--device", "cpu"]
BASE = [*MIN, "--stage", "stage1", "--eval-stage", "stage1"]


@pytest.fixture
def harness(rd, monkeypatch):
    """`main()` 을 통째로 돌리되 판·수집·평가·학습만 가짜로 바꾼다.

    가짜 학습기는 받은 그물을 **그 자리에서 흔든다** — 그래야 "참조를 이전 라운드 정책으로
    갱신한다" 는 돌연변이가 참조 가중치에 드러난다.
    """
    seen = []

    monkeypatch.setattr(rd, "collect_targets",
                        lambda paths, smoke, variants: [("(fake)", "course_X")])
    monkeypatch.setattr(rd, "_collect_job", lambda job: {"outcome": "goal"})
    monkeypatch.setattr(rd, "_stall_start_count", lambda *args, **kw: 0)
    monkeypatch.setattr(rd, "load_dir", lambda data_dir: _FakeDataset())
    monkeypatch.setattr(rd, "eval_boards", lambda names, smoke: [(n, []) for n in names])
    monkeypatch.setattr(rd, "evaluate_policy", lambda net, boards, seeds=(0,): _fake_ev())
    monkeypatch.setattr(rd, "evaluate_teacher", lambda boards, seeds=(0,): _fake_ev(1.0, 99.6))

    def fake_train(net, dataset, cfg, device=None, ref=None):
        rec = {"cfg": cfg, "ref": ref, "ref_id": id(ref), "net_id": id(net),
               "ref_state": None, "ref_grad": None, "ref_training": None}
        if ref is not None:
            rec["ref_state"] = {k: v.detach().clone().cpu() for k, v in ref.state_dict().items()}
            rec["ref_grad"] = [p.requires_grad for p in ref.parameters()]
            rec["ref_training"] = ref.training
            rec["shared"] = bool({id(p) for p in net.parameters()}
                                 & {id(p) for p in ref.parameters()})
        with torch.no_grad():                      # 학생을 흔든다 — 참조는 따라가면 안 된다
            for p in net.parameters():
                p.add_(1.0)
        seen.append(rec)
        return {"epochs": cfg.epochs, "samples": len(dataset), "loss": 0.0, "control": 0.0,
                "turn": 0.0, "anchor": FAKE_ANCHOR_KL, "near_frac": FAKE_NEAR_FRAC,
                "seconds": 0.0}

    monkeypatch.setattr(rd, "train_epochs", fake_train)
    return seen


def _run_main(rd, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", *args])
    assert rd.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _rows(out):
    with open(os.path.join(str(out), "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_cli(*args, timeout=300):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([PY, SCRIPT, *args], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=timeout)


# ----------------------------------------------------------- 기본값은 안 바뀐다

def test_기본은_앵커도_가중도_꺼져_있다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★ 계수 0·near_m 0 이면 `policy_loss` 가 그 블록을 통째로 건너뛴다 —

    과거 성적표가 전부 그 경로에서 나왔다. 참조도 **안 만든다**(만들면 체크포인트를
    한 번 더 읽고 메모리를 한 벌 더 쓴다).
    """
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"), *BASE])
    cfg = harness[0]["cfg"]
    assert (cfg.anchor_coef, cfg.near_m, cfg.near_weight) == (0.0, 0.0, 1.0)
    assert harness[0]["ref"] is None, "앵커를 안 켰는데 참조를 만들었다"
    assert summary["anchor_coef"] == 0.0


def test_기본_성적표에는_앵커_절이_없다(rd, harness, tmp_path, monkeypatch, capsys):
    """기본 실행의 성적표 문구가 예전과 한 글자도 안 달라져야 M3 성적표를 다시 만든다."""
    report = tmp_path / "r.md"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "run"), "--report", str(report), *BASE])
    text = report.read_text(encoding="utf-8")
    # 표식은 `### 앵커`·`anchor_coef` 로 본다 — 성적표에는 `--out` 경로가 그대로 박히는데
    # pytest 의 tmp_path 가 **테스트 이름**(= '앵커' 가 든)으로 만들어져서, 맨 낱말로
    # 찾으면 경로에 걸려 늘 '있다' 가 된다.
    assert "### 앵커" not in text and "anchor_coef" not in text and "near_frac" not in text


# ------------------------------------------------------------ 설정이 흘러간다

def test_앵커_계수가_학습_설정까지_흘러간다(rd, harness, tmp_path, monkeypatch, capsys):
    """★ `--anchor-coef` 를 받아 놓고 `TrainConfig` 에 안 넣으면 아무 일도 안 일어난다."""
    ckpt = _init_ckpt(tmp_path)
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(tmp_path / "run"), "--report", str(tmp_path / "r.md"),
                         "--init", str(ckpt), "--anchor-coef", "10", *BASE])
    assert harness[0]["cfg"].anchor_coef == 10.0
    assert harness[0]["ref"] is not None, "계수는 켰는데 참조를 안 넘겼다 — 앵커가 안 걸린다"
    assert summary["anchor_coef"] == 10.0


def test_회피_가중이_학습_설정까지_흘러간다(rd, harness, tmp_path, monkeypatch, capsys):
    ckpt = _init_ckpt(tmp_path)
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(tmp_path / "run"), "--report", str(tmp_path / "r.md"),
                         "--init", str(ckpt), "--anchor-coef", "10",
                         "--near-m", "30", "--near-weight", "5", *BASE])
    cfg = harness[0]["cfg"]
    assert (cfg.near_m, cfg.near_weight) == (30.0, 5.0)
    assert (summary["near_m"], summary["near_weight"]) == (30.0, 5.0)


def test_스쿼시_정렬은_앵커_설정을_안_지운다(rd, harness, tmp_path, monkeypatch, capsys):
    """`squash_aligned` 는 `dataclasses.replace` 로 새 설정을 만든다 — 그 자리에서 앵커·가중

    값을 흘리면 **스쿼시 체크포인트일 때만** 앵커가 조용히 꺼진다(M3 학생이 바로 그것이다).
    """
    ckpt = _init_ckpt(tmp_path)            # squash=True 체크포인트
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "run"), "--report", str(tmp_path / "r.md"),
               "--init", str(ckpt), "--anchor-coef", "3", "--near-m", "30",
               "--near-weight", "5", *BASE])
    cfg = harness[0]["cfg"]
    assert cfg.squash is True, "스쿼시 정렬 자체가 안 걸렸다 — 이 테스트가 공허해진다"
    assert (cfg.anchor_coef, cfg.near_m, cfg.near_weight) == (3.0, 30.0, 5.0)


# ------------------------------------------------------- 참조는 --init 에 고정이다

def test_참조는_init_체크포인트이고_라운드마다_안_바뀐다(rd, harness, tmp_path, monkeypatch,
                                                        capsys):
    """★★★ 두 가지를 한꺼번에 잠근다.

    ① **가중치**가 `--init` 체크포인트다 — 가짜 학습기가 라운드마다 학생을 흔들어도 참조는
       안 따라간다(참조를 '이전 라운드 정책' 으로 갱신하는 돌연변이가 여기서 걸린다).
    ② **같은 객체**다 — 라운드마다 `--init` 에서 다시 출발하므로 "라운드마다 새로 만든
       참조" 는 가중치가 똑같다. 그래서 ① 만으로는 못 잡고 객체 동일성이 필요하다.
    """
    ckpt = _init_ckpt(tmp_path)
    args = [a for a in BASE]
    args[args.index("--rounds") + 1] = "3"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "run"), "--report", str(tmp_path / "r.md"),
               "--init", str(ckpt), "--anchor-coef", "10", *args])
    assert len(harness) == 3
    want = _ckpt_state(ckpt)
    for rnd, rec in enumerate(harness):
        assert rec["ref"] is not None, f"라운드 {rnd} 에 참조가 안 갔다"
        assert set(rec["ref_state"]) == set(want)
        for k, v in want.items():
            assert torch.equal(rec["ref_state"][k], v), \
                f"라운드 {rnd} 의 참조 {k} 가 --init 체크포인트와 다르다 — 참조가 흘러갔다"
    assert len({rec["ref_id"] for rec in harness}) == 1, \
        "라운드마다 참조를 새로 만들었다 — --init 에 고정이 아니다"


def test_참조는_학생과_파라미터를_공유하지_않는다(rd, harness, tmp_path, monkeypatch, capsys):
    """같은 객체(또는 얕은 복사)를 넘기면 KL 이 **항상 0** 이라 앵커가 조용히 꺼진다.

    `make_reference` 가 얼린 사본을 만든다 — 여기서는 그것이 실제로 쓰였는지를 본다.
    """
    ckpt = _init_ckpt(tmp_path)
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "run"), "--report", str(tmp_path / "r.md"),
               "--init", str(ckpt), "--anchor-coef", "10", *BASE])
    rec = harness[0]
    assert rec["ref_id"] != rec["net_id"]
    assert rec["shared"] is False, "참조가 학생과 같은 파라미터 텐서를 본다"
    assert all(g is False for g in rec["ref_grad"]), "참조가 학습 가능하다"
    assert rec["ref_training"] is False, "참조가 train() 모드다"


def test_앵커를_켜고_init_을_안_주면_거부한다(rd, tmp_path, monkeypatch):
    """★ 참조가 없으면 앵커는 묶을 것이 없다 — 조용히 꺼진 채로 몇 시간 돌면 안 된다."""
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--out", str(out),
                                      "--anchor-coef", "10", *BASE])
    with pytest.raises(SystemExit):
        rd.main()
    assert not out.exists(), "거부했는데 산출물 폴더를 만들었다"


def test_dry_run_이_앵커를_띄우기_전에_검사한다(tmp_path):
    """★★ OMEN 에 띄우기 전 `--dry-run` 한 번이 인자를 걸러야 한다 — 거기서 안 걸리면

    참조 없는 앵커로 몇 시간을 태우고 성적표에서야 안다. 체크포인트 경로 오타도 마찬가지다.
    """
    bad = _run_cli("--dry-run", "--stage", "stage3", "--eval-stage", "stage1",
                   "--anchor-coef", "10")
    assert bad.returncode != 0
    assert "--init" in bad.stderr

    typo = _run_cli("--dry-run", "--stage", "stage3", "--eval-stage", "stage1",
                    "--anchor-coef", "10", "--init", str(tmp_path / "nope.pt"))
    assert typo.returncode != 0, "체크포인트 오타를 dry-run 이 안 걸렀다"

    ckpt = _init_ckpt(tmp_path)
    ok = _run_cli("--dry-run", "--stage", "stage3", "--eval-stage", "stage1",
                  "--anchor-coef", "10", "--select", "best", "--init", str(ckpt))
    assert ok.returncode == 0, ok.stderr[-2000:]
    assert "앵커" in ok.stdout and "10" in ok.stdout and str(ckpt) in ok.stdout
    assert "retain-then-learn" in ok.stdout, "무슨 기준으로 고를지가 preflight 에 없다"


def test_잘못된_앵커_가중값은_거부한다(rd, tmp_path, monkeypatch):
    ckpt = _init_ckpt(tmp_path)
    bads = [["--anchor-coef", "-1"], ["--near-m", "-5"], ["--near-weight", "0"],
            ["--near-weight", "-2"]]
    for extra in bads:
        out = tmp_path / ("x" + "".join(extra).replace("-", "_").replace(".", ""))
        monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--out", str(out),
                                          "--init", str(ckpt), *extra, *BASE])
        with pytest.raises(SystemExit):
            rd.main()
        assert not out.exists(), f"{extra} 를 거부했는데 폴더를 만들었다"


# --------------------------------------------- 켰는데 아무 일도 안 일어나는 꼴을 본다

def test_라운드_기록이_앵커_KL_과_가중_비율을_남긴다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★ M4e 의 `--imitation-floor 0.3` 은 켜 놓고 아무 일도 안 일어났다. 그 꼴을 **숫자로**

    본다 — 라운드 기록의 `anchor`(계수 안 곱한 KL)와 `near_frac`(가중이 실제로 걸린 비율)은
    **학습기가 돌려준 값**이어야 한다. 설정값을 그대로 베껴 적으면 아무것도 확인 못 한다.
    """
    ckpt = _init_ckpt(tmp_path)
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r.md"), "--init", str(ckpt),
               "--anchor-coef", "10", "--near-m", "30", "--near-weight", "5", *BASE])
    row = _rows(out)[0]
    assert row["anchor_coef"] == 10.0
    assert row["anchor"] == pytest.approx(FAKE_ANCHOR_KL)
    assert row["near_frac"] == pytest.approx(FAKE_NEAR_FRAC)
    assert (row["near_m"], row["near_weight"]) == (30.0, 5.0)


def test_성적표가_앵커가_실제로_걸렸는지_표로_낸다(rd, harness, tmp_path, monkeypatch, capsys):
    ckpt = _init_ckpt(tmp_path)
    out, report = tmp_path / "run", tmp_path / "r.md"
    args = [a for a in BASE]
    args[args.index("--rounds") + 1] = "2"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(report), "--init", str(ckpt),
               "--anchor-coef", "10", "--near-m", "30", "--near-weight", "5", *args])
    text = report.read_text(encoding="utf-8")
    # `### 앵커` 로 본다 — 맨 '앵커' 는 `--out` 경로(tmp_path 가 테스트 이름이다)에도 들어 있다.
    assert "### 앵커" in text and "anchor_coef" in text and str(ckpt) in text
    assert f"{FAKE_ANCHOR_KL:.3f}" in text, "앵커 KL 이 성적표에 없다"
    assert f"{FAKE_NEAR_FRAC * 100:.1f}%" in text, "가중 비율이 성적표에 없다"
    assert text.count(f"{FAKE_ANCHOR_KL:.3f}") >= 2, "라운드별로 안 적혔다"

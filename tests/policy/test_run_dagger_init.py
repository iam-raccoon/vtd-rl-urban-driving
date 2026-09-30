"""`scripts/run_dagger.py` — M4k 가 태운 두 결함: 웜스타트(`--init`)와 성적표 경로.

2026-09-30 OMEN 실측(`runs/omen/2026-09-30-m4k-dagger-stage3/log.jsonl`)이 둘을 같이 드러냈다.

① **웜스타트가 없다.** 라운드 0 은 β=1.0 이라 선생님만 몬 데이터인데, 그 데이터로 새로 지은
   학생이 단계 ① 완주율 **16.7%** 로 나왔다 — 같은 판에서 M3 학생은 **100%/98.60** 이다.
   DAgger 가 M3 학생을 버리고 더 어려운 판에서 운전을 처음부터 다시 배운 것이다(5 라운드로는
   못 배운다). 그래서 `--init` 으로 이미 운전할 줄 아는 체크포인트에서 출발한다.

② **성적표 기본 경로가 커밋된 문서였다.** 그 실행이 `--report` 를 안 줘서
   `docs/reports/m3-dagger.md`(M3 의 증거물)를 실패한 단계 ③ 결과로 덮어썼다. 기본값을
   `<--out>/report.md` 로 옮겨, 커밋된 문서는 경로를 **직접** 적어야만 쓰인다.

여기 테스트는 판을 하나도 안 짓는다 — 수집·평가·학습만 가짜로 바꾸고 `main()` 의 **배선**을
그대로 돌린다. 잠그려는 것이 "`--init` 이 실제로 학습이 출발하는 가중치가 되는가" 와
"어떤 인자 조합에서도 레포 문서를 안 건드리는가" 이기 때문이다.
"""
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "run_dagger.py")
COMMITTED_REPORT = os.path.join(REPO, "docs", "reports", "m3-dagger.md")


@pytest.fixture(scope="module")
def rd():
    """스크립트를 모듈로 불러온다 — `scripts/` 는 패키지가 아니라 파일 경로로 직접 로드한다."""
    spec = importlib.util.spec_from_file_location("run_dagger_init_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ 도구

def _init_ckpt(tmp_path, name="init.pt"):
    """알아볼 수 있는 가중치의 작은 **스쿼시** 체크포인트.

    새로 지은 그물과 절대 안 겹치게(값도 몸통 모양도) 만든다 — 그래야 "웜스타트가 실제로
    먹었는가" 를 가중치 하나하나로 확인할 수 있다. 스쿼시로 두는 것은 M3 학생
    (`runs/omen/2026-09-26-m3-squash-warm/policy.pt`, `squash=true`)과 같은 종류여서다.
    """
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
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n


@pytest.fixture
def harness(rd, monkeypatch):
    """`main()` 을 진짜로 돌리되 판·수집·평가·학습만 가짜로 바꾼다.

    바꾸지 **않는** 것이 이 테스트의 대상이다 — `round_start_net`, `squash_aligned`,
    체크포인트 저장, 성적표 경로 결정, 성적표 쓰기.

    `train_epochs` 가 **받은 그물의 가중치를 그 자리에서 복사**해 둔다. 그래야
    "`--init` 을 읽긴 했는데 그 뒤에 새 그물로 덮어썼다" 도 잡힌다.
    """
    seen = {"train": []}

    monkeypatch.setattr(rd, "collect_targets",
                        lambda paths, smoke, variants: [("(fake)", "course_X")])
    monkeypatch.setattr(rd, "_collect_job", lambda job: {"outcome": "goal"})
    monkeypatch.setattr(rd, "_stall_start_count", lambda *args, **kw: 0)
    monkeypatch.setattr(rd, "load_dir", lambda data_dir: _FakeDataset())
    monkeypatch.setattr(rd, "eval_boards", lambda names, smoke: [(n, []) for n in names])
    monkeypatch.setattr(rd, "evaluate_policy",
                        lambda net, boards, seeds=(0,): _fake_ev())
    monkeypatch.setattr(rd, "evaluate_teacher",
                        lambda boards, seeds=(0,): _fake_ev(1.0, 99.6))

    def fake_train(net, dataset, cfg, device=None):
        seen["train"].append({
            "state": {k: v.detach().clone().cpu() for k, v in net.state_dict().items()},
            "cfg": cfg, "squash": net.cfg.squash, "trunk": tuple(net.cfg.trunk)})
        return {"epochs": cfg.epochs, "samples": len(dataset), "loss": 0.0,
                "control": 0.0, "turn": 0.0, "seconds": 0.0}

    monkeypatch.setattr(rd, "train_epochs", fake_train)
    return seen


@pytest.fixture
def committed_report_guard():
    """커밋된 M3 성적표를 지킨다 — 실행이 덮어쓰면 **되돌리고** 실패시킨다.

    이 결함은 실제로 한 번 이 파일을 지웠다. 잠금 테스트가 같은 피해를 반복하면 안 되므로,
    감지만 하지 말고 원본 바이트로 돌려놓는다.
    """
    with open(COMMITTED_REPORT, "rb") as f:
        before = f.read()
    yield before
    with open(COMMITTED_REPORT, "rb") as f:
        after = f.read()
    if after != before:
        with open(COMMITTED_REPORT, "wb") as f:
            f.write(before)
        pytest.fail("실행이 커밋된 성적표 docs/reports/m3-dagger.md 를 덮어썼다 — 되돌렸다")


#: 단계를 **하나도 안 주는** 최소 인자 — 기본 단계(수집 ①, 평가 ①②)로 도는 실행이다.
#: 성적표 경로를 단계로 갈라 놓은 반쪽 가드는 이 실행에서만 드러난다(돌연변이 실험에서
#: 실제로 살아남았다: `--stage` 를 줄 때만 안전하고 안 주면 커밋 문서로 가는 가드).
MIN = ["--rounds", "1", "--seeds", "1", "--epochs", "1", "--eval-seeds", "1",
       "--workers", "1", "--device", "cpu"]
BASE = [*MIN, "--stage", "stage1", "--eval-stage", "stage1"]


def _run_main(rd, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", *args])
    assert rd.main() == 0
    printed = capsys.readouterr().out.strip().splitlines()
    return json.loads(printed[-1])


# ----------------------------------------------------- 결함 ① 웜스타트 `--init`

def test_init_없으면_예전_그대로_새_그물이다(rd):
    """★ 기본값이 바뀌면 커밋된 M3 성적표가 재현되지 않는다 — 그 성적표는 새 그물로 나왔다."""
    torch.manual_seed(7)
    fresh = rd._new_net(False, torch.device("cpu"))
    torch.manual_seed(7)
    got = rd.round_start_net(None, False, torch.device("cpu"))
    assert got.cfg == fresh.cfg
    assert tuple(got.cfg.trunk) == (256, 256) and got.cfg.squash is False
    for k, v in fresh.state_dict().items():
        assert torch.equal(got.state_dict()[k], v), k
    assert torch.allclose(got.log_std.detach(), torch.tensor([-1.0, -1.0]))


def test_init_을_주면_그_체크포인트_가중치에서_출발한다(rd, tmp_path):
    """★★ 웜스타트의 핵심 — 출발 가중치가 체크포인트와 **한 텐서도 안 다르게** 같아야 한다."""
    ckpt = _init_ckpt(tmp_path)
    net = rd.round_start_net(str(ckpt), False, torch.device("cpu"))
    want = _ckpt_state(ckpt)
    assert set(net.state_dict()) == set(want)
    for k, v in want.items():
        assert torch.equal(net.state_dict()[k].cpu(), v), k
    assert net.cfg.squash is True, "체크포인트의 매개화(squash)를 안 따라갔다"
    assert tuple(net.cfg.trunk) == (8, 8), "몸통 모양은 체크포인트가 정한다"


def test_init_은_smoke_의_작은_몸통을_안_쓴다(rd, tmp_path):
    """`--smoke` 는 몸통을 (64,64) 로 줄이지만, 체크포인트를 읽을 때는 모양이 거기서 온다."""
    ckpt = _init_ckpt(tmp_path)
    assert tuple(rd.round_start_net(str(ckpt), True, torch.device("cpu")).cfg.trunk) == (8, 8)
    assert tuple(rd.round_start_net(None, True, torch.device("cpu")).cfg.trunk) == (64, 64)


def test_라운드_학습은_init_가중치에서_출발한다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★★ `main()` 배선 통째로 — 학습기가 **실제로 받은** 그물이 `--init` 체크포인트인가.

    돌연변이 두 가지를 같이 잡는다: `--init` 을 조용히 무시하는 것과, 읽어 놓고 그 뒤에
    새 그물로 덮어쓰는 것. 둘 다 "가중치가 체크포인트와 같다" 에서 걸린다.
    """
    ckpt = _init_ckpt(tmp_path)
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--init", str(ckpt),
                         "--report", str(tmp_path / "r.md"), *BASE])
    assert summary["init"] == str(ckpt)
    assert len(harness["train"]) == 1
    started, want = harness["train"][0]["state"], _ckpt_state(ckpt)
    assert set(started) == set(want), "학습이 받은 그물이 체크포인트와 모양부터 다르다"
    for k, v in want.items():
        assert torch.equal(started[k], v), f"{k} 가 체크포인트 값이 아니다 — 웜스타트가 안 먹었다"


def test_init_없는_실행은_체크포인트를_안_쓴다(rd, harness, tmp_path, monkeypatch, capsys):
    """기본 경로는 안 바뀐다 — 학습이 받는 그물은 새 그물(몸통 256, log_std -1.0)이다."""
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"), *BASE])
    assert summary["init"] is None
    assert harness["train"][0]["trunk"] == (256, 256)
    assert harness["train"][0]["squash"] is False
    assert torch.allclose(harness["train"][0]["state"]["log_std"], torch.tensor([-1.0, -1.0]))


def test_모든_라운드가_같은_init_에서_출발한다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★ 라운드 0 만 웜스타트하면 라운드 1 에서 도로 버려진다 — 결함이 반만 고쳐진다.

    라운드마다 **같은 고정 체크포인트**에서 다시 출발하므로 에폭은 안 쌓인다(이전 라운드
    그물을 이어 쓰던 옛 결함과 다르다 — `run_dagger.py` 학습 루프 주석 참고).
    """
    ckpt = _init_ckpt(tmp_path)
    out = tmp_path / "run"
    args = [a for a in BASE]
    args[args.index("--rounds") + 1] = "3"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--init", str(ckpt),
               "--report", str(tmp_path / "r.md"), *args])
    assert len(harness["train"]) == 3
    want = _ckpt_state(ckpt)
    for rnd, rec in enumerate(harness["train"]):
        for k, v in want.items():
            assert torch.equal(rec["state"][k], v), f"라운드 {rnd} 의 {k} 가 체크포인트와 다르다"


def test_init_이_스쿼시면_손실도_스쿼시로_간다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★ tanh 정책을 clamp 가능도로 학습하면 웜스타트가 **조용히** 망가진다.

    M3 학생(`2026-09-26-m3-squash-warm/policy.pt`)이 바로 `squash=true` 다. 손실 설정을
    그물에 맞추는 것은 `train.squash_aligned` 의 일이고, 여기서는 그것이 실제로 불렸는지를
    학습기가 받은 `TrainConfig` 로 확인한다.
    """
    ckpt = _init_ckpt(tmp_path)
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--init", str(ckpt),
               "--report", str(tmp_path / "r.md"), *BASE])
    assert harness["train"][0]["cfg"].squash is True, "스쿼시 그물인데 손실은 clamp 가능도다"


def test_init_없는_실행의_손실은_예전_그대로_비스쿼시다(rd, harness, tmp_path, monkeypatch,
                                                      capsys):
    out = tmp_path / "run"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(out), "--report", str(tmp_path / "r.md"), *BASE])
    assert harness["train"][0]["cfg"].squash is False


def test_없는_init_경로는_아무것도_하기_전에_거부한다(rd, tmp_path, monkeypatch):
    """오타 하나로 웜스타트가 조용히 빠지면 몇 시간 뒤 성적표에서야 드러난다."""
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--out", str(out),
                                      "--init", str(tmp_path / "nope.pt"), *BASE])
    with pytest.raises(SystemExit):
        rd.main()
    assert not out.exists(), "거부했는데 산출물 폴더를 만들었다"


# ------------------------------------------------- 결함 ② 성적표가 커밋된 문서를 덮어쓴다

def test_기본_성적표_경로는_산출물_폴더_안이다(rd):
    assert rd.resolve_report(None, "/tmp/run") == os.path.join("/tmp/run", "report.md")
    assert rd.resolve_report("", "/tmp/run") == os.path.join("/tmp/run", "report.md")
    assert rd.resolve_report("docs/x.md", "/tmp/run") == "docs/x.md"
    assert not os.path.abspath(rd.resolve_report(None, "/tmp/run")).startswith(
        os.path.join(REPO, "docs")), "기본 성적표가 레포 문서를 가리킨다"


def test_단계를_안_준_기본_실행도_커밋된_성적표를_안_건드린다(
        rd, harness, tmp_path, monkeypatch, capsys, committed_report_guard):
    """★★★ 바로 이 결함 — 2026-09-30 실행이 여기서 `docs/reports/m3-dagger.md` 를 덮어썼다.

    **단계를 하나도 안 준다**(기본 단계 실행). 이 결함을 "기본 단계가 아닐 때만 막는다" 로
    반만 고치면 그 반쪽 가드가 바로 이 실행에서 샌다 — 돌연변이 실험에서 실제로 살아남았고,
    그래서 이 테스트는 `--stage` 를 주는 아래 테스트와 **둘 다** 있어야 한다.
    """
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys, ["--out", str(out), *MIN])
    assert summary["report"] == str(out / "report.md")
    assert (out / "report.md").exists()
    with open(COMMITTED_REPORT, "rb") as f:
        assert f.read() == committed_report_guard, "커밋된 M3 성적표가 바뀌었다"


def test_stage_를_줘도_기본_성적표는_여전히_out_아래다(
        rd, harness, tmp_path, monkeypatch, capsys, committed_report_guard):
    """★★ "기본 단계일 때만 커밋 문서를 쓴다" 식의 반쪽 가드를 막는다 — 단계와 무관해야 한다.

    (M4k 실행이 바로 `--stage stage3` 였다. 단계로 갈라 놓으면 기본 단계 실행이 또 덮어쓴다.)
    """
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--rounds", "1", "--seeds", "1", "--epochs", "1",
                         "--eval-seeds", "1", "--workers", "1", "--device", "cpu",
                         "--stage", "stage3", "--eval-stage", "stage1"])
    assert summary["report"] == str(out / "report.md")
    with open(COMMITTED_REPORT, "rb") as f:
        assert f.read() == committed_report_guard


def test_report_를_직접_주면_그리로_쓴다(rd, harness, tmp_path, monkeypatch, capsys):
    """커밋된 M3 성적표는 **일부러** 다시 만들 수 있어야 한다 — 경로를 적으면 그리로 간다."""
    out, chosen = tmp_path / "run", tmp_path / "m3.md"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(chosen), *BASE])
    assert summary["report"] == str(chosen)
    assert chosen.exists() and "DAgger" in chosen.read_text(encoding="utf-8")
    assert not (out / "report.md").exists(), "직접 준 경로 말고 기본 경로에도 썼다"


def test_init_을_준_성적표는_출발점을_밝힌다(rd, harness, tmp_path, monkeypatch, capsys):
    """웜스타트한 성적표의 완주율은 '처음부터 배운 결과' 가 아니다 — 그 사실이 문서에 있어야 한다.

    반대로 `--init` 없는 기본 실행은 이 줄이 **없어야** 한다(커밋된 M3 성적표가 그대로 재현된다).
    """
    ckpt = _init_ckpt(tmp_path)
    warm, cold = tmp_path / "warm.md", tmp_path / "cold.md"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "w"), "--init", str(ckpt), "--report", str(warm), *BASE])
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "c"), "--report", str(cold), *BASE])
    # "웜스타트" 라는 낱말만으로는 못 가른다 — 성적표 꼬리말이 "PPO 를 웜스타트하면…" 을
    # 이미 쓰고 있다. 이 줄만의 표식(`- 출발점:`)으로 본다.
    warm_text, cold_text = warm.read_text(encoding="utf-8"), cold.read_text(encoding="utf-8")
    assert "- 출발점:" in warm_text and str(ckpt) in warm_text
    assert "- 출발점:" not in cold_text, \
        "기본 실행의 성적표 문구가 달라졌다 — 커밋된 M3 성적표를 못 재현한다"

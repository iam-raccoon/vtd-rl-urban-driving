"""`scripts/run_dagger.py` 의 **학습 시드** — 배치 순서를 실제로 흔드는 손잡이.

M4l 실측: 같은 데이터·같은 설정에서 **배치 순서 시드만 바꿔도** 단계 ① 완주율이
16.7% ↔ 66.7% 로 흔들리는데 손실·MAE·Δθ 는 소수 셋째 자리까지 같다. 그래서 M4m·M4n 은
설정마다 **학습 시드 3 개**를 돌리고 평균과 범위를 같이 냈고, M4o 도 그래야 한다.

★ 그런데 `--seed` 로는 그게 안 된다. `--seed` 는 `torch.manual_seed(a.seed + rnd)` 로
그물 **초기값**을 흔들려는 것인데, `--init` 을 주면 가중치를 체크포인트에서 읽으므로
전역 RNG 를 쓰지 않는다 — 효과가 **0** 이다. 수집도 안 흔들린다(`_collect_job` 이
`seed=1000*rnd+s` 를 쓰고 spawn 프로세스라 부모 전역 시드가 닿지도 않는다). 배치 순서는
`train_epochs` 안의 `torch.Generator().manual_seed(cfg.seed)` 이고, `cfg.seed` 는
**라운드 번호**였다. `--train-seed` 가 그 밑값을 연다.

여기 테스트는 판을 하나도 안 짓는다 — 학습기가 **실제로 받은** `TrainConfig.seed` 를 본다.
"""
import importlib.util
import json
import os
import sys
import types

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "run_dagger.py")


@pytest.fixture(scope="module")
def rd():
    spec = importlib.util.spec_from_file_location("run_dagger_seed_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_ev(goal_rate=1.0, mean_score=98.6):
    ep = types.SimpleNamespace(board="course_A", seed=0, outcome="goal", steps=100,
                               reward=0.0, score=mean_score, sheet=[])
    return {"goal_rate": goal_rate, "mean_score": mean_score, "mean_score_raw": mean_score,
            "mean_score_completed": mean_score, "mean_reward": 0.0, "episodes": [ep]}


class _FakeDataset:
    def __len__(self):
        return 8


MIN = ["--rounds", "3", "--seeds", "1", "--epochs", "1", "--eval-seeds", "1",
       "--workers", "1", "--device", "cpu", "--stage", "stage1", "--eval-stage", "stage1"]


@pytest.fixture
def harness(rd, monkeypatch):
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
        seen.append(cfg)
        return {"epochs": cfg.epochs, "samples": len(dataset), "loss": 0.0, "control": 0.0,
                "turn": 0.0, "anchor": 0.0, "near_frac": 0.0, "seconds": 0.0}

    monkeypatch.setattr(rd, "train_epochs", fake_train)
    return seen


def _run_main(rd, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", *args])
    assert rd.main() == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _rows(out):
    with open(os.path.join(str(out), "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# -------------------------------------------------------------- 기본값 동일성

def test_기본은_예전_그대로_라운드_번호가_배치_시드다(rd, harness, tmp_path, monkeypatch,
                                                      capsys):
    """★★ 과거 성적표(M3·M4k·M4m)가 전부 `cfg.seed == rnd` 에서 나왔다.

    `--train-seed` 기본값 0 은 `0 + rnd == rnd` 라 **비트 단위로 예전과 같다.**
    """
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"), *MIN])
    assert [cfg.seed for cfg in harness] == [0, 1, 2]
    assert summary["train_seed"] == 0
    assert [r["train_seed"] for r in _rows(out)] == [0, 1, 2]


def test_기본_성적표에는_배치_시드_줄이_없다(rd, harness, tmp_path, monkeypatch, capsys):
    report = tmp_path / "r.md"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "run"), "--report", str(report), *MIN])
    assert "- 배치 순서 시드:" not in report.read_text(encoding="utf-8")


# -------------------------------------------------- 실제로 배치 순서를 흔드는가

def test_train_seed_는_라운드마다_더해진다(rd, harness, tmp_path, monkeypatch, capsys):
    """★★★ `train-seed + rnd` 다. **덮어쓰면 안 된다** — 모든 라운드가 같은 배치 순서가

    되면 라운드끼리 비교가 다른 실험이 되고, 예전 기본 동작(`rnd`)과도 어긋난다.
    """
    out = tmp_path / "run"
    summary = _run_main(rd, monkeypatch, capsys,
                        ["--out", str(out), "--report", str(tmp_path / "r.md"),
                         "--train-seed", "100", *MIN])
    assert [cfg.seed for cfg in harness] == [100, 101, 102]
    assert summary["train_seed"] == 100
    assert [r["train_seed"] for r in _rows(out)] == [100, 101, 102]


def test_시드가_다르면_배치_순서도_다르다(rd, harness, tmp_path, monkeypatch, capsys):
    """★ 계획서가 요구하는 '학습 시드 3 개' 가 실제로 **서로 다른 세 실행**이 되는가.

    세 번 돌려 받은 `cfg.seed` 묶음이 하나도 안 겹쳐야 한다 — 겹치면 "시드 3 개" 는
    성적표의 거짓말이 된다.
    """
    sets = []
    for s in ("0", "1", "2"):
        harness.clear()
        _run_main(rd, monkeypatch, capsys,
                  ["--out", str(tmp_path / f"run{s}"), "--report", str(tmp_path / f"{s}.md"),
                   "--train-seed", s, *MIN])
        sets.append([cfg.seed for cfg in harness])
    assert sets == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert len({tuple(x) for x in sets}) == 3, "학습 시드를 바꿨는데 배치 순서가 그대로다"


def test_seed_는_배치_순서를_안_건드린다(rd, harness, tmp_path, monkeypatch, capsys):
    """★ `--seed` 와 `--train-seed` 는 **다른 손잡이**다.

    `--seed` 는 그물 초기값용이고, `--init` 을 주면 가중치를 체크포인트에서 읽어 효과가
    없다. 그 둘을 한 인자로 합치면 "시드 3 개로 돌렸다" 가 거짓이 된다 — 여기서 갈라 둔다.
    """
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "run"), "--report", str(tmp_path / "r.md"),
               "--seed", "77", *MIN])
    assert [cfg.seed for cfg in harness] == [0, 1, 2], "--seed 가 배치 순서까지 바꿨다"


def test_성적표가_배치_시드를_밝힌다(rd, harness, tmp_path, monkeypatch, capsys):
    """시드 하나짜리 성적표를 '그 설정의 결과' 로 읽으면 안 된다 — 문서가 그렇게 말해야 한다."""
    report = tmp_path / "r.md"
    _run_main(rd, monkeypatch, capsys,
              ["--out", str(tmp_path / "run"), "--report", str(report),
               "--train-seed", "2", *MIN])
    text = report.read_text(encoding="utf-8")
    assert "- 배치 순서 시드:" in text and "2 + rnd" in text


def test_dry_run_이_배치_시드를_보여_준다(rd, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_dagger.py", "--dry-run", "--stage", "stage1",
                                      "--eval-stage", "stage1", "--train-seed", "2"])
    assert rd.main() == 0
    out = capsys.readouterr().out
    assert "배치 순서 시드" in out and "2" in out

"""`scripts/probe_anchor.py` 의 의미를 잠근다 — **빠른 조각**에 둔다.

이 스크립트가 내는 표가 M4m 의 결론을 떠받친다. 그래서 잠그는 것은 숫자 서식이 아니라
결론을 바꿀 수 있는 성질들이다:

- 칸마다 **체크포인트를 새로 읽는다**(그물을 돌려 쓰면 계수 사이에 학습이 쌓인다).
- 참조가 **얼린 사본**이고 학습 뒤에도 안 움직인다(같은 객체면 앵커가 조용히 꺼진다).
- `anchor_coef`·`seed`·`anchor_lr` 이 **실제로 학습에 닿는다**(필드만 채우고 배선이 끊긴 꼴을 막는다).
- 손실의 `squash` 가 **그물을 따라간다**(M3 체크포인트는 `squash=True` 다).
- 표에 **평균과 범위가 같이** 나온다(단일 숫자는 뜻이 없다 — M4l).
- 표가 **판정하지 않는다**(합격·불합격을 안 찍는다).

판을 안 돌린다 — `run_cell` 의 평가를 주입해 몇 초 안에 끝낸다.
"""
import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest
import torch

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load():
    path = os.path.join(REPO, "scripts", "probe_anchor.py")
    spec = importlib.util.spec_from_file_location("probe_anchor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_dataset(n=256, seed=0):
    """`tests/policy/test_train.py::toy_dataset` 과 같은 장난감 자료(판을 안 돌린다)."""
    from vtd_rl.policy.dataset import DaggerDataset, Shard
    from vtd_rl.policy.encode import OBJ_DIM, OBJ_N, VEC_DIM
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((n, VEC_DIM)).astype(np.float32)
    control = np.stack([np.tanh(vec[:, 0]), np.tanh(vec[:, 1])], axis=1).astype(np.float32)
    turn = (vec[:, 2] > 0).astype(np.int64) + (vec[:, 3] > 0.8).astype(np.int64)
    ds = DaggerDataset()
    ds.add(Shard(vec, np.zeros((n, OBJ_N, OBJ_DIM), np.float32),
                 np.zeros((n, OBJ_N), np.float32), control, turn, {"board": "toy"}))
    return ds


def _checkpoint(tmp_path, squash=False, seed=0):
    from vtd_rl.policy.net import DrivePolicy, PolicyConfig
    torch.manual_seed(seed)
    path = tmp_path / f"init-{'sq' if squash else 'ns'}.pt"
    DrivePolicy(PolicyConfig(trunk=(32, 32), squash=squash)).save(str(path))
    return str(path)


def _fake_eval(goal=0.5):
    """판을 안 돌리는 평가 — 호출될 때마다 그물을 기록한다."""
    seen = []

    def evaluate(net):
        seen.append(net)
        return {"stage1": {"goal_rate": goal, "mean_score": 90.0,
                           "mean_score_completed": 95.0, "mean_reward": 10.0},
                "stage3": {"goal_rate": 0.0, "mean_score": 50.0,
                           "mean_score_completed": None, "mean_reward": 1.0}}
    return evaluate, seen


# --- 인자 파싱 -------------------------------------------------------------

def test_계수와_시드를_쉼표로_읽는다():
    m = _load()
    assert m.parse_floats("0,0.1,1,10") == [0.0, 0.1, 1.0, 10.0]
    assert m.parse_floats(" 0 , 1 ") == [0.0, 1.0]
    assert m.parse_ints("0,1,2") == [0, 1, 2]
    for bad in ("", " ", ","):
        with pytest.raises(ValueError):
            m.parse_floats(bad)


def test_기본_격자는_계수_넷_곱하기_시드_셋이다():
    """계획서가 정한 격자다 — 시드가 하나로 줄면 M4l 의 잡음 안에서 결론을 내게 된다."""
    m = _load()
    assert m.parse_floats(m.DEFAULT_COEFS) == [0.0, 0.1, 1.0, 10.0]
    assert len(m.parse_ints(m.DEFAULT_SEEDS)) >= 3
    assert m.DEFAULT_EVAL_STAGES == ["stage1", "stage3"]      # 유지 + 학습을 나란히


def test_모르는_단계는_이름을_말하며_거부한다():
    m = _load()
    with pytest.raises(ValueError, match="stage9"):
        m.stage_path("stage9")
    assert m.stage_path("stage1").endswith(os.path.join("curricula", "stage1.json"))


# --- 격자 한 칸 ------------------------------------------------------------

def test_칸이_체크포인트를_새로_읽는다(tmp_path):
    """★ 그물을 돌려 쓰면 계수 사이에 학습이 쌓여 '큰 계수가 덜 움직였다' 가 순서 탓이 된다."""
    from vtd_rl.policy.net import DrivePolicy
    from vtd_rl.policy.train import TrainConfig
    m = _load()
    init = _checkpoint(tmp_path)
    before = DrivePolicy.load(init).state_dict()
    ds = _toy_dataset()
    evaluate, seen = _fake_eval()
    dev = torch.device("cpu")

    rows = [m.run_cell(init, ds, TrainConfig(epochs=1, seed=0, lr=1e-2, anchor_coef=0.0),
                       dev, evaluate) for _ in range(2)]
    assert rows[0]["delta_theta"] == pytest.approx(rows[1]["delta_theta"], rel=1e-9)
    # 체크포인트 파일도 안 건드린다
    after = DrivePolicy.load(init).state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert len(seen) == 2 and seen[0] is not seen[1]


def test_칸이_얼린_참조를_쓰고_참조는_안_움직인다(tmp_path):
    """참조가 학생과 같은 객체면 KL 이 항상 0 이라 앵커가 조용히 꺼진다.

    `policy_loss` 가 그 경우를 `ValueError` 로 거부하므로, 앵커를 켠 칸이 **그냥 도는 것**
    자체가 참조가 별개 사본이라는 증거다. 그 위에 앵커 KL 이 실제로 0 이 아니었음을 본다.

    ★ 배치가 여럿이어야 한다. 출발점에서는 학생 = 참조라 KL 도 그 기울기도 **정확히 0** 이다 —
    최적화 한 걸음짜리 칸은 앵커가 켜졌는지 꺼졌는지 구별이 안 된다(실제 실험은 단계 ③
    36k 행을 256 배치로 돌아 140 걸음이 넘는다).
    """
    from vtd_rl.policy.train import TrainConfig
    m = _load()
    evaluate, _seen = _fake_eval()
    row = m.run_cell(_checkpoint(tmp_path), _toy_dataset(),
                     TrainConfig(epochs=1, seed=0, lr=1e-2, batch_size=32, anchor_coef=1.0),
                     torch.device("cpu"), evaluate)
    assert row["anchor"] > 0.0, "앵커 KL 이 0 이면 참조가 학생을 그대로 따라간 것이다"


def test_anchor_coef가_실제로_학습에_닿는다(tmp_path):
    """★ 칸 설정에 계수만 실리고 손실까지 안 가면 표 전체가 같은 실험 네 번이 된다."""
    from vtd_rl.policy.train import TrainConfig
    m = _load()
    init, ds = _checkpoint(tmp_path), _toy_dataset()
    evaluate, _seen = _fake_eval()
    deltas = []
    for coef in (0.0, 1.0, 10.0):
        row = m.run_cell(init, ds,
                         TrainConfig(epochs=2, seed=0, lr=1e-2, batch_size=32, anchor_coef=coef),
                         torch.device("cpu"), evaluate)
        assert row["anchor_coef"] == coef
        deltas.append(row["delta_theta"])
    assert deltas[0] > deltas[1] > deltas[2], deltas


def test_학습_시드가_실제로_배치_순서를_바꾼다(tmp_path):
    """시드가 안 먹으면 '시드 3 개 평균' 이 같은 숫자 세 개의 평균이 된다 — 범위가 늘 0 이다."""
    from vtd_rl.policy.train import TrainConfig
    m = _load()
    init, ds = _checkpoint(tmp_path), _toy_dataset()
    evaluate, _seen = _fake_eval()
    out = [m.run_cell(init, ds, TrainConfig(epochs=1, seed=s, lr=1e-2, batch_size=32),
                      torch.device("cpu"), evaluate) for s in (0, 1)]
    assert out[0]["seed"] == 0 and out[1]["seed"] == 1
    assert out[0]["delta_theta"] != pytest.approx(out[1]["delta_theta"], rel=1e-9)


def test_anchor_lr이_칸의_학습률이_된다(tmp_path):
    from vtd_rl.policy.train import TrainConfig
    m = _load()
    init, ds = _checkpoint(tmp_path), _toy_dataset()
    evaluate, _seen = _fake_eval()
    base = m.run_cell(init, ds, TrainConfig(epochs=1, seed=0, lr=1e-2),
                      torch.device("cpu"), evaluate)
    low = m.run_cell(init, ds, TrainConfig(epochs=1, seed=0, lr=1e-2, anchor_lr=0.0),
                     torch.device("cpu"), evaluate)
    assert base["lr"] == 1e-2 and low["lr"] == 0.0
    assert base["delta_theta"] > 0.0 and low["delta_theta"] == 0.0


def test_손실의_스쿼시가_그물을_따라간다(tmp_path):
    """★ M3 체크포인트는 `squash=True` 다. 안 맞추면 tanh 정책을 clamp 가능도로 학습해

    실험 전체가 엉뚱한 것을 잰다(M4l 이 H3 으로 따로 확인한 자리). 칸이 쓴 설정을 행에 남겨
    나중에도 확인할 수 있게 한다.
    """
    from vtd_rl.policy.train import TrainConfig
    m = _load()
    ds = _toy_dataset()
    evaluate, _seen = _fake_eval()
    cfg = TrainConfig(epochs=1, seed=0, lr=1e-2)              # squash=False 가 기본
    assert m.run_cell(_checkpoint(tmp_path, squash=True), ds, cfg,
                      torch.device("cpu"), evaluate)["squash"] is True
    assert m.run_cell(_checkpoint(tmp_path, squash=False), ds, cfg,
                      torch.device("cpu"), evaluate)["squash"] is False


# --- 요약과 표 -------------------------------------------------------------

def test_요약은_평균과_범위를_같이_낸다():
    m = _load()
    s = m.summarize([0.0, 0.5, 1.0])
    assert (s["n"], s["mean"], s["min"], s["max"]) == (3, pytest.approx(0.5), 0.0, 1.0)
    # `mean_score_completed` 는 완주 판이 없으면 None 이다 — 0 으로 세면 안 된다
    s2 = m.summarize([None, 1.0, None])
    assert (s2["n"], s2["mean"], s2["min"], s2["max"]) == (1, 1.0, 1.0, 1.0)
    s3 = m.summarize([None, None])
    assert s3 == {"n": 0, "mean": None, "min": None, "max": None}


def _rows(spec):
    """`spec` = [(계수, 시드, 단계① 완주율, 단계③ 완주율), ...] → 행 목록."""
    return [{"anchor_coef": c, "seed": s, "delta_theta": 0.5, "loss": -1.0, "anchor": 0.25,
             "stages": {"stage1": {"goal_rate": g1, "mean_score": 90.0},
                        "stage3": {"goal_rate": g3, "mean_score": 50.0}}}
            for c, s, g1, g3 in spec]


def test_표는_설정마다_평균과_범위를_같이_찍는다():
    """★ M4l: 배치 순서 시드만 바꿔도 16.7% ↔ 66.7% 다. 평균만 찍으면 그 폭이 사라진다."""
    m = _load()
    rows = _rows([(0.0, 0, 1 / 6, 0.0), (0.0, 1, 4 / 6, 0.0), (0.0, 2, 3 / 6, 0.0)])
    line = [ln for ln in m.table_lines(rows, ["stage1", "stage3"]) if ln.startswith("| 0 |")][0]
    # 평균 (1/6 + 4/6 + 3/6)/3 = 0.4444 → 44.4%, 범위는 16.7–66.7
    assert "44.4%" in line and "16.7–66.7" in line, line
    assert "(n=3)" in line and "0,1,2" in line, line


def test_표는_유지와_학습을_나란히_놓는다():
    """단계 ① 만 보면 아무것도 안 배운 정책이 만점이고, 단계 ③ 만 보면 옛 능력을 버린 정책이 만점이다."""
    m = _load()
    lines = m.table_lines(_rows([(1.0, 0, 1.0, 0.5)]), ["stage1", "stage3"])
    assert "단계 ①" in lines[0] and "단계 ③" in lines[0]
    assert lines[0].index("단계 ①") < lines[0].index("단계 ③")
    row = [ln for ln in lines if ln.startswith("| 1 |")][0]
    assert "100.0%" in row and "50.0%" in row, row


def test_표는_판정하지_않는다():
    """★ 문턱은 계획서에 미리 적혀 있다. 스크립트가 합격을 찍으면 결과를 보고 문턱을 고치게 된다."""
    m = _load()
    rows = _rows([(0.0, 0, 1.0, 0.5), (10.0, 0, 0.0, 0.0)])
    text = "\n".join(m.table_lines(rows, ["stage1", "stage3"])
                     + m.caveat_lines(rows, 3, ["stage1", "stage3"]))
    for word in ("합격", "불합격", "달성", "미달", "듣는다", "PASS", "FAIL"):
        assert word not in text, word


def test_계수별로_묶되_순서를_지킨다():
    m = _load()
    rows = _rows([(0.0, 0, 1.0, 0.0), (10.0, 0, 1.0, 0.0), (0.0, 1, 1.0, 0.0)])
    assert [c for c, _b in m.group_by_coef(rows)] == [0.0, 10.0]
    assert [len(b) for _c, b in m.group_by_coef(rows)] == [2, 1]


def test_시드별_원자료를_같이_찍는다():
    """평균 뒤에 무엇이 있는지 숨기지 않는다 — 범위가 선을 가로지르는지 보려면 원자료가 필요하다."""
    m = _load()
    rows = _rows([(1.0, 0, 1 / 6, 0.0), (1.0, 1, 4 / 6, 0.0)])
    text = "\n".join(m.per_seed_lines(rows, ["stage1", "stage3"]))
    assert "16.7%" in text and "66.7%" in text


def test_주의문구가_표본의_한계를_적는다():
    """성적표가 그대로 옮겨 쓸 문장이다 — 단계 ① 독립 판 6 개와 학습 비결정성."""
    m = _load()
    text = "\n".join(m.caveat_lines(_rows([(0.0, 0, 1.0, 0.0)]), 3, ["stage1", "stage3"]))
    assert "6" in text and "16.7" in text and "66.7" in text


# --- 명령줄 ----------------------------------------------------------------

def _cli(*args, timeout=120):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([os.path.join(REPO, ".venv", "bin", "python"),
                           os.path.join(REPO, "scripts", "probe_anchor.py"), *args],
                          capture_output=True, text=True, env=env, cwd=REPO, timeout=timeout)


def test_dry_run이_무엇을_돌릴지_찍는다(tmp_path):
    out = _cli("--dry-run", "--init", _checkpoint(tmp_path), "--data", str(tmp_path / "없음"))
    assert out.returncode == 0, out.stderr[-3000:]
    assert "12 칸" in out.stdout, out.stdout            # 계수 4 × 시드 3
    assert "stage1" in out.stdout and "stage3" in out.stdout
    # ★ 단계 ③ 평가 판에 액터가 실제로 들어 있는지 — `run_dagger.py` 의 `--smoke` 가 액터를
    # 통째로 버려 "아무것도 시험하지 않는" 실행이 됐던 전례가 있다.
    s3 = [ln for ln in out.stdout.splitlines() if "평가 stage3:" in ln][0]
    actors = int(s3.split("액터 ")[1].split("개")[0])
    assert actors > 0, s3
    s1 = [ln for ln in out.stdout.splitlines() if "평가 stage1:" in ln][0]
    assert "액터 0개" in s1, s1                         # 단계 ① 은 액터가 없다(시드가 무의미한 이유)


def test_출력_파일을_덮어쓰지_않는다(tmp_path):
    """두 실행의 행이 섞이면 어느 숫자가 어느 실행인지 알 수 없다."""
    existing = tmp_path / "rows.jsonl"
    existing.write_text("{}\n", encoding="utf-8")
    out = _cli("--init", _checkpoint(tmp_path), "--data", str(tmp_path),
               "--out", str(existing))
    assert out.returncode != 0 and "이미" in (out.stdout + out.stderr)


def test_데이터가_비면_거부한다(tmp_path):
    out = _cli("--init", _checkpoint(tmp_path), "--data", str(tmp_path / "없음"))
    assert out.returncode != 0 and "행이 없다" in (out.stdout + out.stderr)


def test_체크포인트가_없으면_거부한다(tmp_path):
    out = _cli("--init", str(tmp_path / "없음.pt"), "--data", str(tmp_path))
    assert out.returncode != 0 and "체크포인트" in (out.stdout + out.stderr)


def test_한_번의_실행이_표까지_낸다(tmp_path, monkeypatch, capsys):
    """★ OMEN 에서 **명령 하나**로 성적표가 필요로 하는 표가 나와야 한다.

    평가만 가로채고(판을 안 돌린다) 나머지는 진짜로 돌린다 — 격자 순회, 기준점, `--out`
    JSONL, 표 조립까지. 여기서 안 보면 OMEN 에서 몇 시간 돌린 끝에 표 조립에서 터진다.
    """
    import sys
    from vtd_rl.policy.dataset import Shard, save_shard
    m = _load()
    ds = _toy_dataset(128)
    save_shard(Shard(*ds.arrays(), {"board": "toy"}), str(tmp_path / "data" / "r0-toy-s0.npz"))

    calls = []

    def fake_eval_all(net, stages, eval_seeds):
        calls.append(eval_seeds)
        return {name: {"goal_rate": 1.0 / (len(calls) % 3 + 1), "mean_score": 90.0,
                       "mean_score_completed": 95.0, "mean_reward": 10.0}
                for name, _b in stages}

    monkeypatch.setattr(m, "evaluate_all", fake_eval_all)
    out_path = tmp_path / "rows.jsonl"
    argv = sys.argv
    sys.argv = ["probe_anchor.py", "--init", _checkpoint(tmp_path), "--data",
                str(tmp_path / "data"), "--coefs", "0,1", "--seeds", "0,1",
                "--epochs", "1", "--eval-seeds", "3", "--out", str(out_path)]
    try:
        assert m.main() == 0
    finally:
        sys.argv = argv

    text = capsys.readouterr().out
    assert "anchor_coef" in text and "단계 ①" in text and "단계 ③" in text
    assert "— (학습 전) |" in text                     # 기준점 행
    assert "시드별 원자료" in text and "주장하지 않는 것" in text
    rows = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(rows) == 5                               # 기준점 1 + 칸 4
    assert [(r["anchor_coef"], r["seed"]) for r in rows[1:]] == [(0.0, 0), (0.0, 1),
                                                                 (1.0, 0), (1.0, 1)]
    assert calls == [3] * 5                             # --eval-seeds 가 평가까지 흘렀다


def test_원자료_한_줄이_한_칸이다(tmp_path, monkeypatch):
    """`--out` JSONL 이 표의 원천이다 — 행에 계수·시드·Δθ·단계별 숫자가 다 실려야

    나중에 표를 다시 그리거나 시드를 더 붙일 수 있다.
    """
    m = _load()
    evaluate, _seen = _fake_eval()
    from vtd_rl.policy.train import TrainConfig
    row = m.run_cell(_checkpoint(tmp_path), _toy_dataset(),
                     TrainConfig(epochs=1, seed=2, lr=1e-2, anchor_coef=0.1),
                     torch.device("cpu"), evaluate)
    blob = json.loads(json.dumps(row, ensure_ascii=False))     # 직렬화가 되는가
    for key in ("anchor_coef", "seed", "epochs", "lr", "squash", "samples", "loss",
                "anchor", "delta_theta", "theta_norm", "stages"):
        assert key in blob, key
    assert blob["stages"]["stage1"]["goal_rate"] == 0.5

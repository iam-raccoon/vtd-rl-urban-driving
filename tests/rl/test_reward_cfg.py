from vtd_rl.env.reward import RewardConfig
from vtd_rl.rl.reward_cfg import reward_config_from_hparams


def test_hparams가_비면_기본값이다():
    assert reward_config_from_hparams({}) == RewardConfig()


def test_승차감_설정을_복원한다():
    cfg = reward_config_from_hparams({"comfort_steer": -0.01, "comfort_accel": -0.005,
                                      "comfort_on_intent": True})
    assert cfg.comfort_steer == -0.01
    assert cfg.comfort_accel == -0.005
    assert cfg.comfort_on_intent is True


def test_모르는_키는_무시한다():
    """`hparams` 는 `vars(argparse.Namespace)` 라 보상과 무관한 키가 잔뜩 들어 있다."""
    cfg = reward_config_from_hparams({"envs": 30, "steps": 3000000, "comfort_steer": -0.02})
    assert cfg.comfort_steer == -0.02 and cfg.progress_total == RewardConfig().progress_total


def test_없는_키는_기본값을_남긴다():
    cfg = reward_config_from_hparams({"comfort_steer": -0.02})
    assert cfg.comfort_accel == RewardConfig().comfort_accel
    assert cfg.comfort_on_intent is False


def test_violation_mode를_복원한다():
    """M4d 최종 리뷰 Important #2: `if k in _FIELDS` 에 `and k != "violation_mode"` 를 더해도

    이 파일의 테스트 전부가(기존 것들은 승차감 세 필드만 본다) 통과했다 — 이 마일스톤이 새로
    만든 필드가 복원 경로에서 무방비였다. 끊기면 성적표가 `once_per_section` 실행을 `repeat`
    보상으로 평가하게 된다.
    """
    cfg = reward_config_from_hparams({"violation_mode": "once_per_section"})
    assert cfg.violation_mode == "once_per_section"
    # 안 주면 기본값(예전 동작)을 남긴다는 것도 같이 잠근다.
    assert reward_config_from_hparams({}).violation_mode == RewardConfig().violation_mode == "repeat"


def test_item_scale_None_은_옛_hparams_처럼_기본값이다():
    """`train_ppo` 는 `--item-scale` 을 안 주면 `hparams["item_scale"] = None` 으로 남긴다 —

    성적표(`report_m4a.py`)가 이 로그로 보상 설정을 되살릴 때 `None` 에서 터지면 새 실행 전부가 막힌다.
    """
    assert reward_config_from_hparams({"item_scale": None}) == RewardConfig()


def test_item_scale_을_JSON_목록_꼴에서_복원한다():
    cfg = reward_config_from_hparams({"item_scale": [[7, 5.0]]})
    assert cfg.item_scale == ((7, 5.0),)

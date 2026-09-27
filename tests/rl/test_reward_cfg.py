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

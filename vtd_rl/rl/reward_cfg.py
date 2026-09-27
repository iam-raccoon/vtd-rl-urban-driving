"""`hparams` ↔ `RewardConfig` 변환 — `train_ppo.py` 와 `report_m4a.py` 가 함께 쓴다.

성적표를 만들 때 실행의 보상 설정을 되살려야 평가의 '평균 보상' 이 학습 목적함수와 같은 자가
된다(M4c 최종 리뷰 I6). 두 스크립트가 서로를 임포트하지 않으므로 여기 한 곳에 둔다.
"""
import dataclasses

from vtd_rl.env.reward import RewardConfig

_FIELDS = tuple(f.name for f in dataclasses.fields(RewardConfig))


def reward_config_from_hparams(hparams: dict) -> RewardConfig:
    """`hparams`(= `vars(args)`) 에서 `RewardConfig` 필드와 이름이 같은 것만 골라 덮어쓴다.

    보상과 무관한 키(`envs`·`steps` 등)는 무시하고, 없는 키는 기본값을 남긴다 — 그래서
    보상 필드가 생기기 전의 옛 실행 로그로도 안전하게 돌아간다.
    """
    known = {k: v for k, v in (hparams or {}).items() if k in _FIELDS}
    return dataclasses.replace(RewardConfig(), **known)

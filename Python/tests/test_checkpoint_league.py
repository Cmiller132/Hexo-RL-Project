from hexorl.eval.checkpoint_league import CheckpointLeague
from hexorl.eval.scorecard import final_score_from_league_lcb


def test_checkpoint_league_persists_ratings(tmp_path):
    league = CheckpointLeague()
    rating = league.record_match("ckpt_a", color="black", wins=6, losses=2, draws=0)
    assert rating.rating_mean > 1500.0
    assert rating.rating_std > 0.0
    assert rating.lcb == rating.rating_mean - rating.rating_std

    path = tmp_path / "league.json"
    league.save(path)
    loaded = CheckpointLeague.load(path)
    assert loaded.ratings["ckpt_a"].lcb == rating.lcb
    assert loaded.ratings["ckpt_a"].by_color["black"].wins == 6


def test_checkpoint_league_evaluates_both_colors():
    league = CheckpointLeague()
    league.record_match("ckpt_a", color="black", wins=3, losses=1)
    assert not league.both_colors_recorded("ckpt_a")
    league.record_match("ckpt_a", color="white", wins=2, losses=2)
    assert league.both_colors_recorded("ckpt_a")
    assert league.ratings["ckpt_a"].by_color["white"].games == 4


def test_final_score_uses_league_lcb():
    league = CheckpointLeague()
    league.record_match("high_mean_noisy", color="black", wins=1, losses=0)
    league.record_match("steadier", color="black", wins=8, losses=2)
    league.record_match("steadier", color="white", wins=8, losses=2)
    champion = league.champion_by_lcb()
    assert champion.checkpoint_id == "steadier"
    assert final_score_from_league_lcb(champion) == champion.lcb

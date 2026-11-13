"""eval_extra_every_n_epoch: which evals also score the extra FoVs (train_singeo_vigor.py)."""
from singeo.evaluate.vigor import extra_fovs_due


def _due_epochs(every, eval_every=4, epochs=80):
    evals = [e for e in range(1, epochs + 1) if e % eval_every == 0 or e == epochs]
    return [e for e in evals if extra_fovs_due(e, every, epochs)]


def test_none_scores_extras_at_every_eval():
    assert _due_epochs(None) == list(range(4, 81, 4))


def test_every_16_scores_five_of_twenty_evals():
    assert _due_epochs(16) == [16, 32, 48, 64, 80]


def test_last_epoch_always_scores_extras():
    # 78 is neither a multiple of 4 nor of 16, but it is the final eval.
    assert _due_epochs(16, epochs=78) == [16, 32, 48, 64, 78]

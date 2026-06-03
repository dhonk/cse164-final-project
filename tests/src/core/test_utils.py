import pytest

import random
import numpy as np
import torch

import src.core.utils as utils


def _draw():
    """Pull one sample from each RNG seed_everything touches."""
    return (
        random.random(),
        np.random.rand(),
        torch.rand(3),
    )


def _draws_equal(a, b):
    return (
        a[0] == b[0]
        and a[1] == b[1]
        and torch.equal(a[2], b[2])
    )


def test_same_seed_reproduces_draws():
    """Re-seeding with the same value yields identical draws across all RNGs."""
    utils.seed_everything(0)
    first = _draw()
    utils.seed_everything(0)
    second = _draw()
    assert _draws_equal(first, second)


def test_different_seeds_differ():
    """Distinct seeds should (overwhelmingly likely) produce different draws."""
    utils.seed_everything(0)
    a = _draw()
    utils.seed_everything(1)
    b = _draw()
    assert not _draws_equal(a, b)


def test_default_seed_is_zero():
    """Calling with no arg matches calling with seed=0."""
    utils.seed_everything()
    default = _draw()
    utils.seed_everything(0)
    explicit = _draw()
    assert _draws_equal(default, explicit)


def test_rng_state_is_deterministic():
    """Same seed leaves each generator in an identical internal state."""
    utils.seed_everything(123)
    py1, np1, pt1 = random.getstate(), np.random.get_state(), torch.get_rng_state()
    utils.seed_everything(123)
    py2, np2, pt2 = random.getstate(), np.random.get_state(), torch.get_rng_state()

    assert py1 == py2
    assert np1[0] == np2[0] and np.array_equal(np1[1], np2[1])
    assert torch.equal(pt1, pt2)


def test_mask_check_valid():
    mask = np.arange(300)
    assert utils.mask_check(mask)

def test_mask_check_oob():
    mask = np.array([1, 2, 3, 4, -10])
    assert not utils.mask_check(mask)

def test_mask_check_ignore():
    mask = np.array([1, 2, 3, 4, 1000])
    assert not utils.mask_check(mask)
"""Local sanity checks for a1.py — not a Gradescope deliverable.

Run with: python3 test_a1.py
"""
import numpy as np
from a1 import A1


def test_transition_matrix_rows_sum_to_one():
    a1 = A1()
    states = ["sunny", "rainy", "cloudy"]
    sequence = ["sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "rainy", "cloudy"]
    tm = a1.generate_markov_chain(states, sequence)
    assert np.allclose(tm.sum(axis=1), 1.0), f"rows do not sum to 1: {tm.sum(axis=1)}"
    print("PASS: transition matrix rows sum to 1")


def test_sample_length_and_start():
    a1 = A1()
    states = ["sunny", "rainy", "cloudy"]
    sequence = ["sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "rainy", "cloudy"]
    a1.generate_markov_chain(states, sequence)
    samples = a1.generate_samples("sunny", 42, 9)
    assert len(samples) == 10, f"expected 10 states, got {len(samples)}"
    assert samples[0] == "sunny"
    print("PASS: sample length and starting state correct")


def test_stationary_distribution_is_valid():
    a1 = A1()
    states = ["sunny", "rainy", "cloudy"]
    sequence = ["sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "rainy", "cloudy"]
    a1.generate_markov_chain(states, sequence)
    pi = a1.stationary_distribution()
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(pi >= -1e-9)
    print("PASS: stationary distribution is a valid probability vector:", pi)


def test_matches_lecture_LR_example():
    """Reproduces the Finite Markov Chain slide's own walking example."""
    a1 = A1()
    states = ["L", "R"]
    sequence = ["L", "R", "L", "L", "L", "R", "R", "L", "R", "R"]
    tm = a1.generate_markov_chain(states, sequence)
    l_idx, r_idx = states.index("L"), states.index("R")
    assert np.isclose(tm[l_idx, l_idx], 2 / 5), f"P(L|L) should be 2/5, got {tm[l_idx, l_idx]}"
    assert np.isclose(tm[l_idx, r_idx], 3 / 5), f"P(R|L) should be 3/5, got {tm[l_idx, r_idx]}"
    print("PASS: matches Finite Markov Chain slide — P(L|L)=2/5, P(R|L)=3/5")


if __name__ == "__main__":
    test_transition_matrix_rows_sum_to_one()
    test_sample_length_and_start()
    test_stationary_distribution_is_valid()
    test_matches_lecture_LR_example()
    print("ALL LOCAL CHECKS PASSED")
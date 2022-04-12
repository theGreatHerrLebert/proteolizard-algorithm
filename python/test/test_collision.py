from proteolizardalgo.hashing import TimsHasher
import numpy as np

hasher = TimsHasher(32, 32, 15, 2, 10)


def test_collision_no_collision():
    # two windows share same bin but no key, collision should be empty
    H_test = np.array([[1, 2]]).T
    s_test = [1, 2]
    b_test = [1, 1]

    bins, scans = hasher.calculate_collisions(H_test, s_test, b_test)

    assert len(bins) == 0


def test_collision_first():
    # two windows share same bin and one key, collision should occur
    H_test = np.array([[1, 1]]).T
    s_test = [1, 2]
    b_test = [1, 1]

    bins, scans = hasher.calculate_collisions(H_test, s_test, b_test)

    assert np.all(bins == np.array([1, 1])) and np.all(scans == np.array([1, 2]))


def test_collision_second():
    # two windows share same bin and the same scan and one key, collision should occur
    H_test = np.array([[1, 1]]).T
    s_test = [1, 1]
    b_test = [1, 1]

    bins, scans = hasher.calculate_collisions(H_test, s_test, b_test)

    assert np.all(bins == np.array([1])) and np.all(scans == np.array([1]))


def test_collision_third():
    H_test = np.array([[1, 2, 3, 4, 1], [5, 6, 7, 8, 6]]).T
    s_test = [1, 1, 2, 4, 4]
    b_test = [1, 2, 1, 1, 2]

    bins, scans = hasher.calculate_collisions(H_test, s_test, b_test)

    assert np.all(bins == np.array([2, 2])) and np.all(scans == np.array([1, 4]))

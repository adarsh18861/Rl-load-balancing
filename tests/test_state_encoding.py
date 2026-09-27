import pytest

from lb_rl.utils.state_encoding import assert_state_count, decode, encode, state_count


def test_state_count():
    assert state_count([5, 5, 5, 5, 5]) == 5**5


def test_encode_decode_roundtrip():
    radices = [5, 5, 5, 5, 5]
    for idx in range(state_count(radices)):
        values = decode(idx, radices)
        assert encode(values, radices) == idx


def test_assert_state_count_limit():
    with pytest.raises(ValueError):
        assert_state_count([5, 5, 5, 5, 5], max_states=100)
    assert assert_state_count([5, 5, 5, 5, 5], max_states=5**5) == 5**5

import unittest
import preprocess
import numpy as np


class PreprocessTests(unittest.TestCase):

    def test_one_ema(self):
        ema = preprocess.ema(1.0, 2.0, 1.0)
        self.assertAlmostEqual(ema, 2.0, 6, "ema - 1.0: test failed")

    def test_zero_ema(self):
        ema = preprocess.ema(1.0, 2.0, 0.0)
        self.assertAlmostEqual(ema, 1.0, 6, "ema - 0.0: test failed")

    def test_half_ema(self):
        ema = preprocess.ema(1.0, 2.0, 0.5)
        self.assertAlmostEqual(ema, 1.5, 6, "ema [0.5] test failed")

    def test_arr_ema_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        ema = preprocess.arr_ema(arr, 1.0)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_almost_equal(ema, expected)

    def test_arr_ema_zero(self):
        arr = np.array([1.0, 2.0, 3.0])
        ema = preprocess.arr_ema(arr, 0.0)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_almost_equal(ema, expected)

    def test_arr_ema_half(self):
        arr = np.array([1.0, 2.0, 1.5, 2.0])
        ema = preprocess.arr_ema(arr, 0.5)
        expected = np.array([1.0, 1.5, 1.5, 1.75])
        np.testing.assert_almost_equal(ema, expected)

    def test_arr_rema_half(self):
        arr = np.array([1.75, 2.0, 1.5, 2.0, 1.0])
        ema = preprocess.arr_rema(arr, 0.5)
        expected = np.array([1.75, 1.5, 1.5, 1.0, 1.0])
        np.testing.assert_almost_equal(ema, expected)

    def test_roll_arr_fwd(self):
        arr = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0])
        roll_fwd_arr = preprocess.roll_arr_fwd(arr)
        expected = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0])
        np.testing.assert_almost_equal(roll_fwd_arr, expected)

    def test_roll_arr_bwd(self):
        arr = np.array([0.0, 1.0, 2.0, 0.0, 3.0, 0.0])
        roll_fwd_arr = preprocess.roll_arr_bwd(arr)
        expected = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 0.0])
        np.testing.assert_almost_equal(roll_fwd_arr, expected)

    def test_pct_chg(self):
        arr = np.array([1.0, 2.0, 4.0, 2.0, 2.0, 2.2])
        pct_ch_arr = preprocess.pct_chg(arr)
        expected = np.array([0.0, 1.0, 1.0, -0.5, 0.0, 0.1])
        np.testing.assert_almost_equal(pct_ch_arr, expected)

    def test_pct_diff(self):
        prev = np.array([1.0, 1.0, 2.0, 2.0])
        next = np.array([2.0, 1.5, 1.0, 2.0])
        pct_ch_arr = preprocess.pct_diff(prev, next)
        expected = np.array([1.0, 0.5, -0.5, 0.0])
        np.testing.assert_almost_equal(pct_ch_arr, expected)


import unittest
import trading
import numpy as np


class TradingTests(unittest.TestCase):
    def test_new_position(self):
        # zero
        self.assertEqual(trading.new_position(0., 5.0, 1.0), 1.)
        self.assertEqual(trading.new_position(0., -5.0, 1.0), -1.)
        self.assertEqual(trading.new_position(0., 0.5, 1.0), 0.)
        self.assertEqual(trading.new_position(0., -0.5, 1.0), 0.)
        # long
        self.assertEqual(trading.new_position(1., 5.0, 1.0), 1.)
        self.assertEqual(trading.new_position(1., -5.0, 1.0), -1.)
        self.assertEqual(trading.new_position(1., 0.5, 1.0), 1.)
        self.assertEqual(trading.new_position(1., -0.5, 1.0), 1.)
        # short
        self.assertEqual(trading.new_position(-1., 5.0, 1.0), 1.)
        self.assertEqual(trading.new_position(-1., -5.0, 1.0), -1.)
        self.assertEqual(trading.new_position(-1., 0.5, 1.0), -1.)
        self.assertEqual(trading.new_position(-1., -0.5, 1.0), -1.)

    def test_get_state(self):
        value = np.array([0.5, 2.0, 2.2, 0.5, -0.5, -1.5, 0.0, 0.5, 1.5])
        state = trading.get_position(value, 1.0)
        expected = np.array([0., 1., 1., 1., 1., -1., -1., -1., 1.])
        np.testing.assert_array_equal(state, expected)

    def test_pnl(self):
        pos = np.array([0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0])
        px = np.array([1.0, 1.0, 2.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0])
        pnl = trading.get_pnl(pos, px)
        exp = np.array([0.0, 0.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_array_equal(pnl, exp)

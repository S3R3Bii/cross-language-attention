from datetime import date, timedelta
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model import FAMILIES, PENALTIES, feature_row, fit, metrics, predict


class ScientificContracts(unittest.TestCase):
    def histories(self):
        origin = date(2024, 3, 1)
        values = {(origin - timedelta(days=i)).isoformat(): 50 + i * i for i in range(-3, 45)}
        return origin, values, dict(values)

    def test_same_day_yesterday_and_future_do_not_enter_features(self):
        day, own, english = self.histories()
        for family in FAMILIES:
            before = feature_row(day, own, english, family)[0]
            for offset in [-3, -2, -1, 0, 1]:
                key = (day - timedelta(days=offset)).isoformat()
                own[key], english[key] = 999999, 888888
            self.assertEqual(before, feature_row(day, own, english, family)[0])

    def test_declared_fast_and_slow_english_lags(self):
        day, own, english = self.histories()
        fast = feature_row(day, own, english, 'english2')[1]
        slow = feature_row(day, own, english, 'english7')[1]
        self.assertEqual(fast[-5:], ['english_lag2', 'english_lag7', 'english_lag14', 'english_lag28', 'english_mean2_8'])
        self.assertEqual(slow[-5:], ['english_lag7', 'english_lag14', 'english_lag21', 'english_lag35', 'english_mean7_13'])

    def test_constant_predictor_has_finite_penalized_solution(self):
        x = np.array([[1, i] for i in range(12)], dtype=float)
        m = fit(x, np.arange(12) + 20, 0.1, ['constant', 'trend'], [str(i) for i in range(12)])
        self.assertEqual(m['scale'][0], 1)
        self.assertAlmostEqual(m['coefficient'][0], 0)
        self.assertLess(m['fit_max_gradient'], 1e-9)

    def test_training_scaler_is_not_fitted_to_prediction_rows(self):
        x = np.array([[i, i % 2] for i in range(20)], dtype=float)
        m = fit(x, np.arange(20) + 5, 1, ['x', 'cycle'], [str(i) for i in range(20)])
        first = predict(m, np.array([[3., 1.]]))[2][0]
        both = predict(m, np.array([[3., 1.], [100., -100.]]))[2][0]
        self.assertEqual(first, both)

    def test_exponential_overflow_is_rejected_instead_of_capped(self):
        m = {'intercept': 1000., 'mean': [0.], 'scale': [1.], 'coefficient': [0.]}
        with np.errstate(over='ignore'):
            with self.assertRaises(AssertionError):
                predict(m, np.array([[1.]]))

    def test_negative_log_prediction_is_recorded_then_clipped(self):
        m = {'intercept': -2., 'mean': [0.], 'scale': [1.], 'coefficient': [0.]}
        raw, clipped, counts = predict(m, np.array([[1.]]))
        self.assertEqual((raw[0], clipped[0], counts[0]), (-2., 0., 0.))

    def test_log_and_raw_losses_answer_different_questions(self):
        actual = np.array([0., 9.]); pred = np.array([0., 0.])
        score = metrics(actual, np.array([0., 0.]), pred, mase_scale=3)
        self.assertEqual(score['mae'], 4.5)
        self.assertEqual(score['mse'], 40.5)
        self.assertAlmostEqual(score['msle'], np.log(10) ** 2 / 2)
        self.assertEqual(score['mase'], 1.5)

    def test_finite_search_matches_the_documented_bounded_extension(self):
        self.assertEqual(PENALTIES, [0.0001, 0.001, 0.01, 0.1, 1., 10., 100.])
        self.assertEqual(len(FAMILIES) * len(PENALTIES), 21)


if __name__ == '__main__':
    unittest.main()

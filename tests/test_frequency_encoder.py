"""Самостоятелни тестове за FrequencyEncoder."""

import unittest

import numpy as np

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ml_experiment_lab.encoding import FrequencyEncoder


class FrequencyEncoderTests(unittest.TestCase):
    """Проверява честотното кодиране на категории."""

    def test_frequency_calculation(self):
        """Проверява изчисляването на относителните честоти."""

        X = np.array(
            [
                ["Варна"],
                ["София"],
                ["Варна"],
                ["Пловдив"],
                ["Варна"],
            ],
            dtype=object,
        )

        encoder = FrequencyEncoder()

        encoder.fit(X)

        self.assertEqual(
            encoder.frequency_maps_[0],
            {
                "Варна": 0.6,
                "София": 0.2,
                "Пловдив": 0.2,
            },
        )

        self.assertAlmostEqual(
            sum(encoder.frequency_maps_[0].values()),
            1.0,
        )

    def test_fit_transform(self):
        """Проверява едновременно обучение и трансформация."""

        X = np.array(
            [["A"], ["A"], ["B"], ["C"]],
            dtype=object,
        )

        encoder = FrequencyEncoder()

        transformed = encoder.fit_transform(X)

        np.testing.assert_allclose(
            transformed,
            [
                [0.5],
                [0.5],
                [0.25],
                [0.25],
            ],
        )

        self.assertEqual(
            transformed.shape,
            (4, 1),
        )

        self.assertEqual(
            transformed.dtype,
            np.dtype(float),
        )

    def test_unknown_categories(self):
        """Проверява непознати категории при трансформация."""

        X_train = np.array(
            [["A"], ["A"], ["B"]],
            dtype=object,
        )

        X_test = np.array(
            [["A"], ["C"], ["B"], ["D"]],
            dtype=object,
        )

        encoder = FrequencyEncoder().fit(X_train)

        transformed = encoder.transform(X_test)

        np.testing.assert_allclose(
            transformed,
            [
                [2 / 3],
                [0.0],
                [1 / 3],
                [0.0],
            ],
        )

    def test_multiple_columns(self):
        """Проверява независимо кодиране на няколко колони."""

        X = np.array(
            [
                ["A", "X"],
                ["A", "Y"],
                ["A", "Y"],
                ["B", "Y"],
                ["B", "Z"],
            ],
            dtype=object,
        )

        encoder = FrequencyEncoder()

        transformed = encoder.fit_transform(X)

        self.assertEqual(
            encoder.frequency_maps_[0],
            {"A": 0.6, "B": 0.4},
        )

        self.assertEqual(
            encoder.frequency_maps_[1],
            {"X": 0.2, "Y": 0.6, "Z": 0.2},
        )

        np.testing.assert_allclose(
            transformed,
            [
                [0.6, 0.2],
                [0.6, 0.6],
                [0.6, 0.6],
                [0.4, 0.6],
                [0.4, 0.2],
            ],
        )

        self.assertEqual(
            transformed.shape,
            (5, 2),
        )

    def test_transform_does_not_change_frequencies(self):
        """Проверява, че Test не променя научените честоти."""

        X_train = np.array(
            [["A"], ["A"], ["B"]],
            dtype=object,
        )

        X_test = np.array(
            [["B"], ["B"], ["B"], ["C"]],
            dtype=object,
        )

        encoder = FrequencyEncoder().fit(X_train)

        original_frequencies = tuple(
            frequency_map.copy()
            for frequency_map in encoder.frequency_maps_
        )

        transformed = encoder.transform(X_test)

        self.assertEqual(
            encoder.frequency_maps_,
            original_frequencies,
        )

        self.assertNotIn(
            "C",
            encoder.frequency_maps_[0],
        )

        np.testing.assert_allclose(
            transformed[:, 0],
            [1 / 3, 1 / 3, 1 / 3, 0.0],
        )

    def test_single_category(self):
        """Проверява колона само с една категория."""

        X_train = np.array(
            [["A"], ["A"], ["A"]],
            dtype=object,
        )

        encoder = FrequencyEncoder()

        transformed = encoder.fit_transform(X_train)

        self.assertEqual(
            encoder.frequency_maps_[0],
            {"A": 1.0},
        )

        np.testing.assert_array_equal(
            transformed,
            np.ones((3, 1)),
        )

    def test_feature_names(self):
        """Проверява имената на изходните характеристики."""

        X = np.array(
            [
                ["Варна", "България"],
                ["Берлин", "Германия"],
            ],
            dtype=object,
        )

        encoder = FrequencyEncoder().fit(X)

        self.assertEqual(
            encoder.n_features_in_,
            2,
        )

        self.assertEqual(
            tuple(encoder.get_feature_names_out()),
            ("x0", "x1"),
        )

        self.assertEqual(
            tuple(
                encoder.get_feature_names_out(
                    ["City", "Country"]
                )
            ),
            ("City", "Country"),
        )

    def test_transform_before_fit(self):
        """Проверява трансформация преди обучение."""

        encoder = FrequencyEncoder()

        X = np.array(
            [["A"], ["B"]],
            dtype=object,
        )

        with self.assertRaises(NotFittedError):
            encoder.transform(X)

    def test_invalid_input(self):
        """Проверява отхвърлянето на невалидни входни данни."""

        invalid_inputs = (
            np.array(["A", "B"]),
            np.empty((0, 1), dtype=object),
            np.empty((2, 0), dtype=object),
            np.array(
                [["A"], [np.nan]],
                dtype=object,
            ),
        )

        for X in invalid_inputs:
            with self.subTest(shape=X.shape):
                with self.assertRaises(ValueError):
                    FrequencyEncoder().fit(X)

        encoder = FrequencyEncoder().fit(
            np.array(
                [["A"], ["B"]],
                dtype=object,
            )
        )

        with self.assertRaises(ValueError):
            encoder.transform(
                np.array(
                    [["A", "B"]],
                    dtype=object,
                )
            )

    def test_clone_and_refit(self):
        """Проверява съвместимостта с clone и повторно обучение."""

        X = np.array(
            [["A"], ["A"], ["B"]],
            dtype=object,
        )

        encoder = FrequencyEncoder().fit(X)

        cloned_encoder = clone(encoder)

        self.assertFalse(
            hasattr(
                cloned_encoder,
                "frequency_maps_",
            )
        )

        new_X = np.array(
            [["C"], ["C"], ["D"], ["D"]],
            dtype=object,
        )

        cloned_encoder.fit(new_X)

        self.assertEqual(
            cloned_encoder.frequency_maps_[0],
            {"C": 0.5, "D": 0.5},
        )

        self.assertEqual(
            encoder.frequency_maps_[0],
            {"A": 2 / 3, "B": 1 / 3},
        )

        encoder.fit(new_X)

        self.assertEqual(
            encoder.frequency_maps_[0],
            {"C": 0.5, "D": 0.5},
        )

        self.assertNotIn(
            "A",
            encoder.frequency_maps_[0],
        )


if __name__ == "__main__":
    unittest.main()
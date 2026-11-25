"""Методи за кодиране на категориални характеристики."""

from __future__ import annotations

from collections import Counter

import numpy as np

from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)

from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)


class FrequencyEncoder(
    OneToOneFeatureMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Кодира категориите чрез относителната им честота.

    Всяка входна колона създава една изходна характеристика.

    Честотите се определят единствено от данните,
    предоставени на метода fit().

    Непознатите категории се преобразуват в 0.0.

    Липсващите стойности трябва да бъдат обработени
    преди използването на този трансформатор.
    """

    def fit(
        self,
        X,
        y=None,
    ) -> FrequencyEncoder:
        """Научава относителните честоти на категориите."""

        values = validate_data(
            self,
            X,
            reset=True,
            dtype=object,
        )

        row_count = values.shape[0]

        frequency_maps = []

        for column_index in range(values.shape[1]):

            column_values = values[:, column_index]

            category_counts = Counter(column_values)

            frequency_map = {
                category: count / row_count
                for category, count
                in category_counts.items()
            }

            frequency_maps.append(frequency_map)

        self.frequency_maps_ = tuple(frequency_maps)

        return self

    def transform(
        self,
        X,
    ) -> np.ndarray:
        """Преобразува категориите чрез научените честоти."""

        check_is_fitted(
            self,
            attributes=["frequency_maps_"],
        )

        values = validate_data(
            self,
            X,
            reset=False,
            dtype=object,
        )

        transformed = np.zeros(
            values.shape,
            dtype=float,
        )

        for column_index, frequency_map in enumerate(
            self.frequency_maps_
        ):

            for row_index, category in enumerate(
                values[:, column_index]
            ):

                transformed[
                    row_index,
                    column_index,
                ] = frequency_map.get(
                    category,
                    0.0,
                )

        return transformed
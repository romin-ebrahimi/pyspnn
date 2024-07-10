import numpy as np
import spnn_cpp


class SPNN:
    def __init__(
        self,
        smoothing_matrix=None,
    ):
        self._X = None
        self._y = None
        # Smoothing matrix is the covariance matrix in most cases.
        self._smoothing_matrix = smoothing_matrix
        self._inverse_cov_matrix = None
        self._classes = None
        self._n_classes = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fit the Scale Invariant Probabilistic Network. This method pre-computes
        the training data covariance matrix, the inverse covariance matrix, and
        the unique classes in the target array `y`.

        Args:
            X : array-like input data of shape (n_samples, n_features)
            y : array-like input target values of shape (n_samples,)
        """
        if self._X is None:
            self._X = X

        if self._y is None:
            self._y = y
            self._classes = np.unique(self._y)
            self._n_classes = len(self._classes)

        if self._smoothing_matrix is None:
            self._smoothing_matrix = np.cov(self._X)

        # TODO: generate matrix inverse
        # if non invertible return an error.
        self._inverse_cov_matrix = np.ndarray()

        return None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability estimates for each category.

        Args:
            X: array-like of shape (n_samples, n_features) to be scored, where
            `n_samples` is the number of samples and `n_features` is the number
            of features.

        Returns:
            array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self._classes`.
        """
        # TODO: fill in error checks

        # Use the C++ method on the backend for faster processing.
        probabilities = spnn_cpp.spnn_predict(
            self._X.tolist(),
            self._y.tolist(),
            X.tolist(),
            self._inverse_cov_matrix.tolist(),
        )

        return np.array(probabilities)

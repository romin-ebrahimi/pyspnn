import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import spnn_cpp


class SPNN:
    def __init__(
        self,
        identity=True,
    ):
        self.X_ = None
        self.y_ = None
        self.smoothing_matrix_ = None
        self._classes = None
        self._n_classes = None
        # If identity=True, then SPNN becomes original PNN.
        self._identity = identity
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        smoothing_matrix=None,
    ):
        """
        Fit the Scale Invariant Probabilistic Network. This method pre-computes
        the training data covariance matrix, the inverse covariance matrix, and
        the unique classes in the target array `y`. If identity=True, then
        multiply the smoothing matrix by the identity matrix, which results in
        the original PNN with diagonal smoothing elements only instead of the
        full smoothing matrix of the SPNN.

        Args:
            X : array-like input data of shape (n_samples, n_features)
            y : array-like input target values of shape (n_samples,)
            smoothing_matrix: typically the covariance matrix (optional)
        """
        assert y.ndim == 1, "Target y should be of shape (n_samples,)"
        assert X.shape[0] == y.shape[0], "X and y should have same row count."
        X, y = check_X_y(X, y)

        if self.X_ is None:
            self.X_ = X

        if self.y_ is None:
            self.y_ = y
            self._classes = np.unique(self.y_)
            self._n_classes = len(self._classes)

        if smoothing_matrix is None:
            self.smoothing_matrix_ = np.cov(self.X_, rowvar=False)
        else:
            self.smoothing_matrix_ = smoothing_matrix

        if self._identity:
            identity_matrix = np.identity(self.smoothing_matrix_.shape[0])
            self.smoothing_matrix_ = np.multiply(
                self.smoothing_matrix_,
                identity_matrix,
            )

        self._is_fitted = True

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate class probability estimates for the given prediction data.

        Args:
            X: array-like of shape (n_samples, n_features) to be scored, where
            `n_samples` is the number of samples and `n_features` is the number
            of features.

        Returns:
            array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model.
        """
        assert X.shape[1] == self.X_.shape[1], "Features missing from X."
        X = check_array(X)
        check_is_fitted(self)

        # Use the C++ method on the backend for faster processing.
        probabilities = spnn_cpp.spnn_predict(
            self.X_.tolist(),
            self.y_.tolist(),
            X.tolist(),
            self.smoothing_matrix_.tolist(),
        )

        return np.array(probabilities)

#include <armadillo>
#include <boost/python.hpp>

using namespace boost::python;

// Xtr (list) contains the covariates from the training data.
// Ytr (list) contains the labels from the training data.
// Xte (list) contains the covariates for the prediction data set.
// smoothing_matrix (list) contains the smoothing / covariance matrix.
list spnn_predict(
    list Xtr,
    list Ytr,
    list Xte,
    list smoothing_matrix
) {
    int size_tr = len(Xtr);
    int size_te = len(Xte);
    int n_features = len(Xtr[0]);
    int mat_dim = len(smoothing_matrix[0]);

    arma::mat X_tr(size_tr, n_features);
    arma::mat Y_tr(size_tr, 1);
    arma::mat X_te(size_te, n_features);

    // Convert from boost::python::list to arma::mat for matrix processing.
    // Iterate through list values and allocate to their respective arma::mat.
    arma::mat cov_matrix(mat_dim, mat_dim);
    for (int i = 0; i < mat_dim; i++) {
        for (int j = 0; j < mat_dim; j++){
            cov_matrix(i, j) = extract<double>(smoothing_matrix[i][j]);
        }
    }
    // Calculate the pseudo-inverse of the smoothing matrix.
    arma::mat inv_cov_matrix = arma::pinv(cov_matrix);

    // Convert from boost::python::list to arma::mat for matrix processing.
    for (unsigned int i = 0; i < size_tr; i++) {
        Y_tr(i, 0) = extract<int>(Ytr[i]);
        for (unsigned int j = 0; j < n_features; j++) {
            X_tr(i, j) = extract<double>(Xtr[i][j]);
        }
    }

    for (unsigned int i = 0; i < size_te; i++) {
        for (unsigned int j = 0; j < n_features; j++) {
            X_te(i, j) = extract<double>(Xte[i][j]);
        }
    }

    // Classes is an ordered mapping of the unique classes in the training data.
    arma::mat classes = arma::unique(Y_tr);
    arma::mat N_k(1, classes.n_rows);
    arma::mat Xd(1, n_features);
    double f;

    // Get the class counts for each class k and store in N_k.
    for (unsigned int k = 0; k < classes.n_rows; k++) {
        N_k(0, k) = static_cast<double>(
            arma::size(arma::find(X_tr == classes(k, 0)))[0]
        );
    }

    list probabilities;
    list empty_list;
    empty_list.append(object()); // Empty list for class probabilities.
    empty_list *= classes.n_rows;
    probabilities.append(empty_list); // Append Python [None] type.
    probabilities *= len(Xte);

    // Iterate through prediction data rows and generate density estimates.
    for (unsigned int i = 0; i < size_te; i++) {
        // Iterate through the training data, add the parzen density estimate
        // f_k to the running total for the class k, then estimate probability.
        arma::mat fk(1, classes.n_rows, arma::fill::zeros);
        for (unsigned int j = 0; j < size_tr; j++) {
            // What class is in Y_tr[j] relative to k index in classes matrix?
            arma::uvec k_find = arma::find(classes == Y_tr(j, 0));
            int k_index = static_cast<int>(k_find.at(0));

            Xd = X_te.row(i) - X_tr.row(j);
            f = std::exp(
                -0.5 * arma::as_scalar(arma::abs(Xd * inv_cov_matrix * Xd.t()))
            );

            //if (!std::isfinite(fi)) { fi = 0.0; }

            // Add to the density estimate f_k by scaling with N_k.
            fk(0, k_index) += f;
        }
        // Use f_k to estimate probabilities and then add values to Python list.
        for (unsigned int k = 0; k < fk.n_cols; k++) {
            fk(0, k) = fk(0, k) / N_k(0, k);
        }
        for (unsigned int k = 0; k < fk.n_cols; k++) {
            probabilities[i][k] = fk(0, k) / arma::accu(fk);
        }
    }

    return probabilities;
}

BOOST_PYTHON_MODULE(spnn_cpp) {
    using namespace boost::python;
    def("spnn_predict", spnn_predict);
    // Python usage example:
    // import spnn_cpp
    // spnn_cpp.predict()
}
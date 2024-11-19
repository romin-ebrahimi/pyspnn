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
    unsigned int size_tr = len(Xtr);
    unsigned int size_te = len(Xte);
    unsigned int n_features = len(Xtr[0]);
    unsigned int mat_dim = len(smoothing_matrix[0]);

    arma::mat X_tr(size_tr, n_features);
    arma::mat Y_tr(size_tr, 1);
    arma::mat X_te(size_te, n_features);

    // Convert from boost::python::list to arma::mat for matrix processing.
    // Iterate through list values and allocate to their respective arma::mat.
    arma::mat cov_matrix(mat_dim, mat_dim);
    for (unsigned int i = 0; i < mat_dim; i++) {
        for (unsigned int j = 0; j < mat_dim; j++){
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
    list probabilities;
    list empty_list;
    empty_list.append(object()); // Empty list for class probabilities.
    empty_list *= classes.n_rows;
    probabilities.append(empty_list); // Append Python [None] type.
    probabilities *= len(Xte);
    arma::mat Xd(1, n_features);
    arma::mat m_dist;
    double f;
    
    // Iterate through prediction data rows and generate probability estimates.
    arma::mat fk(1, classes.n_rows, arma::fill::zeros);
    for (unsigned int i = 0; i < size_te; i++) {
        // Iterate through categories and generate parzen density estimates.
        for (unsigned int k = 0; k < classes.n_rows; k ++) {
            // Iterate through the training data, add the density estimate f_k
            // to the running total for the given class k in vector fk.
            arma::uvec j_indices = arma::find(Y_tr == k);
            for (unsigned int j = 0; j < j_indices.n_rows; j++) {
                // Generate density estimate for training example j.
                Xd = X_te.row(i) - X_tr.row(j_indices(j, 0));
                m_dist = -0.5 * Xd * inv_cov_matrix * Xd.t();
                f = std::exp(m_dist(0, 0));
                if (std::isnan(f)) { f = 0.0; }
                // Add to the running total density estimate for class k.
                fk(0, k) += f;
            }
            // Scale density estimate for class k by size of training data.
            fk(0, k) /= j_indices.n_rows;
        }

        // Use fk to estimate probabilities and then add values to Python list.
        for (unsigned int k = 0; k < fk.n_cols; k++) {
            probabilities[i][k] = fk(0, k) / arma::accu(fk);
        }
        // Reset all density elements to zero before next test data row.
        fk.zeros();
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
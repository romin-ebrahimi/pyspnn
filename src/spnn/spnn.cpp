#include <armadillo>
#include <boost/python.hpp>

using namespace boost::python;

// Xtr (list) contains the covariates from the training data.
// Ytr (list) contains the labels from the training data.
// Xte (list) contains the covariates for the prediction data set.
// inverse_cov_matrix (list) contains the inverse covariance matrix.
list spnn_predict(
    list Xtr,
    list Ytr,
    list Xte,
    list inverse_cov_matrix,
) {
    ssize_t size_tr = len(Xtr);
    ssize_t size_te = len(Xte);
    ssize_t n_features = len(Xtr[0]);
    list probabilities;
    probabilities.append(object()); // Append Python [None] type.
    probabilities *= len(Xte);
    // TODO: create empty arma::mat for inverse_cov_matrix.
    // TODO: create empty arma::cube for storing the slices of training data.

    // Convert from boost::python::list to arma::mat for matrix processing.
    // Iterate through list values and allocate to their respective arma::mat.
    for (unsigned int i = 0; i < len(inverse_cov_matrix); i++) {
        for (unsigned int j = 0; j < len(inverse_cov_matrix[0]); j++){

        }
    }
    
    // Iterate through training data and create a 3D arma::mat where each slice
    // contains the training data for a given class k.
    for (unsigned int j = 0; j < size_tr; j++) {
            

    }

    // Iterate through prediction data rows and generate density estimates.
    for (unsigned int i = 0; i < size_te; i++) {
        // Iterate through each of the k classes and generate f_k.

        // Apply softmax to f_k and add the density estimates to probabilities.
        
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
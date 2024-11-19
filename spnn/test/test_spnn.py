import boost_cpp
from sklearn import datasets
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from spnn.spnn import SPNN
import unittest


class TestSPNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = datasets.load_iris()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_00_hello_world(self):
        """
        Testing C++ hello world method for testing the build of Python boost.
        """
        value = boost_cpp.hello_world()
        self.assertTrue(value == "Hello World.")

    def test_01_spnn(self):
        """
        Test SPNN model class within a scikit-learn Pipeline.
        """
        Xtr, Xte, Ytr, Yte = train_test_split(
            self.data.data,
            self.data.target,
            test_size=0.2,
            shuffle=True,
        )

        model_pipe = Pipeline([("spnn", SPNN(identity=True))])
        model_pipe.fit(
            X=Xtr,
            y=Ytr,
        )

        preds = model_pipe.predict_proba(X=Xte)
        print(preds)
        print(Yte)  # TODO: use a scoring method
        precision = average_precision_score(y_true=Yte, y_score=preds)
        print(precision)

        self.assertTrue(len(preds) > 0)  # TODO:


if __name__ == "__main__":
    unittest.main()

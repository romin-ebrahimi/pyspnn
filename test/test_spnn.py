import boost_cpp
from sklearn import datasets
from sklearn.pipeline import Pipeline
from spnn import SPNN
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
        Xtr = self.data.data[:100,]
        Xte = self.data.data[100:,]
        Ytr = self.data.target[:100,]
        Yte = self.data.target[100:,]
        model_pipe = Pipeline([("spnn", SPNN())])
        model_pipe.fit(
            X=Xtr,
            y=Ytr,
        )
        preds = model_pipe.predict_proba(X=Xte)

        print(Yte)  # TODO: use a scoring method

        self.assertTrue(len(preds) > 0)  # TODO:


if __name__ == "__main__":
    unittest.main()

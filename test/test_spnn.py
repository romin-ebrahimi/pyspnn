import boost_cpp
import spnn_cpp
import numpy as np
import pandas as pd
import unittest


class TestSpectral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = 666 # TODO: create a standard data set
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_00_hello_world(self):
        """
        Testing C++ hello world method for testing the build of Python boost:
        /etl/src/boost.cpp
        """
        value = boost_cpp.hello_world()
        self.assertTrue(value == "Hello World.")


if __name__ == "__main__":
    unittest.main()

import add_data
import unittest

class add_data_test(unittest.TestCase):
    def setUp(self):
        self.command = ['add_data.py', './image/test4.jpg', '-flip', 'False', '-keyword', 'exciting']
    def test_setup_data(self):
        img, flip, keyword = add_data.setup_data(self.command)
        expected_shape = (604, 371, 3)
        expected_flip = False
        expected_keword = 'exciting'
        self.assert_equal_setup_data(img.shape, expected_shape, flip, expected_flip, keyword, expected_keword)

    def assert_equal_setup_data(self, img_shape, expected_img_shape, flip, expected_flip, keyword, expected_keyword):
        self.assertEqual(img_shape, expected_img_shape)
        self.assertEqual(flip, expected_flip)
        self.assertEqual(keyword, expected_keyword)
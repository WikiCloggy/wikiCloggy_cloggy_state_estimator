import unittest
import label_maker

class label_maker_test(unittest.TestCase):
    def setUp(self):
        self.label = ['a', 'b', 'c']
    def testSetupLabel(self):
        command_list = ['label_maker.py', '-add', 'd', 'e']
        label_maker.setup_label(command_list, self.label)
        expected_label = ['a', 'b', 'c', 'd', 'e']
        self.assertEqual(expected_label, self.label)
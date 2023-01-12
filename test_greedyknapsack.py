import unittest
import numpy as np
import greedyalgorithm_knapsackproblem as gkp


class MyTestCase(unittest.TestCase):

    def test_class_Item(self):
        self.assertTrue(isinstance(gkp.Item("TestItem", 5), gkp.Item))

    def test_class_Item_random(self):
        self.assertTrue(isinstance(gkp.Item(), gkp.Item))

    def test_class_ItemList(self):
        self.assertTrue(isinstance(gkp.ItemList(), gkp.ItemList))

    def test_ItemList_size(self):
        count = 0
        s = gkp.ItemList()
        for i in range(0, 5):
            s.item_add(gkp.Item(str(i), np.random.randint(1, 100)))
        count += not s.item_list_size() == 5
        self.assertFalse(count)

    def test_ItemList_add_remove_in_list(self):
        check = 0
        i_list = [gkp.Item(str(i)) for i in range(0, 5)]
        s = gkp.ItemList(i_list)
        check += (len(i_list) != s.item_list_size())
        i_add = gkp.Item("test")
        s.item_add(i_add)
        item_search_byid = s.item_in_list_byid(i_add)
        check += not item_search_byid['check']
        item_search_byidx = s.item_in_list_byindex(item_search_byid['idx'])
        check += not item_search_byidx['check']

        check += not item_search_byid['item'].item_compare(i_add)
        check += not item_search_byidx['item'].item_compare(i_add)

        s.item_remove(i_add)
        check += s.item_in_list_byid(i_add)['check']
        self.assertFalse(check)


if __name__ == '__main__':
    unittest.main()

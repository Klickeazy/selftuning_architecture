import numpy as np
from copy import deepcopy as dc
# import time
# import matplotlib.pyplot as plt
# import pickle
# from pathlib import Path


class Item:
    """
    Item class containing item id string and int cost value
    """
    count = 1

    def __init__(self, id_in=None, value_in=None, low=1, high=1000):
        if id_in is None:
            self.id = "Rand"+str(self.count)
            self.count += 1
        else:
            self.id = str(id_in)

        if value_in is None:
            self.value = np.random.randint(low, high)
        else:
            self.value = value_in

    def item_compare(self, item_check):
        if not isinstance(item_check, Item):
            raise Exception('Incorrect data type')

        if self.id == item_check.id and self.value == item_check.id:
            return 1
        else:
            return 0


class ItemList:
    """
    Set of items with array of id strings, array of int values and int net value
    Functions to calculate net value, change item (add/remove), find item
    """

    def __init__(self, item_set=None):
        self.net_value = 0
        if item_set is None:
            self.ids = []
            self.values = []
        else:
            self.ids = [str(x.id) for x in item_set]
            self.values = [x.value for x in item_set]
        self.list_net_value()

    def list_net_value(self):
        self.net_value = sum(self.values)

    def item_add(self, new_item):
        if not isinstance(new_item, Item):
            raise Exception('Incorrect data type')

        self.ids.append(new_item.id)
        self.values.append(new_item.value)
        self.list_net_value()

    def item_add_item_list(self, new_item_list):
        if not isinstance(new_item_list, ItemList):
            raise Exception('Incorrect data type')
        for c in new_item_list.ids:
            if not self.item_in_list_byid(c)['check']:
                self.item_add(new_item_list.item_in_list_byid(c)['item'])

    def item_remove(self, old_item):
        if not isinstance(old_item, Item):
            raise Exception('Incorrect data type')

        item_check = self.item_in_list_byid(old_item.id)
        if item_check['check']:
            del self.ids[item_check['idx']]
            del self.values[item_check['idx']]
            self.list_net_value()
        else:
            raise Exception('Item not in list')

    def item_remove_item_list(self, old_item_list):
        if not isinstance(old_item_list, ItemList):
            raise Exception('Incorrect data type')
        for c in old_item_list.ids:
            if not self.item_in_list_byid(c)['check']:
                self.item_remove(old_item_list.item_in_list_byid(c)['item'])

    def item_change(self, target, decision):
        if decision == 'add':
            self.item_add(target)
        elif decision == 'remove':
            self.item_remove(target)
        else:
            raise Exception('Check decision')

    def item_list_size(self):
        return len(self.ids)

    def item_in_list_byid(self, check_item_id):

        try:
            idx = self.ids.index(check_item_id)
            check = 1
            ret_item = Item(self.ids[idx], self.values[idx])
        except ValueError:
            check = 0
            idx = -1
            ret_item = Item("", 0)
        return {'check': check, 'idx': idx, 'item': ret_item}

    def item_in_list_byindex(self, idx):
        if idx < len(self.ids):
            check = 1
            ret_item = Item(self.ids[idx], self.values[idx])
        else:
            check = 0
            ret_item = Item("", 0)
        return {'check': check, 'idx': idx, 'item': ret_item}


def list_compare(check_list1, check_list2):
    if not isinstance(check_list1, ItemList) or not isinstance(check_list2, ItemList):
        raise Exception('Incorrect data type')

    item_diff1 = ItemList()
    item_diff2 = ItemList()
    item_intersection = ItemList()
    merge_list = dc(check_list1)
    merge_list.item_add_item_list(check_list2)
    for c in merge_list.ids:
        test_c1 = check_list1.item_in_list_byid(c)
        test_c2 = check_list2.item_in_list_byid(c)
        if test_c1['check'] and test_c2['check']:
            item_intersection.item_add(test_c1['item'])
        else:
            if test_c1['check']:
                item_diff1.item_add(test_c1['item'])
            if test_c2['check']:
                item_diff2.item_add(test_c2['item'])
    return {'list1': item_diff1, 'list2': item_diff2, 'intersect': item_intersection}


def greedy_selection(choice_list, limit, item_list, metric):
    """
    Run greedy selection to iteratively select most valuable choice until limit is reached
    :param choice_list: available choices
    :param limit: hard (inclusive) constraint
    :param item_list: initial set of choices if present or empty ItemList by default
    :param metric: evaluation of types
    :return: item list after greedy selection
    """
    work_list = dc(item_list)
    choices = dc(choice_list)
    while metric(work_list, limit):
        choice_iterations = []
        for c in choices.ids:
            choice_iterations.append(dc(item_list))
            choice_iterations[-1].item_add(choices.item_in_list_byid(c)['item'])
        values = [i.net_value for i in choice_iterations]
        idx_max = values.index(max(values))
        work_list = dc(choice_iterations[idx_max])
        choices = choices.item_remove(choices.item_in_list_byindex(idx_max)['items'])
    return work_list


def greedy_rejection(choice_list, limit, item_list, metric):
    work_list = dc(item_list)
    choices = dc(choice_list)
    work_list.item_add_item_list(choices)
    while not metric(work_list, limit):
        choice_iterations = []
        for c in choices.ids:
            choice_iterations.append(dc(work_list))
            choice_iterations[-1].item_remove(work_list.item_in_list_byid(c)['item'])
        values = [i.net_value for i in choice_iterations]
        idx_max = values.index(max(values))
        choices.item_remove_item_list(list_compare(work_list, choice_iterations[idx_max])['list1'])
        work_list = dc(choice_iterations[idx_max])
    return work_list


def greedy_algorithm(choice_list, limit, algorithm, item_list=None, metric=None):
    if item_list is None:
        item_list = ItemList()
    if not isinstance(item_list, ItemList) or not isinstance(choice_list, ItemList):
        raise Exception('Incorrect data type')
    if metric is None:
        metric = limit_check

    if algorithm == 'selection':
        return greedy_selection(choice_list, limit, item_list, metric)
    elif algorithm == 'rejection':
        greedy_rejection(choice_list, limit, item_list, metric)
    else:
        raise Exception('Check algorithm')


def limit_check(item_list, limit):
    """
    Evaluate if hard constraints on the set are satisfied
    :param item_list:   selection list of ItemList type
    :param limit:   hard upper limit constraint on the set
    :return: 1 if below limit, 1 otherwise
    """
    if not isinstance(item_list, ItemList):
        raise Exception('Incorrect data type')
    return item_list.item_list_size() < limit

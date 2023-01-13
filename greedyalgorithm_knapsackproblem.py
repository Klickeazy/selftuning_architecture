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
    rand_id_count = [1]

    def __init__(self, id_in=None, value_in=None, low=1, high=1000):
        if id_in is None:
            self.id = "R"+str(self.rand_id_count[0])
            self.rand_id_count[0] += 1
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

    def display_item(self):
        print('ID: ', self.id, ' - Value: ', self.value)


class ItemList:
    """
    Set of items with array of id strings, array of int values and int net value
    Functions to calculate net value, change item (add/remove), find item
    """
    rand_id_count = [0]

    def __init__(self, item_set=None, name=None):
        self.net_value = 0
        if item_set is None:
            self.ids = []
            self.values = []
        else:
            self.ids = [str(x.id) for x in item_set]
            self.values = [x.value for x in item_set]
        if name is None:
            self.listid = "L" + str(self.rand_id_count[0])
            self.rand_id_count[0] += 1
        else:
            self.listid = dc(name)
        self.list_net_value()

    def list_net_value(self):
        if len(self.values) > 0:
            self.net_value = sum(self.values)
        else:
            self.net_value = None

    def display_list(self):
        print(self.listid, ':: N: ', self.item_list_size(), ' - Net value: ', self.net_value)
        # for c in range(0, self.item_list_size()):
        #     print('ID: ', self.ids[c], ' - Value: ', self.values[c])
        print('IDs: ', self.ids)
        print('Values: ', self.values)

    def display_id(self, search_id):
        search = self.item_in_list_byid(search_id)
        search['item'].display_item()

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
            # item_check['item'].display_item()
            self.ids.remove(item_check['item'].id)
            self.values.remove(item_check['item'].value)
            self.list_net_value()
        else:
            raise Exception('Item not in list')

    def item_remove_item_list(self, old_item_list):
        if not isinstance(old_item_list, ItemList):
            raise Exception('Incorrect data type')
        for c in old_item_list.ids:
            if self.item_in_list_byid(c)['check']:
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


def initialize_rand_items(n_items, name=None):
    initial_list = ItemList(name=name)
    for _ in range(0, n_items):
        initial_list.item_add(Item())
    return initial_list


def list_compare(check_list1, check_list2):
    if not isinstance(check_list1, ItemList) or not isinstance(check_list2, ItemList):
        raise Exception('Incorrect data type')

    item_diff1 = ItemList(name='list1')
    item_diff2 = ItemList(name='list2')
    item_intersection = ItemList(name='list intersection')
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


def greedy_selection(choice_list, limit, item_list, metric, policy="max"):
    """
    Run greedy selection to iteratively select most valuable choice until limit is reached
    :param choice_list: available choices
    :param limit: hard (inclusive) constraint
    :param item_list: initial set of choices if present or empty ItemList by default
    :param metric: evaluation of types
    :return: item list after greedy selection
    """
    work_list = dc(item_list)
    work_list.listid = 'Work_list'
    choices = dc(choice_list)
    choices.listid = 'Choices'
    # work_list.display_list()
    # choices.display_list()

    while metric(work_list, limit):
        choice_iterations = []
        for c in choices.ids:
            choice_iterations.append(dc(work_list))
            choice_iterations[-1].item_add(choices.item_in_list_byid(c)['item'])
            # choice_iterations[-1].display_list()
        values = [i.net_value for i in choice_iterations]
        if policy == "max":
            target_idx = values.index(max(values))
        elif policy == "min":
            target_idx = values.index(min(values))
        else:
            raise Exception("Check target policy")
        work_list = dc(choice_iterations[target_idx])
        # work_list.display_list()
        choices.item_remove(choices.item_in_list_byindex(target_idx)['item'])
        # choices.display_list()
    work_list.listid = 'Work_list'
    return work_list


def greedy_rejection(choice_list, limit, item_list, metric, policy="max"):
    work_list = dc(item_list)
    work_list.listid = 'Work_list'
    # work_list.display_list()
    choices = dc(choice_list)
    choices.listid = 'Choices'
    # choices.display_list()
    work_list.item_add_item_list(choices)
    # work_list.display_list()

    while not metric(work_list, limit+1):
        choice_iterations = []
        # choices.display_list()
        for c in choices.ids:
            choice_iterations.append(dc(work_list))
            # work_list.item_in_list_byid(c)['item'].display_item()
            choice_iterations[-1].item_remove(work_list.item_in_list_byid(c)['item'])
        values = [i.net_value for i in choice_iterations]
        if policy == "max":
            target_idx = values.index(max(values))
        elif policy == "min":
            target_idx = values.index(min(values))
        else:
            raise Exception("Check target policy")
        # list_compare(work_list, choice_iterations[idx_max])['list1'].display_list()
        # choices.display_list()
        choices.item_remove_item_list(list_compare(work_list, choice_iterations[target_idx])['list1'])
        # choices.display_list()
        work_list = dc(choice_iterations[target_idx])
    work_list.listid = 'Work_list'
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
        return greedy_rejection(choice_list, limit, item_list, metric)
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


def greedy_simultaneous_optimal(choice_list, change, item_list=None, metric=None, policy="max"):
    if item_list is None:
        item_list = ItemList(name='Item List')
    if metric is None:
        metric = limit_check

    work_list = dc(item_list)
    work_list.listid='Work_List'

    for _ in range(0, change):
        work_list.display_list()
        # references = []
        references = [work_list]
        compare = list_compare(choice_list, item_list)
        if compare['list1'].item_list_size() > 0:
            # At least one item in the choices is not already in the item list and can be selected
            select = greedy_selection(choice_list, item_list.item_list_size()+1, item_list, metric, policy)
        else:
            select = ItemList()
        select.listid = 'Select'
        references.append(select)
        if compare['list1'].item_list_size() > 0:
            # At least one item in the choice list is not in the item list and can be selected
            reject = greedy_selection(choice_list, item_list.item_list_size()-1, item_list, metric, policy)
        else:
            reject = ItemList()
        reject.listid = 'Reject'
        references.append(select)

        values = [i.values for i in references]
        if policy == "max":
            # print('Max:', max(values))
            target_idx = values.index(max(values))
        elif policy == "min":
            # print('Min:', min(values))
            target_idx = values.index(max(values))
        else:
            raise Exception('Check Policy')

        # NEED TO FIX CHOICES BASED ON SELECTION/REJECTION
        work_list = dc(references[target_idx])

    return work_list



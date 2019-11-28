import random as random
import numpy as np


def get_rand_range(start, end, num_items=None):
    if num_items is None:
            return random.random() * (end - start) + start
    else:
        return np.random.random(num_items) * (end - start) + start


def get_rand_int(start, end, num_items=None):
    if num_items==None:
        return random.randint(start, end)
    else:
        rand_ints_list = list()
        for i in range(num_items):
            rand_ints_list.append(random.randint(start, end))
        return np.asarray(rand_ints_list)


def get_rand_out_of_list_item(objects_list, num_items=None, objects_weights=None):
    # return random item within a given list
    if num_items is None:
        if objects_weights is None:
            return objects_list[random.randrange(len(objects_list))]
        else:
            assert len(objects_list) == len(objects_weights)
            obj_wght_cumsum = np.cumsum(objects_weights)
            rr = np.random.random_sample(num_items)
            for list_idx in range(len(objects_list)-1):
                if obj_wght_cumsum[list_idx] >= rr:
                    return objects_list[list_idx]
            return objects_list[-1]
    else:
        res_list = list()
        for i in range(num_items):
            single_item = get_rand_out_of_list_item(objects_list, num_items=None, objects_weights=objects_weights)
            res_list.append(single_item)
        return res_list


def get_rand_list_item(items_list):
    return items_list[random.randrange(len(items_list))]


if __name__ == "__main__":
    obj_list = ['a', 'b', 'c']
    a = get_rand_out_of_list_item(obj_list, num_items=10, objects_weights=[0.8, 0.1, 0.1])
    print(a)
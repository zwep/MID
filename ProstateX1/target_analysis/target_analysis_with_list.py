# encoding: utf-8

"""
Here we analyze the output scores... using only lists.... quite funny
"""

import collections
import os
import re


def get_value(input_key, input_list):
    """
    Get list from list of dicts with key...
    :param input_key:
    :return:
    """
    temp = []
    for x in input_list:
        for k, v in x.items():
            temp.append(x[k][input_key])
    return temp


def get_value_cond(input_key, input_list, cond):
    """
    Get list from list of dicts with key...
    :param input_key:
    :param input_list:
    :param cond: dict with key (column name): value (cond value)
    :return:
    """
    final_temp = []
    for k_cond, v_cond in cond.items():
        temp = []
        for x in input_list:
            for k, v in x.items():
                if x[k][k_cond] == v_cond:
                    temp.append(x[k][input_key])
        final_temp.append(temp)
    return final_temp


def calc_distr(input_list):
    """
    Calculates some statistics
    :param input_list:
    :return:
    """
    n = len(input_list)
    x_mean = sum(input_list)/n
    x_var = sum([(x-x_mean) ** 2 for x in input_list])/(n-1)
    x_max = max(input_list)
    x_min = min(input_list)

    res_dict = {'min': x_min, 'max': x_max, 'mean': x_mean, 'var': x_var}
    return res_dict


dir_data = r'C:\Users\C35612\data\mid\ProstateX-TrainingLesionInformationv2'

file_name = 'ProstateX-Findings-Train.csv'
file_path = os.path.join(dir_data, file_name)
if os.path.isfile(os.path.join(dir_data, file_name)):
    with open(file_path, 'r', encoding='utf-8') as f:
        dat_pat = [re.sub('\n', '', x).split(',') for x in f.readlines()]

n_pat = len(dat_pat)
list_pat = [{dat_pat[i][0]: {x: y for x, y in zip(dat_pat[0][1:], dat_pat[i][1:])}} for i in range(1, n_pat)]

for x in list_pat:
    for k, v in x.items():
        x[k]['pos'] = list(map(float, x[k]['pos'].split()))

# Getting an idea how frequent the ClinSig value is
res_clinsig = get_value('ClinSig', list_pat)
collections.Counter(res_clinsig).most_common(2)

# Getting an idea how frequent the fid is
res_fid = get_value('fid', list_pat)
collections.Counter(res_fid).most_common(5)

# Getting an idea how frequent the zone is
res_zone = get_value('zone', list_pat)
collections.Counter(res_zone).most_common(10)

# Getting an idea on the spread of the position
res_pos = get_value('pos', list_pat)
pos_x, pos_y, pos_z = map(list, zip(*res_pos))
calc_distr(pos_x)
calc_distr(pos_y)
calc_distr(pos_z)

# Now trying a 'group-by' with lists..

fid_cond = {'fid': '1'}
collections.Counter(get_value_cond('ClinSig', list_pat, fid_cond)[0]).most_common(2)

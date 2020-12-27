import json
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd


train_df = pd.read_csv(
    '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/paths/train.csv')
dev_df = pd.read_csv(
    '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/paths/dev.csv')
test_df = pd.read_csv(
    '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/paths/test.csv')
print('train', train_df[train_df['label'] == 1].shape[0])
print('dev', dev_df[dev_df['label'] == 1].shape[0])
print('test', test_df[test_df['label'] == 1].shape[0])


# # list_a = ['healthywebnosymp','healthywebnosymp','covidwebnocough', 'asthmawebwithcough']

# # list_b - ['asthmaandroidwithcough', 'covidandroidnocough']

# aug_list = ['mono', 'aug']

# with open(
#         '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/KDD_paper_data/files.json'
# ) as json_file:
#     data = json.load(json_file)

# value_count = {}
# unique_count = {}
# for key, value in data.items():
#     count = 0
#     if 'android' in key:
#         for iterable in value:
#             count += len(iterable)
#             for i in iterable:
#                 if not any(ele in i for ele in aug_list):
#                     unique_id = re.findall("_.........._", i)
#                     assert len(unique_id) == 1
#                     if len(unique_id) == 1:
#                         unique_count[unique_id[0]] = unique_count.get(
#                             unique_id[0], 0) + 1

#     elif 'web' in key:
#         for iterable in value:
#             count += len(iterable)
#             for i in iterable:
#                 if not any(ele in i for ele in aug_list):
#                     unique_id = re.findall("[0-9]" * 6, i)
#                     if len(unique_id) == 0:
#                         print(i)
#                     assert len(unique_id) == 1
#                     if len(unique_id) == 1:
#                         unique_count[unique_id[0]] = unique_count.get(
#                             unique_id[0], 0) + 1

#     value_count[key] = count

# print(value_count)
# print(sum(value_count.values()))
# print('*' * 30)
# print('*' * 30)
# print('*' * 30)
# print('*' * 30)
# print('*' * 30)
# print(len(unique_count.keys()))
# print(sum(unique_count.values()))

# counts_log = []

# for key, item in unique_count.items():
#     counts_log.append(item)
#     print(key, item)

# plt.figure()
# plt.hist(counts_log, bins=100)
# plt.xlabel("number samples given")
# plt.ylabel("count")
# plt.savefig('figs/numsamplesgiven2.png')

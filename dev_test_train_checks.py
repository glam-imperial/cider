import os
import sys
import numpy as np
import pandas as pd
import re


def sanity_check(task, cross_val=False):
    ids = []
    unique_ids = []
    full_paths = []
    count = 0
    for dset in ['train', 'dev', 'test']:
        if cross_val and not dset == 'dev':
            df = pd.read_csv(os.path.join('paths', dset + task + 'crossval.csv'))
        elif not cross_val:
            df = pd.read_csv(os.path.join('paths', dset + task + '.csv'))
        count += len(df.index)
        ids.extend(df.iloc[:, 0].tolist())
        unique_ids_inset = list(set(df.id.unique().tolist()))

        assert not any(ele in unique_ids for ele in unique_ids_inset
                       ), 'people in more than one set, Investigate'
        assert not any(ele in full_paths for ele in unique_ids_inset
                       ), 'people in more than one set, Investigate'
        unique_ids.extend(unique_ids_inset)
        full_paths.extend(df.path.tolist())
        # print(unique_ids_inset)
        print('number of unique users=', len(unique_ids_inset))


        assert len(set(ids)) == len(
            ids), f"Duplicates across train/dev/test. Investigate!"

        def func(x):
            x = x.replace(
                '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/KDD_paper_data/',
                '')
            x = x[:x.index('/')]
            return x

        categs = df.path.map(func)
        # print(categs.value_counts())
    print('total samples', count)

if __name__ == "__main__":
    sanity_check("task3")

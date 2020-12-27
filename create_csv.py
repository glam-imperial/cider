import os
import sys
import numpy as np
import pandas as pd
import re
import argparse

def user_id(path):
    '''
    given the name of the file returns it's user id
    '''
    if 'android' in path:
        unique_id = re.findall("_.{10}_", path)
    elif 'web' in path:
        unique_id = re.findall("[0-9]{6}", path)

    assert len(unique_id) == 1
    return unique_id[0]

def create_csv(task='task1', location='bitbucket', dev_split=0.1, test_split=0.1, cross_val=False):
    # task 1: For **task1**, we use covidandroidnocough + covidandroidwithcough  + covidwebnocough + covidwebwithcough
    #                                                               v.s.
    #                                           healthyandroidnosymp + healthywebnosymp
    # ( 66 user (141 sample) / 220 users (298 samples) in total);

    task_data = pd.DataFrame(columns=['path', 'label', 'id'])

    if 'all' in task:
        positive_labels = ['covidandroidnocough', 'covidandroidwithcough',
                        'covidwebnocough', 'covidwebwithcough']

        negative_labels = ['healthyandroidnosymp', 'asthmaandroidwithcough',
                        'healthywebwithcough', 'healthyandroidwithcough',
                        'asthmawebwithcough', 'healthywebnosymp']
    elif 'task1' in task:
        positive_labels = ['covidandroidnocough', 'covidandroidwithcough',
                        'covidwebnocough', 'covidwebwithcough']

        negative_labels = ['healthyandroidnosymp', 'healthywebnosymp']

    elif 'task2' in task:
        positive_labels = ['covidandroidwithcough', 'covidwebwithcough']

        negative_labels = ['healthywebwithcough', 'healthyandroidwithcough']

    elif 'task3' in task:
        positive_labels = ['covidandroidwithcough', 'covidwebwithcough']

        negative_labels = ['asthmaandroidwithcough', 'asthmawebwithcough']

    aug_list = ['mono', 'aug']

    if location == 'bitbucket':
        rootdir = '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/KDD_paper_data'
    else:
        rootdir = '/rds/general/user/hgc19/home//COVID_Audio_Diagnosis/KDD_paper_data'


    for subdir, dirs, files in os.walk(rootdir):

        if any(ele in subdir for ele in positive_labels):
            label = 1
        elif any(ele in subdir for ele in negative_labels):
            label = 0
        else:
            continue

        for wav_file in files:
            if '.wav' in wav_file and not any(ele in wav_file for ele in aug_list):
                unique_id = user_id(os.path.join(subdir, wav_file))
                task_data = pd.concat([
                    task_data,
                    pd.DataFrame(np.array([[os.path.join(subdir, wav_file), label, unique_id]
                                        ]),
                                columns=['path', 'label', 'id'])
                ],
                                    ignore_index=True)

    categs = positive_labels + negative_labels

    # if cross_val:
    #     split_psuedorandom_crossval(task_data, categs, 5, task)

    # split_psuedorandom(task_data, categs, dev_split, test_split, task, cross_val)
    # for dset in ['train', 'dev', 'test']:
    #     df = task_data.iloc[eval(dset), :]
    #     df = df.iloc[np.random.choice(range(len(df)), len(df), replace=False), :]
    #     df.to_csv(os.path.join('paths', dset+'.csv'))
    # sanity_check(task, cross_val)

def split_psuedorandom(df, categories, dev_split, test_split, task, cross_val):
    ''' Split data into train/test/dev preserving the distribution across
        data categories.

        update:
        We need to ensure that people who have recorded more than one clip do not appear
        in more than 1 set. - implementation is poss as a person cannot exist in more than
        one class.
    '''
    splits = (1 - dev_split - test_split, dev_split, test_split)    # Train/dev/test
    
    unique_ids = list(set(df.id.unique().tolist()))
    np.random.shuffle(unique_ids)
    n = len(unique_ids)
    total_data = 0
    for i, dset_name in enumerate(['train', 'dev', 'test']):
        if i == 'dev' and cross_val:
            continue
        n_split = int(np.ceil(n*splits[i]))
        dset = df[pd.DataFrame(df.id.tolist()).isin(unique_ids[:n_split]).any(1).values]
        unique_ids = unique_ids[n_split:]
        print(f"saving the file {os.path.join('paths', dset_name+task+'.csv')}")
        dset.to_csv(os.path.join('paths', dset_name+task+'.csv'))
        total_data += len(dset.index)
    assert not unique_ids, f"people remaining for category: {unique_ids}"
    assert len(df.index) == total_data, f"Rows have been lost"

def split_psuedorandom_crossval(df, categories, k, task):
    # Iterate through df rows and place into folds
    users = [[] for _ in range(k)]
    uids = df.id.unique().tolist()
    df['categs'] = df.path.map(strip_category)
    for uid in uids:
        categ = df.loc[df.id == uid, 'categs'].tolist()[0]
        list_idx = get_fewest_users_with_category(users, categ, df)
        users[list_idx].append(uid)
        users.sort(key=len)

    # Compute unique users per category per fold
    unique_df = pd.DataFrame({'col': (df.id + ';' + df.categs).unique()})
    unique_df[['id', 'categs']] = unique_df.col.str.split(';', expand=True)
    base = None
    for i,us in enumerate(users):
        vcs = pd.DataFrame({f'Fold_{i}': unique_df.loc[unique_df.id.isin(us), 'categs'].value_counts()})
        if not isinstance(base, pd.DataFrame):
            base = vcs
        else:
            base = base.merge(vcs, left_index=True, right_index=True)

    print('\n\nUserIds per class:\n\n',base)
    print('\n\nDifference in UserIds per class:\n\n',(base - np.array(base.iloc[:,0]).reshape(-1,1)))
    print(base.sum().sum(), len(uids))
    assert base.sum().sum() == len(uids)

def get_fewest_users_with_category(users, categ, df):
    ''' Given a category, return the list index who has the
        fewest userids of that category.
    '''
    counts = []
    for i in range(len(users)):
        uids = users[i]
        categ_uids = df.loc[(df.id.isin(uids)) & (df.categs == categ), 'id'].unique()
        counts.append(len(categ_uids))
    return np.argmin(counts)


def split_psuedorandom_crossval_old(df, categories, k, task):
    ''' Split data into train/test/dev preserving the distribution across
        data categories.

        update:
        We need to ensure that people who have recorded more than one clip do not appear
        in more than 1 set. - implementation is poss as a person cannot exist in more than
        one class.
    '''
    # splits = (1 - dev_split - test_split, dev_split, test_split)    # Train/dev/test
    df['fold'] = -1
    for categ in categories:
        # rows = df.loc[df.path.str.contains(categ)].index.tolist()
        rows = df.loc[df.path.str.contains(categ)]
        unique_ids = rows.id.unique().tolist()
        print(df.loc[df.id.isin(unique_ids) & ~df.path.str.contains(categ)])


        sys.exit()



        np.random.shuffle(unique_ids)
        n = len(unique_ids)
        total_data = 0
        for i in range(k):
            # fold_ids = unique_ids[int(np.ceil(n*i/10)):int(np.ceil((i+1)*n/10))]
            fold_ids = unique_ids[int(n*i/10):int((i+1)*n/10)]
            df.loc[df.id.isin(fold_ids), 'fold'] = i
            total_data += len(df['fold'][df.id.isin(fold_ids)])
        print(len(rows), total_data)
        # print(df.loc[df.path.str.contains(categ)].fold.unique())
        print(df.fold.value_counts())
        # print(df.loc[df.path.str.contains(categ)])
        assert len(rows) == total_data, f"people remaining for category: {categ}"
        print(df.fold.value_counts())
        sys.exit()


            # dset.to_csv(os.path.join('paths', 'cross_val', dset_name+task+'.csv'))
    assert len(df.index) == total_data, f"Rows have been lost"

def sanity_check(task, cross_val=False):
    ids = []
    unique_ids = []
    for dset in ['train', 'dev', 'test']:
        if cross_val and dset == 'dev':
            continue

        df = pd.read_csv(os.path.join('paths', dset + task + '.csv'))
        ids.extend(df.iloc[:,0].tolist())
        unique_ids_inset = list(set(df.id.unique().tolist()))
        assert not any(ele in unique_ids for ele in unique_ids_inset), 'people in more than one set, Investigate'
        unique_ids.extend(unique_ids_inset)
        print(unique_ids)
        print('number of unique users=', len(unique_ids_inset))
        print(df)
        # print(len(set(ids)), len(ids))
        assert len(set(ids)) == len(ids), f"Duplicates across train/dev/test. Investigate!"
        categs = df.path.map(strip_category)
        print(categs.value_counts())

def strip_category(x):
    x = x.replace('/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/KDD_paper_data/', '')
    x = x[:x.index('/')]
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='What task?')

    parser.add_argument('--task', type=str, help='what task do you want to perform?', default='all', choices=['all', 'task1', 'task2', 'task3'])
    parser.add_argument('--location', type=str, help='where is the data', default='bitbucket', choices=['bitbucket', 'hpc'])
    args = parser.parse_args()
    create_csv(args.task, args.location, cross_val=True)
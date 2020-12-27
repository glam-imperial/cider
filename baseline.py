import os
import sys
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score

def extract_smile():
    # Instructions:
    # 1. Download and build opensmile using insructions here: https://github.com/audeering/opensmile
    # 2. Add the smilextract to path (found here for me: /vol/bitbucket/aeg19/opensmile/build/progsrc/smilextract)
    save_file_dir = '/vol/bitbucket/aeg19/COVID_Audio_Diagnosis/smile/'
    csv_path = '/vol/bitbucket/aeg19/COVID_Audio_Diagnosis/paths/cross_val/all.csv'
    config_path = '/vol/bitbucket/aeg19/opensmile/config/compare16/ComParE_2016.conf'

    df = pd.read_csv(csv_path)[['path']]
    df.columns = ['data_path']
    df['smile_features_path'] = None

    for row in range(len(df)):
        # Create dest. path
        file_path = df.data_path[row]
        dest_path = file_path.replace(
            '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/KDD_paper_data/', save_file_dir
        ).replace('.wav', '.csv')
                
        # TODO: uncomment below to run full extraction (takes c. 10 mins)
        # # Create smile features
        # os.system(f"SMILExtract -C {config_path} -I " + file_path + " -csvoutput " + dest_path)

        # Save dest. path
        df['smile_features_path'][row] = dest_path

        # if row % 5 == 0:
        #     print('\n\nROW #:\t',row,'\n\n`')

    create_features_csv(df)

def create_features_csv(df):
    ''' Concatenate individual smile features csv's into
        one input features csv
    '''
    for row, path in enumerate(df.smile_features_path.tolist()):
        tmp_df = pd.read_csv(path, sep=';')
        if len(df.columns) == 2:
            df[tmp_df.columns] = None
        df.iloc[row, 2:] = tmp_df.values[0]

        if row % 20 == 0:
            print('\n\nROW #:\t',row,'\n\n`')
            
    df.to_csv(os.path.join('paths', 'smile_features', 'concat_compare.csv'))

def oversample(x, y, cls=1):
    msk = np.isclose(y, cls)
    x_ = np.concatenate([x] + [x[msk]] * 2, axis=0)
    y_ = np.concatenate([y] + [y[msk]] * 2, axis=0)
    return x_, y_

def eval_preds(y_true, logits):
    y_pred = [0 if l < 0 else 1 for l in logits]
    return {
        'unweighted_accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'UAR': recall_score(y_true, y_pred, average='macro'),
        'roc_auc': auc(*roc_curve(y_true, logits)[:2]),
    }

def train_svm(task):
    df = pd.read_csv('paths/smile_features/concat_compare.csv', index_col=0)
    splits = pd.read_csv(f'paths/cross_val/{task}.csv', index_col=0)        # TODO set paths here
    
    X, Y, orig_folds = [], [], []
    for row in df.values:
        path = row[0]
        data = splits[splits['path'] == path]
        if len(data) == 0:
            continue
        X.append(row[3:])
        assert data.shape[0] == 1, path
    
        Y.append(data['label'].values[0])
        fold = data['fold'].values[0]
    
        assert fold in range(4)
        orig_folds.append(fold)

    test_results = {}
    for dev_idx in range(3):
        train_folds = list([i for i in range(3) if i!=dev_idx])
        fold_allocs = {dev_idx:'dev', train_folds[0]:'train', train_folds[1]:'train', 3:'test'}
        folds = [fold_allocs[f] for f in orig_folds]

        X = np.array(X)
        Y = np.array(Y)
        folds = np.array(folds)
        X = PCA(100).fit_transform(X)
        train_x = X[folds == 'train']
        train_y = Y[folds == 'train']
        dev_x = X[folds == 'dev']
        dev_y = Y[folds == 'dev']
        test_x = X[folds == 'test']
        test_y = Y[folds == 'test']
        nontest_x = X[folds != 'test']
        nontest_y = Y[folds != 'test']
        print('train shapes:', train_x.shape, train_y.shape)
        best_comp = 1e-5
        best_uar = None
        for comp in [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]:  # SVM complexities (linear kernel)
            clf = svm.LinearSVC(C=comp, random_state=0, max_iter=1e7)
            train_x_, train_y_ = oversample(train_x, train_y)

            scaler = StandardScaler().fit(train_x_)
            train_x_scl = scaler.transform(train_x_)
            dev_x_scl = scaler.transform(dev_x)

            clf.fit(train_x_scl, train_y_)
            dev_y_pred = clf.decision_function(dev_x_scl)
            results = eval_preds(dev_y, dev_y_pred)
            if best_uar is None or best_uar < results['unweighted_accuracy']:
                best_comp = comp
                best_uar = results['unweighted_accuracy']
        print("best complexity parameter", best_comp)
        
        # Train with best comp and train+dev
        clf = svm.LinearSVC(C=best_comp, random_state=0, max_iter=1e7)
        nontest_x_, nontest_y_ = oversample(nontest_x, nontest_y)
        
        scaler = StandardScaler().fit(nontest_x_)
        nontest_x_scl = scaler.transform(nontest_x_)
        test_x_scl = scaler.transform(test_x)
        
        clf.fit(nontest_x_scl, nontest_y_)
        test_y_pred = clf.decision_function(test_x_scl)
        results = eval_preds(test_y, test_y_pred)
        test_results[dev_idx] = results
        print('\nDev fold =', dev_idx, '\n', results)

    print('\n\n** Test results **\n\n')
    uar_scores = [r['UAR'] for r in test_results.values()]
    print(f"UAR:\tmean: {np.mean(uar_scores)}\tstd: {np.std(uar_scores)}")
    roc_auc_scores = [r['roc_auc'] for r in test_results.values()]
    print(f"roc_auc:\tmean: {np.mean(roc_auc_scores)}\tstd: {np.std(roc_auc_scores)}")

    # return {
    #     'roc_auc': [np.mean(roc_auc_scores), np.std(roc_auc_scores)],
    #     'UAR': [np.mean(uar_scores), np.std(uar_scores)],
    # }
    return {
        'roc_auc': roc_auc_scores,
        'UAR': uar_scores,
    }

if __name__ == '__main__':
    # extract_smile()
    # all_results = {}
    # for task in ['task1', 'task2', 'task3', 'all']:
    #     all_results[task] = train_svm(task)

    # print('\n\n****** All task results ******\n\n')
    # [print(k,'\t',v) for k,v in all_results.items()]

    # Results used in the paper
    results = {
        'task1': {'roc_auc': [0.6968896713615024, 0.6968896713615024, 0.6968896713615024], 'UAR': [0.6765698356807512, 0.6765698356807512, 0.6765698356807512]},
        'task2': {'roc_auc': [0.6354166666666666, 0.5625, 0.6875], 'UAR': [0.5625, 0.5520833333333333, 0.6354166666666667]},
        'task3': {'roc_auc': [0.5636363636363637, 0.55, 0.5636363636363637], 'UAR': [0.5, 0.5181818181818182, 0.5]},
        'all': {'roc_auc': [0.7311007957559681, 0.7026230474506336, 0.7280798703212497], 'UAR': [0.6782714412024757, 0.637783672266431, 0.6463675213675213]},
    }
    for task, v in results.items():
        for metric, scores in v.items():
            print(task, '\t', metric, f"\tmean: {np.mean(scores)}\tstd: {np.std(scores)}")

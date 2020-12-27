import sys
import os
import glob
import re
from itertools import product

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from conv_model import Conv_Model
from data_preprocessing import COVID_dataset

import wandb

def make_sample_weights(data_set):
    '''
    makes the weights for each of the classes
    '''
    data_set_index = data_set.data_index
    num_pos = data_set_index['label'].value_counts(normalize=True)[1]
    num_neg = data_set_index['label'].value_counts(normalize=True)[0]

    # norm_pos = num_pos/(num_pos + num_neg)
    # norm_neg = 1 - norm_pos

    pos_weight = num_neg/num_pos
    print('weight for covid', pos_weight)
    weights = torch.Tensor([pos_weight])
    # num_positive = batch_labels.sum()
    # neg_weight = num_positive / len(batch_labels)
    # pos_weight = 1 - num_positive / len(batch_labels)
    # weights = batch_labels.float()
    # weights[weights == 0] = neg_weight
    # weights[weights == 1] = pos_weight

    return weights


def run_train(epoch, loader_train, model, device, optimizer, weight, args):
    model.train()
    loader_train = tqdm(loader_train, position=0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    for i, (audio, label) in enumerate(loader_train):
        model.zero_grad()

        audio = audio.to(device)
        label = label.to(device)
        predicts_soft = model(audio)

        loss = criterion(predicts_soft, label.unsqueeze(1).float())
        loss.backward()

        # get accuracy
        predicts = torch.sigmoid(predicts_soft.detach())
        predicts = np.where(predicts.cpu().numpy()>0.5, 1, 0)
        score = f1_score(label.cpu().numpy(), predicts)

        # if scheduler is not None:
        #     scheduler.step()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']

        loader_train.set_description(
            (f'epoch: {epoch + 1}; f1: {score:.5f}; loss: {loss.item():.3f}'))

        if args.logger == 'wandb':
            wandb.log({"F1": score, "loss": loss.item()})


def eval(model, audio, label, device, criterion):

    audio = audio.to(device)
    label = label.to(device)
    predicts_soft = model(audio)    

    loss = criterion(predicts_soft, label.unsqueeze(1).float())
    # get accuracy
    predicts_soft = torch.sigmoid(predicts_soft).cpu().numpy()
    predicts = np.where(predicts_soft > 0.5, 1, 0)
    return loss, predicts, predicts_soft


def run_eval(epoch, loader_test, model, device, weight, args, do_save_model=True):
    model.eval()
    loader_test = tqdm(loader_test, position=0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    with torch.no_grad():
        y_hts = []
        ys = []
        logits_list = []
        losses = []
        for i, (audio, label) in enumerate(loader_test):
            label = label.to(device)
            if args.eval_type != 'maj_vote':
                loss, predicts, _ = eval(model, audio, label, device, criterion)
            else:
                clips = audio
                clip_loss, clip_predicts = 0, []
                for audio in clips:
                    loss, predicts, predicts_soft = eval(model, audio, label, device, criterion)
                    clip_loss += loss
                    clip_predicts.append((predicts, predicts_soft))

                # Aggregate predicts and loss
                loss = clip_loss / len(clips)
                positive = np.count_nonzero([c[0] for c in clip_predicts])
                votes = {'1': positive, '0': len(clip_predicts)-positive}
                # If its a tie, use logits
                if votes['1'] == votes['0']:
                    logits = (
                        sum([c[1] for c in clip_predicts if c[0].item() == 0]), # Negative
                        sum([c[1] for c in clip_predicts if c[0].item() == 1]), # Positive
                    )
                    predicts = np.argmax(logits).reshape(1,1)
                else:
                    predicts = np.array(int(max(votes.items(), key=lambda x: x[1])[0])).reshape(1,1)

            # for ROC-AUC
            average_logits = [c[1][0][0] for c in clip_predicts]
            logits_list.append(np.mean(average_logits))

            y_hts.append(predicts)
            ys.append(label.cpu().numpy())
            losses.append(loss.item())

        ys = np.concatenate(ys)
        y_hts = np.concatenate(y_hts)
        # logits_list = np.concatenate(logits_list)

        score = f1_score(ys, y_hts)
        acc = accuracy_score(ys, y_hts)
        recall = recall_score(ys, y_hts, average='macro')
        # ROC-AUC
        fpr, tpr, _ = roc_curve(ys, logits_list)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        loader_test.set_description((f'epoch: {epoch + 1}; test f1: {score:.5f}; test AUC {roc_auc:.5f}'
                                        f'test loss: {sum(losses)/len(losses):.3f}'))

        outputs = {"Test F1": score, "Test loss": sum(losses) / len(losses),
                        "epoch": epoch, "Test acc": acc, "test AUC": roc_auc,
                        "ROC": wandb.Image(plt), "UAR": recall,
                        "FPR": fpr, "TPR": tpr}
        # print(outputs)

        if args.logger == 'wandb':
            wandb.log(outputs)

        # Save model if required
        if args.dirname and args.do_train and do_save_model and epoch >= 29:
            roc_auc *= 100
            prevs = glob.glob(args.dirname+'/*F1*') + glob.glob(args.dirname+'/*AUC*')
            new_dirname = os.path.join(args.dirname, f"{args.task}_AUC-{roc_auc:.3f}")
            # Overwrite current worst if already k saved models and current model
            # better than an existing one
            if len(prevs) == args.save_model_topk:
                func = lambda p: float(re.search(
                    r"(?<=(?:F1)-)[0-9]+\.[0-9]{3}$" if 'F1' in p else r"(?<=(?:AUC)-)[0-9]+\.[0-9]{3}$",
                    p).group(0))
                scores = [func(p) for p in prevs]
                if roc_auc not in scores and roc_auc > min([func(p) for p in prevs]):
                    replace_dir = sorted(prevs, key=func)[0]
                    # issues with multiple threads and new_dirname already existing
                    if not os.path.exists(new_dirname):
                        os.rename(replace_dir, new_dirname)
                    save_model(model, new_dirname)
            else:
                if not os.path.exists(new_dirname):
                    os.mkdir(new_dirname)
                    save_model(model, new_dirname)
                    with open(os.path.join(new_dirname, 'epoch.txt'), 'w') as f:
                        f.write(str(epoch + 1))


def save_model(model, new_dirname):
    path = os.path.join(new_dirname, 'model.pt')
    torch.save(model.state_dict(), path)
    with open(os.path.join(new_dirname, 'config.txt'), 'w') as f:
        f.write(str(model))


def main(args):
    ''' Launch training and eval.
    '''
    hyp_params = dict(
        dropout=False,
        depth_scale=args.depth_scale,
        lr=args.lr,
        wsz=args.wsz,
        batch_size=args.batch_size,
        n_fft=args.nfft,
        sample_rate=args.sr,
        task=args.task,
        breathcough=args.breathcough,
    )
    # Init dir for saving models
    args.dirname = None
    if args.save_model_topk > 0:
        existing = glob.glob('models/run_*')
        idxs = [int(re.search(r"(?<=_)[0-9]+$", s).group(0)) for s in existing]
        ext = max(idxs) + 1 if idxs else 1
        args.dirname = os.path.join(os.getcwd(), 'models/run_'+str(ext))
        assert not os.path.exists(args.dirname), f"Dirname already exists:\nf{dirname}"
        os.mkdir(args.dirname)
        with open(os.path.join(args.dirname,'args.txt'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    # if args.logger == 'wandb':
    #     wandb.init(project=args.task, config=hyp_params)

    # dataset and dataloader:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise() if args.noise else NoneTransform(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    # Train and dev split
    device = 'cuda'
    print(args)
    train_dataset = COVID_dataset(
        dset='train',
        folds=args.folds['train'],
        task=args.task,
        transform=transform_train,
        window_size=hyp_params["wsz"],
        n_fft=args.nfft,
        sample_rate=args.sr,
        masking=args.masking,
        pitch_shift=args.pitch_shift,
        cross_val=args.cross_val,
        breathcough=args.breathcough
    )
    dev_dataset = COVID_dataset(
        dset='dev',
        folds=args.folds['dev'],
        task=args.task,
        transform=transform_test,
        eval_type=args.eval_type,
        window_size=hyp_params["wsz"],
        n_fft=args.nfft,
        sample_rate=args.sr,
        cross_val=args.cross_val,
        breathcough=args.breathcough
    )

    batch_size = args.batch_size
    train_weight = make_sample_weights(train_dataset).to(device)
    dev_weight = make_sample_weights(dev_dataset).to(device)

    loader_train = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

    loader_dev = DataLoader(dev_dataset,
                            batch_size=batch_size if args.eval_type != 'maj_vote' else 1,
                            shuffle=True,
                            num_workers=4)

    # Model
    model = Conv_Model(
        dropout=args.dropout,
        depth_scale=args.depth_scale,
        input_shape=(int(1024*args.nfft/2048)+1,int(94*args.wsz*args.sr/48000)),       # Dynamically adjusts for different input sizes
        device=device,
        breathcough=args.breathcough
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)
    # Load saved model
    if args.saved_model_dir:
        path = os.path.join(args.saved_model_dir, 'model.pt')
        model.load_state_dict(torch.load(path))
        print('Loading saved model:', args.saved_model_dir)

    if args.logger == 'wandb':
        wandb.watch(model)

    if args.do_train:
        for epoch in range(50):
            run_train(epoch, loader_train, model, device, optimizer, train_weight, args)
            run_eval(epoch, loader_dev, model, device, dev_weight, args)

    # if args.do_eval and not args.do_train:
    #     run_eval(0, loader_dev, model, device, dev_weight, args)

    if args.folds['test']:
        # Recover best saved model
        # prevs = glob.glob(args.dirname+'/*F1*') + glob.glob(args.dirname+'/*AUC*')
        # func = lambda p: float(re.search(
        #         r"(?<=(?:F1)-)[0-9]+\.[0-9]{3}$" if 'F1' in p else r"(?<=(?:AUC)-)[0-9]+\.[0-9]{3}$",
        #     p).group(0))
        # path = sorted(prevs, key=func)[-1]
        # with open(os.path.join(path, 'epoch.txt'), 'r') as f:
        #     best_epoch = int(f.readlines()[0])
        best_epoch = 10 # TODO

        train_dataset = COVID_dataset(
            dset='train',
            folds=args.folds['train'] + args.folds['dev'],
            task=args.task,
            transform=transform_train,
            window_size=hyp_params["wsz"],
            n_fft=args.nfft,
            sample_rate=args.sr,
            masking=args.masking,
            pitch_shift=args.pitch_shift,
            cross_val=args.cross_val,
            breathcough=args.breathcough
        )
        test_dataset = COVID_dataset(
            dset='test',
            folds=args.folds['test'],
            task=args.task,
            transform=transform_test,
            eval_type=args.eval_type,
            window_size=hyp_params["wsz"],
            n_fft=args.nfft,
            sample_rate=args.sr,
            cross_val=args.cross_val,
            breathcough=args.breathcough
        )
        train_weight = make_sample_weights(train_dataset).to(device)
        test_weight = make_sample_weights(test_dataset).to(device)

        loader_train = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)

        loader_test = DataLoader(test_dataset,
                                batch_size=batch_size if args.eval_type != 'maj_vote' else 1,
                                shuffle=True,
                                num_workers=4)

        # Model
        model = Conv_Model(
            dropout=args.dropout,
            depth_scale=args.depth_scale,
            input_shape=(int(1024*args.nfft/2048)+1,int(94*args.wsz*args.sr/48000)),       # Dynamically adjusts for different input sizes
            device=device,
            breathcough=args.breathcough
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if args.logger == 'wandb':
            wandb.watch(model)

        for epoch in range(best_epoch):
            run_train(epoch, loader_train, model, device, optimizer, train_weight, args)
            run_eval(epoch, loader_test, model, device, test_weight, args)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=5.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if self.std == 0.:
            return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)


class NoneTransform(object):
    """ Does nothing to the tensor, to be used instead of None
    
    Args:
        data in, data out, nothing is done
    """
    def __call__(self, x):
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='COVID_detector')
    # Hparam args
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0001)
    parser.add_argument('--dropout', type=bool, help='Drop out if true it is fixed at 0.5. Note for now it only applies to the first layer', default=False)
    parser.add_argument('--depth_scale', type=float, help='a parameter which multiplies the number of channels.', choices=[1, 0.5, 0.25, 0.125], default=1)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--wsz', type=int, help='Size of the audio clip window size (seconds).', default=1)
    parser.add_argument('--nfft', type=int, help='n_fft parameter', default=2048)
    parser.add_argument('--sr', type=int, help='sample rate parameter', default=48000)
    # augmentation args
    parser.add_argument('--masking', type=bool, help='do we perform time and frequency masking or not?', default=False)
    parser.add_argument('--pitch_shift',type=bool,help='perform a pitch shift provides the step size ot shift',default=False)
    parser.add_argument('--noise', type=bool, help='add gaussian noise to the specgram', default=False)
    # Sweep helpers
    parser.add_argument('--wsz_ds', type=str, help='Input as: window size:depth scale.', default="")
    parser.add_argument('--wsz_ds_bsz', type=str, help='Input as: window size:depth scale:batch size.', default="")
    parser.add_argument('--wsz_ds_bsz_nfft_sr', type=str, help='Input as: window size:depth scale:batch size:nfft:sample rate.', default="")
    # Config arg
    parser.add_argument('--logger', type=str, help='Type of logger to use.', choices=['default', 'wandb'], default='wandb')
    parser.add_argument('--save_model_topk', type=int, help='Save the k best performing models according to validation F1.', default=3)
    parser.add_argument('--do_train', type=bool, help='Run training loop.', default=True)
    parser.add_argument('--do_eval', type=bool, help='Run eval loop.', default=True)
    parser.add_argument('--eval_type', type=str, help='Type of eval to run', choices=['beginning', 'random','maj_vote'], default='maj_vote')
    parser.add_argument('--saved_model_dir', type=str, help='Path to dir containing saved model.', default="")
    # What task to do
    parser.add_argument('--task', type=str, help='what task do you want to perform?', default='task1', choices=['all', 'task1', 'task2', 'task3'])
    parser.add_argument('--location', type=str, help='where is the data', default='bitbucket', choices=['bitbucket', 'hpc'])
    parser.add_argument('--cross_val', type=bool, help='perform cross validation', default=False)
    parser.add_argument('--breathcough', type=bool, help='do we want to stack a cough and breath sample?', default=False)

    # Hack to use script with debugger by loading args from file
    if len(sys.argv) == 1:
        print('using txt')
        with open(os.getcwd()+'/args.txt', 'r') as f:
            args = argparse.Namespace(**json.loads(f.read()))
    else:
        print('not using txt')
        args = parser.parse_args()
        # with open(os.getcwd()+'/args.txt', 'w') as f:
        #     json.dump(vars(args), f, indent=4)

    print(args)
    if args.eval_type == 'maj_vote' and args.batch_size > 1:
        print(f"If running majority voting eval, eval batch size must be 1.")

    # Other processing
    assert not (args.do_train and args.saved_model_dir), f"Can't run training and load saved model. Investigate!"
    if args.saved_model_dir:
        saved_model_args_path = os.path.join(os.path.dirname(args.saved_model_dir), 'args.txt')
        with open(saved_model_args_path, 'r') as f:
            saved_model_args = json.loads(f.read())
        # Combine current args with saved model params
        hparams = ["lr", "dropout", "depth_scale", "wsz", "nfft", "sr"]
        for k,v in saved_model_args.items():
            if k in hparams:
                setattr(args, k, v)
            elif not hasattr(args, k):
                setattr(args, k, None)

    # Sweeps
    if hasattr(args, "wsz_ds") and args.wsz_ds:
        args.wsz = int(args.wsz_ds.split(':')[0])
        args.depth_scale = float(args.wsz_ds.split(':')[1])
    if hasattr(args, "wsz_ds_bsz") and args.wsz_ds_bsz:
        args.wsz = int(args.wsz_ds_bsz.split(':')[0])
        args.depth_scale = float(args.wsz_ds_bsz.split(':')[1])
        args.batch_size = int(args.wsz_ds_bsz.split(':')[2])
    if hasattr(args, "wsz_ds_bsz_nfft_sr") and args.wsz_ds_bsz_nfft_sr:
        args.wsz = int(args.wsz_ds_bsz_nfft_sr.split(':')[0])
        args.depth_scale = float(args.wsz_ds_bsz_nfft_sr.split(':')[1])
        args.batch_size = int(args.wsz_ds_bsz_nfft_sr.split(':')[2])
        args.nfft = int(args.wsz_ds_bsz_nfft_sr.split(':')[3])
        args.sr = int(args.wsz_ds_bsz_nfft_sr.split(':')[4])

    return args


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    if args.cross_val:
        args.cv_folds = 4
        cv_test_fold = 3
        for dev_fold in range(args.cv_folds - 1):
            print(f'Fold: {dev_fold}')
            if args.logger == 'wandb':
                run = wandb.init(project=args.task+'crossval_cough+breath_2', reinit=True)

            train_folds = [i for i in range(args.cv_folds - 1) if i != dev_fold]
            args.folds = {'train': train_folds, 'dev': [dev_fold], 'test': [cv_test_fold]}

            print(args)
            main(args)

    else:
        print(args)
        main(args)

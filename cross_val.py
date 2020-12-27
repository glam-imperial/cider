import os
import json
import argparse
import sys

from create_csv import create_csv
from main import main

import wandb


if __name__ == "__main__":

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
    #what task to do
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



    with open(os.getcwd()+'/args.txt', 'r') as f:
        args = argparse.Namespace(**json.loads(f.read()))


    for i in range(10):


        print(f'Fold: {i}')
        if args.logger == 'wandb':
            run = wandb.init(project=args.task+'crossval_cough+breath', reinit=True)
        # create new train/test splits
        create_csv(task=args.task+'crossval', location=args.location, dev_split=0, test_split=0.2, cross_val=args.cross_val)

        # train on new split config and evaluate with test

        main(args)

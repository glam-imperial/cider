import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from create_csv import create_csv
import random
import re


class COVID_dataset(Dataset):
    '''
    Custom COVID dataset.
    '''
    def __init__(self, dset, folds, eval_type='random',
                 transform=None, task='all',
                 window_size=1,
                 sample_rate=48000,
                 hop_length=512,
                 n_fft=2048,
                 masking=False,
                 pitch_shift=False,
                 cross_val=False,
                 breathcough=False):
        df = pd.read_csv(os.path.join('paths/cross_val', task+'.csv'))
        rows = df[df.fold.isin(folds)].index.tolist()
        np.random.shuffle(rows)
        self.data_index = df.iloc[rows]

        self.dset = dset
        self.root_dir = '/vol/bitbucket/hgc19/COVID_Audio_Diagnosis/KDD_paper_data'
        self.window_size = window_size * sample_rate
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.transform = transform
        self.eval_type = eval_type
        self.masking = masking
        self.pitch_shift = pitch_shift
        self.breathcough = breathcough

    def __len__(self):
        return len(self.data_index.index)

    def custom_transform(self, signal):
        """
        create log spectrograph of signal
        """
        stft = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        if self.masking:
            log_spectrogram = self.spec_augment(log_spectrogram)
        if self.transform:
            log_spectrogram = self.transform(log_spectrogram)
        return log_spectrogram

    def pad(self, signal):
        sample_signal = np.zeros((self.window_size,))
        sample_signal[:signal.shape[0],] = signal
        return sample_signal

    def __getitem__(self, index):

        # get path of chosen index
        audio_path = self.data_index['path'].iloc[index]
        label = self.data_index['label'].iloc[index]


        chunks = self.load_process(audio_path)
        # get path of a cough or breath sample which was provided by the same user
        # if a cough sample is provided need to get a breath sample and visa
        # versa

        if self.breathcough:
            # flag is used to insure that cough and breath are always passed to the model in the same
            # order.
            audio_path_2, label2, flag = self.return_pair(audio_path)
            if label2 != None:
                assert label == label2, 'pairs samples have mismatching labels, Investigate!'
            if audio_path_2 == None: # there is no pair (patient didn't give cough and breath)
                print('*'*30)
                print('No Pair!')
                label2 = label
                if self.dset == 'train' or self.eval_type != 'maj_vote':
                    chunks_2 = torch.zeros(chunks.size())
                else:
                    chunks_2 = [torch.zeros(chunks[0].size()) for i in range(len(chunks))]
            else:
                chunks_2 = self.load_process(audio_path_2)

            if self.dset == 'train' or self.eval_type != 'maj_vote':
                if flag == 'cough':
                    return torch.cat([chunks, chunks_2], dim=0), label
                elif flag == 'breath':
                    return torch.cat([chunks_2, chunks], dim=0), label

            else:
                if flag == 'cough':
                    return [torch.cat([i, j], dim=0) for i, j in zip(chunks, chunks_2)], label
                elif flag == 'breath':
                    return [torch.cat([j, i], dim=0) for i, j in zip(chunks, chunks_2)], label

        return chunks, label

    def load_process(self, audio_path):
        # load the data
        signal, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        # perform pitch shift:
        if self.pitch_shift:
            step = np.random.uniform(-6,6)
            signal = librosa.effects.pitch_shift(
                signal, sample_rate, step)

        # For train, sample random window size from audiofile
        if self.dset == 'train' or self.eval_type != 'maj_vote':
            # Apply padding if necessary. Else sampsle random window.
            if signal.shape[0] <= self.window_size:
                sample_signal = self.pad(signal)
            else:
                if self.eval_type == 'random':
                    rand_indx = np.random.randint(0, signal.shape[0] - self.window_size)
                else:
                    rand_indx = 0
                sample_signal = signal[rand_indx:rand_indx + self.window_size]

            # perform transformations
            sample_signal = self.custom_transform(sample_signal)

            return sample_signal
        # For eval/test, chunk audiofile into chunks of size wsz and
        # process and return all
        else:
            chunks = np.array_split(signal, int(np.ceil(signal.shape[0] / self.window_size)))
            def process_chunk(chunk):
                if chunk.shape[0] <= self.window_size:
                    sample_signal = self.pad(chunk)
                chunk =  self.custom_transform(sample_signal)
                return chunk
            chunks = [process_chunk(chunk) for chunk in chunks]

            return chunks

    def spec_augment(self,
                     spec: np.ndarray,
                     num_mask=2,
                     freq_masking_max_percentage=0.15,
                     time_masking_max_percentage=0.3):

        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0,
                                   high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0,
                                   high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0

        return spec

    def return_pair(self, audio_path):
        '''
        function that given a path to an audio file of a person coughing returns a sample of the same
        person coughing or breathing (depending on whether the original sample is cough or breath)
        inputs: audio_path --> str
        output: audio_path_2 --> str, label2 --> str
        '''

        if 'web' in audio_path:
            if 'breathe' in audio_path:
                audio_path_2 = audio_path.replace('breathe', 'cough')
                flag = 'cough'
            elif 'cough' in audio_path:
                num_cough = re.findall('cough', audio_path)
                if len(num_cough) == 1:
                    audio_path_2 = audio_path.replace('cough', 'breathe')
                else:
                    audio_path_2 = self.nth_repl(audio_path, 'cough', 'breathe', 2)

                flag = 'breath'

            else:
                raise Exception('This should not be a possibility - path should contain breathe of cough')

            assert self.data_index['path'].isin([audio_path_2]).any(), f'{audio_path_2} not in data'
            # getting the label to check that it is the same
            label2 = self.data_index.loc[self.data_index['path'] == audio_path_2]['label'].iloc[0]

            return audio_path_2, label2, flag

        elif 'android' in audio_path:
            # this is more complicated as breathe and cough samples have different unique codes so can't just
            # swap breathe with cough as in web
            if 'breaths' in audio_path:
                # folder -> breaths
                # file --> breath
                audio_path_2 = audio_path.replace('breath', 'cough', 1)
                audio_path_2 = audio_path_2.replace('breaths', 'cough', 1)
                flag = 'cough'
            elif 'cough' in audio_path:
                num_cough = re.findall('cough', audio_path)
                flag = 'breath'
                if len(num_cough) == 2:
                    audio_path_2 = audio_path.replace('cough', 'breath', 1)
                    audio_path_2 = audio_path_2.replace('cough', 'breaths', 1)
                else:
                    audio_path_2 = self.nth_repl(audio_path,'cough', 'breath', 2)
                    audio_path_2 = self.nth_repl(audio_path_2, 'cough', 'breaths', 2)
            else:
                raise Exception(
                    'This should not be a possibility - path should contain breathe of cough'
                )
            audio_path_2 = re.sub("[0-9]{13}", "", audio_path_2)
            audio_path_2 = audio_path_2.replace('.wav', "")
            rows_to_swap = self.data_index[
                self.data_index['path'].str.contains(
                    audio_path_2)]


            if len(rows_to_swap["path"].values.tolist()) == 0: # no pairs pad with zeros
                return None, None, flag
            audio_path_2 =  np.random.choice(rows_to_swap["path"].values.tolist())
            assert self.data_index['path'].isin(
                [audio_path_2]).any(), f'{audio_path_2} not in data'

            label2 = self.data_index.loc[self.data_index['path'] ==
                                         audio_path_2]['label'].iloc[0]

            return audio_path_2, label2, flag

        else:
            raise Exception(
                'This should not be a possibility - path should contain breathe of cough'
            )

    def nth_repl(self, s, sub, repl, n):
        find = s.find(sub)
        # If find is not -1 we have found at least one match for the substring
        i = find != -1
        # loop util we find the nth or we find no match
        while find != -1 and i != n:
            # find + 1 means we start searching from after the last match
            find = s.find(sub, find + 1)
            i += 1
        # If i is equal to n we found nth match so replace
        if i == n:
            return s[:find] + repl + s[find + len(sub):]
        return s


if __name__ == "__main__":
    test_dataset = COVID_dataset('dev', None)
    for i in tqdm(range(len(test_dataset))):
        sample, label = test_dataset[i]
        print(sample.shape)
        break
        plt.figure()
        librosa.display.specshow(sample,
                                sr=48000,
                                hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram (dB)")
        path_to_save = 'figs/log_spectrogram'+str(i)+'.png'
        plt.savefig(path_to_save)
        plt.close()
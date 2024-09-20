import argparse
import torch 
import glob
import json
import pickle
import numpy as np 
import torchaudio
from tqdm import tqdm
from data import DataProcessor
from tools import get_monophone_mid
from s3prl.hub import *

def main(model_name, save_pth, data_pth, extra_class):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load upstream model 
    upstream_model = eval(model_name)().to(device)
    upstream_model.eval()
    
    # Wav paths 
    wav_pths = list(glob.glob(data_pth+"**/*.flac", recursive=True))

    # Load phoneme force align result 
    mfa_json = 'info/dev-clean-framewise-phone-merge-20ms.json'
    with open(mfa_json, 'r') as fp:
        mfa = json.load(fp)
    print(f"Load MFA result from {len(mfa)} utterances")

    NLAYER = 12
    D = 3072
    N_phone = 41

    N_list = {
        'phone-type': N_phone,
        'gender': N_phone*2,
        'pitch': N_phone*3
    }
    N = N_list[extra_class]

    if extra_class == 'phone-type':
        record = [torch.zeros((N, D)) for i in range(12)] 
        record_n = [[0 for i in range(N)] for i in range(12)]
    elif extra_class == 'gender':
        record = [torch.zeros((N, D)) for i in range(12)] 
        record_n = [[0 for i in range(N*2)] for i in range(12)]
        extra_info = 'info/libri-dev-spk-gender.json'
        with open(extra_info, 'r') as fp:
            gender_dict = json.load(fp)
    elif extra_class == 'pitch':
        record = [torch.zeros((N, D)) for i in range(12)] 
        record_n = [[0 for i in range(N*3)] for i in range(12)]
        extra_info = 'info/pitch-discrete-dev-clean.json'
        with open(extra_info, 'r') as fp:
            pitch_dict = json.load(fp)

    for pth in tqdm(wav_pths): 
        key = pth.split('/')[-1].split('.')[0]
        phoneme = mfa[key]
        if extra_class == 'gender':
            gender = key.split('-')[0]
            g = 0 if gender_dict[gender] == 'M' else 1 
        elif extra_class == 'pitch':
            pitch = pitch_dict[key]
            gap = len(phoneme)-len(pitch)
            if gap > 0:
                pitch = [-1]*gap + pitch
            elif gap < 0:
                pitch = pitch[gap:]
        check_idx = get_monophone_mid(phoneme)
        check_phone = [phoneme[idx] for idx in check_idx]
        if extra_class == 'pitch':
            check_pitch = [pitch[idx] for idx in check_idx]
        # Forward models to get FFC layer results
        wav, sr = torchaudio.load(pth)
        input_wavs = [wav.view(-1).to(device)]
    
        with torch.no_grad():
            out = upstream_model(input_wavs)
            fc_results = out["fc_results"]

        for layer_idx, (fc1, fc2) in enumerate(fc_results):
            check_keys = fc1.squeeze(1)[check_idx,:]
            tau = round(D*0.01)
            for k in range(len(check_idx)):
                keys = torch.abs(check_keys[k]) # D 
                assert D == len(keys)
                p = check_phone[k] 
                _, topk_indices = torch.topk(keys, tau)
                topk_indices = topk_indices.cpu()
                if extra_class == 'phone-type':
                    record[layer_idx][p, topk_indices] += 1 
                    record_n[layer_idx][p] += 1 
                elif extra_class == 'gender':
                    record[layer_idx][g*N_phone+p, topk_indices] += 1 
                    record_n[layer_idx][g*N_phone+p] += 1 
                elif extra_class == 'pitch':
                    pc = check_pitch[k]
                    if pc == -1:
                        continue
                    record[layer_idx][pc*N_phone+p, topk_indices] += 1 
                    record_n[layer_idx][pc*N_phone+p] += 1 

    for idx in range(12):
        for pidx in range(N):
            if record_n[idx][pidx] != 0:
                record[idx][pidx,:] /= record_n[idx][pidx]
        record[idx] = np.array(record[idx])

    with open(save_pth, 'wb') as fp:
        pickle.dump(record, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', help='Model name')
    parser.add_argument('-s', '--save-pth', help='Result save path')
    parser.add_argument('-d', '--data-pth', help='Dataset directory')
    parser.add_argument('-c', '--extra-class', choices=['phone-type', 'gender', 'pitch'])
    args = parser.parse_args()
    main(**vars(args))

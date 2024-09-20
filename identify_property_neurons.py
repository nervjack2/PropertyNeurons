import argparse 
import json 
import numpy as np 
import os
import pickle
import matplotlib.pyplot as plt
from tools import sort_by_same_phone, sort_voiced_unvoiced

group_name = {
    'phone-type': ['vowels', 'voiced-consonants', 'unvoiced-consonants'],
    'gender': ['male', 'female'],
    'pitch': ['<129.03Hz', '129.03-179.78Hz', '>179.78Hz'],
}

def find_group_neurons(pkl_pth, sigma, phone_idx, s_idx, extra_class):
    n_phone_group_dict = {
        'phone-type': 1,
        'gender': 2,
        'pitch': 3
    }
    n_phone_group = n_phone_group_dict[extra_class]

    results = {}
    with open(pkl_pth, 'rb') as fp:
        data = pickle.load(fp)
    n_layer = len(data)
    results = {}
    for idx in range(n_layer):
        v_datas = []
        NPHONE = data[idx].shape[0] // n_phone_group
        for i in range(n_phone_group):
            sample_idx = [NPHONE*i+x for x in phone_idx]
            v_datas.append(data[idx][sample_idx,:]) 
        if extra_class == 'phone-type':
            # Split according to broad phone type 
            new_v_datas = []
            for i in range(3):
                new_v_datas.append(v_datas[0][s_idx[i]:s_idx[i+1],:])
            v_datas = new_v_datas
        n_group = len(v_datas)
        # Calculate the common activated neurons (notated as N in the paper) for a specfic group (e.g. Male, Female)
        keys_group = {}
        for g_idx in range(n_group):
            n_phone, D = v_datas[g_idx].shape
            random_baseline = round(D*0.01)/D
            num_dim = [0 for i in range(n_phone)]
            for i in range(n_phone):
                num_dim_meaningful = np.sum(v_datas[g_idx][i] > random_baseline)
                num_dim[i] = num_dim_meaningful
            # Activation pattern of different phones
            indices = [[i for i in range(D)] for i in range(n_phone)]
            for i in range(n_phone):
                indices[i] = sorted(indices[i], key=lambda x: v_datas[g_idx][i][x], reverse=True)[:num_dim[i]]
            # Calculate the probability of being activated given a specific group
            keys = {}
            for i in range(n_phone):
                nd = len(indices[i])
                for j in range(nd):
                    keys[indices[i][j]] = keys.get(indices[i][j], 0)+1 
            n_keys = len(keys)
            for k, v in keys.items():
                keys[k] = v/n_phone
            # Get neurons which probability higher than a threshold (0.8 in the paper)
            n_match = np.sum(np.array(list(keys.values())) >= sigma)
            print(f"There are {n_match} detected keys for group {g_idx} of property {extra_class} in layer {idx}.")
            indices = sorted(keys.keys(), key=lambda x: keys[x], reverse=True)[:n_match]
            keys_group[group_name[extra_class][g_idx]] = indices
        results[idx+1] = keys_group
    return results

def find_property_neurons(group_neurons):
    layer_keys = {}
    for l in group_neurons.keys():
        # Creating lookup table for each group
        group_keys = {}
        for g in group_neurons[l].keys():
            group_keys[g] = {index: 1 for index in group_neurons[l][g]} 
        # Calculating property neurons
        ps_keys = []
        for g1 in group_neurons[l].keys():
            for index1 in group_neurons[l][g1]:
                flag = True 
                for g2 in group_neurons[l].keys():
                    if g1 == g2:
                        continue
                    if index1 in group_keys[g2]:
                        flag = False 
                        break 
                if flag:
                    ps_keys.append(index1)
        print(f"There are {len(ps_keys)} property neurons in {l} layer.")    
        layer_keys[l] = ps_keys
    return layer_keys

def main(pkl_pth, save_pth, extra_class, sigma=0.8):
    phone_label_pth = 'info/dev-clean-label-merge.json'
    with open(phone_label_pth, 'r') as fp:
        phone_label = json.load(fp)
    sort_phone = sorted(phone_label, key=lambda x: phone_label[x][1], reverse=True)
    sort_phone_same = sort_by_same_phone(sort_phone)
    sort_phone_unvoiced, num_type = sort_voiced_unvoiced(sort_phone_same)
    split_idx = [0]+[sum(num_type[:i+1]) for i in range(len(num_type))]
    n_phone = len(sort_phone)
    phone_idx = [phone_label[x][0] for x in sort_phone_unvoiced]
    group_neurons = find_group_neurons(pkl_pth, sigma, phone_idx, split_idx, extra_class)
    property_neurons = find_property_neurons(group_neurons) 
    with open(save_pth, 'w') as fp:
        json.dump(property_neurons, fp, indent=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl-pth', help='Activating probability .pkl path.')
    parser.add_argument('-s', '--save-pth', help='Path to save the index of property neurons.')
    parser.add_argument('-c', '--extra-class', choices=['phone-type', 'gender', 'pitch'])
    parser.add_argument('--sigma', default=0.8, type=float)
    args = parser.parse_args()
    main(**vars(args))
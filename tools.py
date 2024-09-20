import numpy as np
import math 

def find_ps_keys(x):
    data = {}
    for p in x.keys():
        layer_keys = {}
        for l in x[p].keys():
            # Creating lookup table for each group
            group_keys = {}
            for g in x[p][l].keys():
                group_keys[g] = {index: 1 for index in x[p][l][g]} 
            # Calculating property-specific keys
            ps_keys = []
            for g1 in x[p][l].keys():
                for index1 in x[p][l][g1]:
                    flag = True 
                    for g2 in x[p][l].keys():
                        if g1 == g2:
                            continue
                        if index1 in group_keys[g2]:
                            flag = False 
                            break 
                    if flag:
                        ps_keys.append(index1)
            print(f"There are {len(ps_keys)} property-specific keys for property {p} in {l} layer.")    
            layer_keys[l] = ps_keys
        data[p] = layer_keys 
    return data 

def get_DBI(data_2d, nc, s_idx=None):
    N = data_2d.shape[0]
    n_phone = N // nc
    data = []
    for i in range(nc):
        if s_idx == None:
            data.append(data_2d[i*n_phone:(i+1)*n_phone,:])
        else:
            data.append(data_2d[s_idx[i]:s_idx[i+1],:])
    v = np.zeros((nc,2)) # Centroid
    for i in range(nc):
        v[i,:] = data[i].mean(axis=0)
    d = np.zeros((nc,nc)) # Distance between centroid
    for i in range(nc):
        for j in range(nc):
            d[i,j] = np.linalg.norm(v[i]-v[j])
    s = np.zeros((nc)) # Intra-class distance
    for i in range(nc):
        m = len(data[i])
        acc = 0
        for j in range(m):
            acc += np.linalg.norm(data[i][j,:]-v[i])
        s[i] = acc/m
    R = np.zeros((nc,nc))
    for i in range(nc):
        for j in range(nc):
            if i != j:
                R[i,j] = (s[i]+s[j])/d[i,j]
    R_max = np.max(R,axis=0)
    DB = sum(R_max)/nc
    return DB

def get_silhouette_score(data_2d, nc, s_idx=None):
    N = data_2d.shape[0]
    n_phone = N // nc
    data = []
    for i in range(nc):
        if s_idx == None:
            data.append(data_2d[i*n_phone:(i+1)*n_phone,:])
        else:
            data.append(data_2d[s_idx[i]:s_idx[i+1],:])
    data_len = [len(data[i]) for i in range(nc)]
    a = [np.zeros((data_len[i])) for i in range(nc)]
    for i in range(nc):
        for j in range(data_len[i]):
            dis = 0
            for k in range(data_len[i]):
                if j != k:
                    dis += np.linalg.norm(data[i][j,:]-data[i][k,:])
            dis /= (data_len[i]-1)
            a[i][j] = dis 
    b = [np.zeros((data_len[i])) for i in range(nc)]
    for i in range(nc):
        for j in range(data_len[i]):
            min_dis = math.inf
            for k in  range(nc):
                if i == k:
                    continue 
                dis = 0
                for s in range(data_len[k]):
                    dis += np.linalg.norm(data[i][j,:]-data[k][s,:])
                dis /= data_len[k]
                min_dis = min(min_dis, dis)
            b[i][j] = min_dis
    
    aa, bb = [], []
    for i in range(nc):
        aa += list(a[i])
        bb += list(b[i])
    s = [0 for i in range(N)]
    for i in range(N):
        s[i] = (bb[i]-aa[i])/max(bb[i],aa[i])
    return sum(s)/N

def get_monophone_mid(phoneme):
    pidx = []
    pre_x = phoneme[0]
    start_x = 0 
    for idx, x in enumerate(phoneme):
        if x != pre_x:
            end_x = idx-1 
            mid_x = (start_x + end_x) // 2 
            pidx.append(mid_x)
            start_x = idx 
        pre_x = x 
    return pidx 

def parse_num(k):
    return ''.join([x for x in k if not x.isdigit()])

def sort_by_same_phone(phoneme):
    keys = {}
    for p in phoneme:
        n = parse_num(p)
        if n not in keys:
            keys[n] = [p]
        else:
            keys[n].append(p)
    print(f"There are {len(keys)} group of phone after merging")
    new_list = []
    for k, v in keys.items():
        new_list += v 
    return new_list

def sort_voiced_unvoiced(phoneme):
    # ARPABET
    phoneme_type = {
        "vowels": [
            'IY', 'IH', 'EH', 'AE', 'AA', 'AH', 'AO', 
            'UH', 'EY', 'AY', 'OY', 'AW', 'OW', 'ER', 'UW'
        ],
        "voiced-consonants": [
            'B', 'D', 'G', 'JH', 'DH', 'Z', 'ZH', 
            'V', 'M', 'N', 'NG', 'L', 'R', 'W', 'Y'
        ],
        "unvoiced-consonants": [
            'P', 'T', 'K', 'CH', 'TH', 'S', 'SH', 'F', 'HH'
        ],
    }
    voiced_v = []
    voiced_c = []
    unvoiced_c = []
    for p in phoneme:
        n = parse_num(p) 
        if n in phoneme_type["vowels"]:
            voiced_v.append(p)
        elif n in phoneme_type["voiced-consonants"]:
            voiced_c.append(p)
        elif n in phoneme_type["unvoiced-consonants"]:
            unvoiced_c.append(p)
    num_type = [len(voiced_v), len(voiced_c), len(unvoiced_c)]
    print(f"There are {sum(num_type)} keys after filtering out silence and unrecognized phone")
    return voiced_v + voiced_c + unvoiced_c, num_type
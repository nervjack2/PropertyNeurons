import numpy as np
import math 

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
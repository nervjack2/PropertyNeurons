# Property Neurons
The official implementation of SLT 2024 paper 
[Property Neurons in Self-Supervised Speech Transformers](https://arxiv.org/abs/2409.05910)

## Installation
1. Install [s3prl](https://github.com/s3prl/s3prl) to S3PRL_PATH
2. Copy modified files into S3PRL_PATH
```
cp -r s3prl/upstream/* S3PRL_PATH/s3prl/upstream/
```
3. Run this repository in s3prl environment 

## Example commands on LibriSpeech dev-clean
1. Calculate activating probability as mentioned in Section 3.2 of the paper
  ```
  python3 match_phone_s3prl.py -m [MODEL_NAME] -s [SAVE_PATH] -d [LIBRI_ROOT]/dev-clean/ -c [CONDITION]
  ```
  MODEL_NAME: Upstream model tag in s3prl. Choose from hubert_base, wavlm_base, and wav2vec2_base_960.
  
  SAVE_PATH: Path to save activating probability. 
  
  LIBRI_ROOT: Root path of LibriSpeech.
  
  CONDITION: Condition of activating probabiltiy. Choose from phone-type, gender, and pitch.

2. Identifying property neurons as mentioned in Section 3.2 and Section 4.1
  ```
  python3 identify_property_neurons.py -p [PROB_PATH] -s [SAVE_PATH] -c [CONDITION]
  ```
  PROB_PATH: Path to activating probability
  
  SAVE_PATH: Path to save property neurons
  
  CONDITION: Condition of activating probabiltiy. Choose from phone-type, gender, and pitch.

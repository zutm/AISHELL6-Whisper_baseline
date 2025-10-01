# AISHELL6-Whisper baseline

[AISHELL6-whisper: A Chinese Mandarin Audio-visual Whisper Speech Dataset with Speech Recognition Baselines](https://arxiv.org/abs/2509.23833)

## Introduction

We release [AISHELL6-Whisper](https://zutm.github.io/AISHELL6-Whisper), an audio-visual whisper speech dataset in Chinese Mandarin. We propose an audio-visual speech recognition (AVSR) baseline based on the [Whisper-Flamingo](https://github.com/roudimit/whisper-flamingo)  framework, which integrates visual features into the Whisper speech recognition and translation model with gated cross attention. For simultanous whisper speech and normal speech recognition, We integrates a parallel training strategy to align embeddings across speech types, and employs a projection layer to adapt to whisper speech's spectral properties.



**Important:** to use the project, a minor change is required in the AV-HuBERT code following Whisper-Flamingo.
Specfically, comment out [line 624](https://github.com/facebookresearch/av_hubert/blob/e8a6d4202c208f1ec10f5d41a66a61f96d1c442f/avhubert/hubert.py#L624) and add this after line 625: `features_audio = torch.zeros_like(features_video)`. This is needed since we only use video inputs with AV-HuBERT, not audio. Otherwise you will get an error about 'NoneType' object. 

# Download our AISHELL6-Whisper dataset
Download our AISHELL6-Whisper dataset at [https://zutm.github.io/AISHELL6-Whisper](https://zutm.github.io/AISHELL6-Whisper)

# Environment
Please follows the virtual environment preparation method in [virtual-environment-for-training-and-testing](https://github.com/roudimit/whisper-flamingo/blob/main/README.md#virtual-environment-for-training-and-testing) to build up the envirnment.
Create a fresh virtual environment:
```
conda create -n AISHELL6-Whisper_baseline python=3.8 -y
conda activate AISHELL6-Whisper_baseline
```

Clone av_hubert's repo and install Fairseq:
```
python -m pip install pip==24.0
pip --version
git clone -b muavic https://github.com/facebookresearch/av_hubert.git
cd av_hubert
git submodule init
git submodule update
# Install av-hubert's requirements
pip install -r requirements.txt
# Install fairseq
cd fairseq
pip install --editable ./
cd ../..
```
Install extra packages used in our project:
```
pip install numpy==1.22 tiktoken==0.5.2 pytorch-lightning==2.1.3 numba==0.58.1 transformers==4.36.2 evaluate ffmpeg-python pandas wget librosa tensorboardX
```

# Preparation
### Step1: Prepare {train,valid,test}.tsv and {train,valid,test}.wrd
Prepare {train,valid,test}.tsv and {train,valid,test}.wrd in "data/AISHELL6-Whisper/{train,valid,test}" following AV-Hubert. Please replace the 'AISHELL6-Whisper_dataset_dir' with the path of AISHELL6-Whisper dataset.
```
python preparation/AISHELL6-Whisper_manifest.py --base_dir AISHELL6-Whisper_dataset_dir --output_dir data/AISHELL6-Whisper
```
### Step2: Prepare the correspondence file for whisper and normal speech
Use the command to create "data/AISHELL6-Whisper/w2n.txt" and "data/AISHELL6-Whisper/n2text.txt" to prepare the correspondence file for whisper and normal speech, or you can use the 'w2n.txt' in our realeased dataset. Please replace the 'AISHELL6-Whisper_dataset_dir' with the path of AISHELL6-Whisper dataset. 
```
python preparation/whisper2normal.py --base_dir AISHELL6-Whisper_dataset_dir --output_dir data/AISHELL6-Whisper
```
### Step3: Download checkpoint of AV-Hubert to train audio-visual whisper speech recognition basline on AISHELL6-Whisper
mkdir models
wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt -P models

# Training

### Step 1: Fine-tune audio-only Whisper on AISHELL6-Whisper
Training Command:
```
python -u whisper_ft_normalwhisper.py config/audio/audio_zh_largev3.yaml
```
We train for two epochs at this stage.

### Step 2: Train audio-visual whisper speech recognition basline on AISHELL6-Whisper
Training Command:
```
python -u whisper_ft_video_normalwhisper.py config/audio-visual/av_zh_largev3.yaml
```
We train for four epochs at this stage.

### Training progress
Model weights will be saved in `models/checkpoint`.
Tensorboard can be opened to monitor several metrics.
```
cd log
tensorboard --logdir .  --port 6008
```


# Decoding Script
We calculate CER and WER following [compute-cer.py](https://github.com/wenet-e2e/wenet/blob/main/tools/compute-cer.py) and [compute-wer.py](https://github.com/wenet-e2e/wenet/blob/main/tools/compute-wer.py).

Set configuration parameters in 'eval.sh' (see `whisper_decode_video.py` for argument details):
- Use `--noise-snr 1000` to evaluate in clean conditions.
- For GPU without fp16, and for cpu, use `--fp16 0`.
- For decoding on whisper speech, add `--align`.
- For audio-only decoding, use `--modalities asr`, for audio-visual decoding, use `--modalities avsr`.
- In the paper we report results with beam size 1.

```
bash eval.sh 
```


# Acknowledgments
This code based is based on the following repos: [Whisper-Flamingo](https://github.com/roudimit/whisper-flamingo), [Whisper](https://github.com/openai/whisper), [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [wenet](https://github.com/wenet-e2e/wenet).

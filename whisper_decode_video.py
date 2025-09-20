
import os
import json
import argparse
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import pandas as pd
import whisper
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
import editdistance
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    load_data,
    load_video_feats,
    add_noise,
    WhisperVideoCollatorWithPadding,
)
from utils_batch_samplers import LengthBatchSampler
from fairseq.scoring.wer import WerScorer, WerScorerConfig
import sacrebleu
import librosa
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--lang', default='ru', type=str, help='decoding language')
parser.add_argument('--model-type', default='medium', help='Whisper model size, note: large-v2, not large')
parser.add_argument('--noise-snr',  default=1000, type=int, help='>100 is off, so 1000 means clean audio')
parser.add_argument('--noise-fn', default=None, help='testing noise file')
parser.add_argument('--beam-size', default=1, type=int, help='if 1 use greedy else beam search')
parser.add_argument('--modalities', default="avsr", help='asr for audio-only, avsr for audio-visual')
parser.add_argument('--use_av_hubert_encoder', default=0, type=int, help='if 1 use av hubert encoder')
parser.add_argument('--av_fusion', default="", help='N/A for whisper, "separate" for Whisper-Flamingo')
parser.add_argument('--fp16', default=1, type=int, help='if 1 use fp16, if 0 use GPU if available or cpu if not')
parser.add_argument('--checkpoint-path', default=None, help='path to load the checkpoint from')
parser.add_argument('--decode-path', default="decode/", help='path to save the decode results')
parser.add_argument('--whisper-path', default="/DKUdata/2023/wangyuan/data1/avsr/whisper-flamingo/openai/", help='path to download OpenAI whisper weights')
parser.add_argument('--av-hubert-path', default="av_hubert/avhubert/", help='path to avhubert code')
parser.add_argument('--av-hubert-ckpt', default="models/large_noise_pt_noise_ft_433h_only_weights.pt", 
                                        help='path to avhubert ckpt (needed to load the model architecture)')
parser.add_argument('--task', default='transcribe', type=str, help='transcribe, En-X, X-En')
parser.add_argument('--normalizer', default='fairseq', type=str, help='whisper OR fairseq')
parser.add_argument('--use-original-whisper', default=0, type=int, 
                                        help='if 1, ignore checkpoint-path and use original whisper')
parser.add_argument('--align', action='store_true', help='use projection layer')
                                        
args = parser.parse_args()
SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)



class MuavicVideoDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None, train=False, noise_snr=0) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob 
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.train = train
        self.noise_snr = noise_snr
        print("Dataloader max length : {}".format(max_length))
        print("Loaded {} noise wavs".format(len(self.noise_fn)))

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        lang, audio_path, text, _ = self.audio_info_list[id]
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
        if np.random.rand() > self.noise_prob: 
            audio = audio.astype(np.float32)
        else: 
            audio = add_noise(audio, self.noise_fn, noise_snr=self.noise_snr).astype(np.float32)
        audio_frames = len(audio.flatten()) // 160
        # pad audio to cfg.audio_max_length (longer samples filtered out already)
        if self.max_length != None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels) # freq by time
    
        if self.spec_augment:
            if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T # expects time by freq
            elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T # expects time by freq
            else:
                raise NotImplementedError 

        # Seems like Whisper decode always predicts first token with space, so add a space in the beginning
        # dec_input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(" " + text)
        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)], 
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        video_path = audio_path.replace('audio/hfm', 'video/camera').replace('.wav', '.avi')

        
        frame_count = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        if not os.path.exists(video_path) or frame_count == 0:
            video=np.zeros((round(len(audio.flatten()) / 16000 * 25),88, 88, 1), dtype=np.float32)
        else:
            video = load_video_feats(video_path, train=self.train)
            video = video.astype(np.float32)
        # Trim some videos longer than the audio
        max_video_len = round(len(audio.flatten()) / 16000 * 25)
        if len(video) > max_video_len:
            video = video[:max_video_len]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "video": video
        }


# process lang and task
visible = True if 'visible' in args.lang else False
if 'visible' in args.lang:
    args.lang = args.lang.replace('-visible', '')


audio_transcript_pair_list = load_data(480000, 350, [args.lang],
                                       include_audio_lens=True)

test_dataset =  audio_transcript_pair_list['test']


# test_dataset = [[i[0], i[1], i[2], i[3]] for i in test_dataset] # use original paths
multilingual = True if 'large' in args.model_type or 'en' not in args.model_type else False
print("Multilingual tokenizer : {}".format(multilingual))

# We use the transcribe token (not translate) for En-X to enable new capabilities
task = 'transcribe' if args.task == 'X-En' else 'transcribe'
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=multilingual, task=task) 
special_token_set = set(tokenizer.special_tokens.values())

print(args.checkpoint_path)
args.checkpoint_path= None if args.use_original_whisper else args.checkpoint_path
# If the original Whisper from OpenAI is used, crop / pad the audio to 30s
dataset = MuavicVideoDataset(test_dataset, 
                                tokenizer, 
                                SAMPLE_RATE, 
                                args.model_type,
                                max_length=None if args.checkpoint_path else SAMPLE_RATE * 30,
                                spec_augment="", # no spec augment
                                noise_prob=1 if args.noise_snr != 1000 else 0,
                                noise_fn = args.noise_fn,
                                train=False, # video center crop, no flip
                                noise_snr=args.noise_snr,)   

# For beam size of 1, use batch decoding with ~40s of audio per batch
# For beam size >1, each audio sample is decoded separately
length_sorter = LengthBatchSampler(batch_bins=SAMPLE_RATE * 40 if args.checkpoint_path and \
                                   args.beam_size == 1 else 1,
                            shapes=[i[3] for i in test_dataset],
                            sort_in_batch='descending',
                            sort_batch='descending',
                            drop_last=False)


dataloader = torch.utils.data.DataLoader(dataset,
                    num_workers=8,
                    collate_fn=WhisperVideoCollatorWithPadding(),
                    batch_sampler=length_sorter)

print("Loading Whisper")
whisper_model = whisper.load_model(args.model_type, 
                                   download_root=args.whisper_path, 
                                   video=True if args.av_fusion != "None" else 0,
                                   video_model_path=args.av_hubert_ckpt,
                                   av_hubert_path=args.av_hubert_path,
                                   av_hubert_encoder=args.use_av_hubert_encoder,
                                   av_fusion=args.av_fusion,
                                   add_gated_x_attn=1 if args.av_fusion == 'separate' else 0)

if args.checkpoint_path is not None:
    print("Loading checkpoint")
    state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    state_dict = state_dict['state_dict']
    state_dict_updated = {k[6:]: v  for k, v in state_dict.items()} # remove 'model.'
    try: # newer models have learnable scaler init 1
        whisper_model.load_state_dict(state_dict_updated) 
    except BaseException as e:
        print(str(e))
        print("Loading weights with strict=False")
        whisper_model.load_state_dict(state_dict_updated, strict=False) 

options = whisper.DecodingOptions(task=task, language=args.lang, fp16=args.fp16, without_timestamps=True, 
                                  beam_size=None if args.beam_size == 1 else args.beam_size,)

if args.checkpoint_path is not None:
    out_path = '{}/{}/{}/test/{}/snr-{}/visible-{}/beam-{}/{}' \
                .format(args.decode_path,args.checkpoint_path, args.lang, args.modalities, args.noise_snr, int(visible), 
                        args.beam_size, args.noise_fn.split('/')[-2])
else:
    out_path = '{}/{}/{}/test/{}/snr-{}/visible-{}/beam-{}/{}' \
                .format(args.decode_path,args.model_type,args.lang, args.modalities, args.noise_snr, int(visible),
                        args.beam_size, args.noise_fn.split('/')[-2])
os.makedirs(out_path, exist_ok=True)

if args.fp16:
    whisper_model.align.half()#
# Convert new paramters to fp16
if args.fp16 and args.use_av_hubert_encoder == 1:
    whisper_model.encoder.video_projection_scalar.half()
    whisper_model.encoder.video_model.half()
    whisper_model.align.half()#
    model_to_num_layers = {'small': 12, 'medium': 24, 'medium.en': 24, 'large-v2': 32, 'large-v3': 32}
    if args.av_fusion == 'separate':
        for i in range(model_to_num_layers[args.model_type]):
            try:
                whisper_model.decoder.blocks[i].attn_gate.data = whisper_model.decoder.blocks[i].attn_gate.half()
                whisper_model.decoder.blocks[i].ff_gate.data = whisper_model.decoder.blocks[i].ff_gate.half()
            except:
                continue

hypo, refs = [], []
whisper_model.eval() # AV-HuBERT batch norm and dropout


path_out=os.path.join(out_path, 'pred.txt') 
with open(path_out, 'w+') as f:
    for i, b in enumerate(tqdm(dataloader)):
        if args.fp16:
            input_ids = b["input_ids"].half().cuda()
            video = b["video"].half().cuda()
        else:
            if torch.cuda.is_available():
              input_ids = b["input_ids"].cuda()
              video = b["video"].cuda()
            else:
              input_ids = b["input_ids"]
              video = b["video"]
        labels = b["labels"]
        with torch.no_grad():
            # NOTE: haven't implemented padding mask for AV-HuBERT, but it seems to work fine without it
            if args.modalities == "avsr":
                results = whisper_model.decode(input_ids, args.align, options, video)
            elif args.modalities == "asr": 
                results = whisper_model.decode(input_ids, args.align,options, video, test_a=True)
            elif args.modalities == "vsr": 
                results = whisper_model.decode(input_ids,args.align, options, video, test_v=True)
            else:
                raise NotImplementedError
            
            for r, l in zip(results, labels):
                hypo.append(r.text)
                print('HYPO: {}'.format(r.text))
                f.write('HYPO: {}\n'.format(r.text))

                l[l == -100] = tokenizer.eot
                ref = tokenizer.decode([t for t in l if t.item() not in special_token_set])
                refs.append(ref)
                print('REF: {}'.format(ref))
                f.write('REF: {}\n'.format(ref))

if args.lang == 'en' or args.task == 'transcribe':
    if args.normalizer == 'whisper':
        if args.lang == 'en':
            std = EnglishTextNormalizer()
        else:
            std = BasicTextNormalizer()
        c_err, c_len, w_err, w_len = 0, 0, 0, 0
    else:
        scorer = WerScorer(
            WerScorerConfig(
                wer_tokenizer="13a",
                wer_remove_punct=True,
                wer_char_level=False,
                wer_lowercase=True
            )
        )
    with open(os.path.join(out_path, 'wer.368862'), 'w+') as f:
        for h, r in zip(hypo, refs):
            if args.normalizer == 'whisper':
                w_err += editdistance.eval(std(r).split(), std(h).split())
                w_len += len(r.split())
            else: 
                scorer.add_string(ref=r, pred=h)
                wer = scorer.score()
        if args.normalizer == 'whisper':
            wer = 100. * w_err/w_len
        print("WER: %.4f" % wer)
        f.write("WER: %.4f\n" % wer)
    with open(os.path.join(out_path, 'wer.json'), 'w+',) as fp:
        json.dump({'pred': hypo, 'refs': refs}, fp)
else:
    with open(os.path.join(out_path, 'bleu.368862'), 'w+') as f:
        bleu = sacrebleu.corpus_bleu(hypo, [refs]) #NOTE: [ref] not ref
        print("BLEU: %.4f" % bleu.score)
        f.write("BLEU: %.4f\n" % bleu.score)
    with open(os.path.join(out_path, 'bleu.json'), 'w+',) as fp:
        json.dump({'pred': hypo, 'refs': refs}, fp)

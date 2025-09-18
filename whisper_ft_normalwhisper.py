import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
import pandas as pd
import whisper
import evaluate
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    load_data,
    load_wave,
    add_noise,
    WhisperDataCollatorWhithPaddingNormalWhisper,
    whisper_optimizer,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import LengthBatchSampler
import librosa
import torch.nn.functional as F
SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class MuavicSpeechDatasetNormalWhisper(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, model_name, max_length, 
                 spec_augment, noise_prob=0, noise_fn=None) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        print("Dataloader max length : {}".format(max_length))
        print("Loaded {} noise wavs".format(len(self.noise_fn)))
        # path for normal speech corresponding to the whisper speech
        self.whisper2normal=dict()
        with open('data/w2n.txt','r') as f:
            for line in f:
                id1,id2=line.split(' ')
                self.whisper2normal[id1]=id2.strip()
        # text for normal speech
        self.normal2text=dict()
        with open('data/n2text.txt','r') as f:
            for line in f:
                id1,text=line.split(' ')
                self.normal2text[id1]=text.strip()
        
    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        lang, audio_path, text, _ = self.audio_info_list[id]
        
        # find path for normal speech corresponding to the whisper speech
        basename=audio_path.split('/')[-1][:-4]
        basename_normal=self.whisper2normal[basename]
        normal_audio_path=audio_path.replace(basename,basename_normal)
        text_normal=self.normal2text[basename_normal]

        # whisper speech
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    

        if np.random.rand() > self.noise_prob: 
            audio = audio.astype(np.float32)
        else: 
            audio = add_noise(audio, self.noise_fn, noise_snr=0).astype(np.float32)
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
        # normal speech
        audio_normal, sr = librosa.load(normal_audio_path, sr=16000, mono=True) 

        if np.random.rand() > self.noise_prob: 
            audio_normal = audio_normal.astype(np.float32)
        else: 
            audio_normal = add_noise(audio_normal, self.noise_fn, noise_snr=0).astype(np.float32)
        audio_frames_normal = len(audio_normal.flatten()) // 160
        # pad audio to cfg.audio_max_length (longer samples filtered out already)
        if self.max_length != None:
            audio_normal = whisper.pad_or_trim(audio_normal.flatten(), length=self.max_length)
        n_mels_normal = 80 if self.model_name != 'large-v3' else 128
        mel_normal = whisper.log_mel_spectrogram(audio_normal, n_mels=n_mels_normal) # freq by time
    
        if self.spec_augment:
            if self.spec_augment == "ls-double":
                mel_normal = torch.from_numpy(spec_augment(mel_normal.T.numpy(), audio_frames_normal)).T # expects time by freq
            elif self.spec_augment == "ls-basic":
                mel_normal = torch.from_numpy(spec_augment(mel_normal.T.numpy(), audio_frames_normal, n_freq_mask=1, n_time_mask=1)).T # expects time by freq
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

        # dec_input_ids and labels for normal speech
        dec_input_ids_normal = [self.tokenizer.sot, 
                    self.tokenizer.special_tokens["<|{}|>".format(lang)], 
                    self.tokenizer.transcribe, 
                    self.tokenizer.no_timestamps] + \
                    self.tokenizer.encode(" " + text_normal)
        labels_normal = dec_input_ids_normal[1:] + [self.tokenizer.eot]
        
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "input_ids_normal": mel_normal,
            "labels_normal":labels_normal,
            "dec_input_ids_normal":dec_input_ids_normal, 
        }

class WhisperModelModule(LightningModule):
    def __init__(self, cfg, model_name, lang, train_dataset, val_dataset, test_dataset) -> None:
        super().__init__()
        self.model_name = model_name
        print("Loading Whisper model and weights")
        self.model = whisper.load_model(model_name, 
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='openai/',
                                        dropout_rate=cfg.dropout_rate,)
        multilingual = True if 'large' in model_name or 'en' not in model_name else False
        print("Multilingual tokenizer : {}".format(multilingual))
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=multilingual, task='transcribe')

        # if 'large' in self.model_name: # only decoder training
        #     for p in self.model.encoder.parameters():
        #         p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.train_dataset = train_dataset
        self.__val_dataset = val_dataset
        self.__test_dataset = test_dataset
        self.special_token_set = set(self.tokenizer.special_tokens.values())

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        input_ids_normal = batch["input_ids_normal"] #corresponding normal speech
        labels_normal = batch["labels_normal"].long()
        dec_input_ids_normal = batch["dec_input_ids_normal"].long()
        
        combined_inputs = torch.cat([input_ids, input_ids_normal], dim=0)
        combined_features, x_v = self.model.encoder(combined_inputs)
        audio_features, audio_features_normal = torch.chunk(combined_features, 2, dim=0)


        audio_features_align = audio_features+self.model.align(audio_features)

        audio_features_normal_align=audio_features_normal

        combined_audio_features = torch.cat([audio_features_align, audio_features_normal_align], dim=0)  # (2B, T, C)
        repeated_dec_input_ids = torch.cat([dec_input_ids, dec_input_ids_normal], dim=0) 

        combined_out = self.model.decoder(repeated_dec_input_ids, combined_audio_features)  # (2B, seq_len, vocab_size)
        out, out_normal = torch.chunk(combined_out, 2, dim=0)

        loss_whisper = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss_normal = self.loss_fn(out_normal.view(-1, out_normal.size(-1)), labels_normal.view(-1))
        loss_all=loss_whisper+loss_normal
        self.log("train/loss", loss_all,on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss_whisper", loss_whisper,on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss_normal", loss_normal,on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss_all
    
    
    def validation_step(self, batch, batch_id, dataloader_idx=None):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        input_ids_normal = batch["input_ids_normal"]

        audio_features, x_v = self.model.encoder(input_ids)

        audio_features = audio_features + self.model.align(audio_features)
        out = self.model.decoder(dec_input_ids, audio_features)

        audio_features_normal, x_v = self.model.encoder(input_ids_normal)

        out_normal=self.model.decoder(dec_input_ids, audio_features_normal)
        
        loss_whisper = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss_normal = self.loss_fn(out_normal.view(-1, out_normal.size(-1)), labels.view(-1))
        
        # loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        labels[labels == -100] = self.tokenizer.eot
        # remove all decoder predictions after first eot for proper decoding
        tokens = torch.argmax(out, dim=2)
        tokens_normal = torch.argmax(out_normal, dim=2)

        # Set all decoder predictions after first eot to eot
        # TODO: fix for large-v3, which predicts <eot> in the beginning
        eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))
        first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).cuda() * eot_find, dim=1, keepdim=True)
        tokens[torch.arange(eot_find.shape[1]).cuda() > first_eot] = self.tokenizer.eot
        
        eot_find_normal = (torch.where(tokens_normal == self.tokenizer.eot, 1, 0))
        first_eot_normal = torch.argmax(torch.arange(eot_find_normal.shape[1], 0, -1).cuda() * eot_find_normal, dim=1, keepdim=True)
        tokens_normal[torch.arange(eot_find_normal.shape[1]).cuda() > first_eot_normal] = self.tokenizer.eot

        # calculate next token prediction, not include lang tag, task, and no timestamps token
        mask = ~(tokens[:, 3:] == self.tokenizer.eot) # torch.ne fails for some reason
        n_correct = torch.sum(
            tokens[:, 3:].masked_select(mask).eq(labels[:, 3:].masked_select(mask))
        )
        total = torch.sum(mask)
        acc = n_correct.item() / (total.item() + 1e-8)
        acc = acc if acc < 1 else 0
        
        mask_normal = ~(tokens_normal[:, 3:] == self.tokenizer.eot) # torch.ne fails for some reason
        n_correct_normal = torch.sum(
            tokens_normal[:, 3:].masked_select(mask_normal).eq(labels[:, 3:].masked_select(mask_normal))
        )
        total_normal = torch.sum(mask_normal)
        acc_normal = n_correct_normal.item() / (total_normal.item() + 1e-8)
        acc_normal = acc_normal if acc_normal < 1 else 0
        

        o_list, o_list_full, l_list, l_list_full = [], [], [], []
        for o, l in zip(tokens, labels):
            o_list.append(self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set]))
            # o_list_full.append(self.tokenizer.decode(o))
            l_list.append(self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set]))
            # l_list_full.append(self.tokenizer.decode(l))
        wer, cer = wer_cer(hypo=o_list, ref=l_list)
        
        o_listn, o_list_fulln, l_listn, l_list_fulln = [], [], [], []
        for o, l in zip(tokens_normal, labels):
            o_listn.append(self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set]))
            # o_list_full.append(self.tokenizer.decode(o))
            l_listn.append(self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set]))
            # l_list_full.append(self.tokenizer.decode(l))
        wern, cern = wer_cer(hypo=o_listn, ref=l_listn)
        
        # for i, (hypo, hypo_full, ref, ref_full) in enumerate(zip(o_list, o_list_full, l_list, l_list_full)):
        for i, (hypo, ref, hypon, refn) in enumerate(zip(o_list, l_list,o_listn, l_listn)):
            print("-"*10)
            print("PRED: {}".format(hypo))
            # print(hypo_full)
            print("REF:  {}".format(ref))
            print("PREDnormal: {}".format(hypon))
            # print(hypo_full)
            print("REFnormal:  {}".format(refn))
            # print(ref_full)
            if i == 1: break
        
        # log_prefix = 'val' if dataloader_idx == 1 else 'val_noisy_multi_babble'
        # log_prefix = {0: 'test_noisy_multi_babble', 1: 'test', 2: 'val_noisy_multi_babble', 3: 'val'}
        log_prefix = {0: 'val_noisy_multi_babble', 1: 'val', 2: 'test_noisy_multi_babble', 3: 'test'}

        self.log("{}/loss_whisper".format(log_prefix[dataloader_idx]), loss_whisper, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/cer".format(log_prefix[dataloader_idx]), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer".format(log_prefix[dataloader_idx]), wer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc".format(log_prefix[dataloader_idx]), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/loss_normal".format(log_prefix[dataloader_idx]), loss_normal, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/cer_normal".format(log_prefix[dataloader_idx]), cern, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/wer_normal".format(log_prefix[dataloader_idx]), wern, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc_normal".format(log_prefix[dataloader_idx]), acc_normal, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        return {
            "cer": cer,
            "wer": wer,
            "loss": loss_whisper
        }

    def configure_optimizers(self):
        model = self.model
        optimizer, scheduler = whisper_optimizer(model, self.cfg, self.t_total, video=False)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = MuavicSpeechDatasetNormalWhisper(self.__train_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=self.cfg.spec_augment,
                                      noise_prob=cfg.noise_prob,
                                      noise_fn=cfg.noise_fn,
                                    )   
        
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * self.cfg.batch_size,
                    shapes=[i[3] for i in self.__train_dataset],
                    sort_in_batch='descending',
                    sort_batch='shuffle',
                    drop_last=True,)
        if cfg.num_devices > 1:
            print("Using distributed sampler")
            length_sorter = DistributedSamplerWrapper(length_sorter)
        return torch.utils.data.DataLoader(dataset,
                        batch_sampler=length_sorter,
                        num_workers=self.cfg.num_worker,
                        collate_fn=WhisperDataCollatorWhithPaddingNormalWhisper())

    def val_dataloader_clean(self):
        dataset = MuavicSpeechDatasetNormalWhisper(self.__val_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=0,
                                    )
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[3] for i in self.__val_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPaddingNormalWhisper()
                          )
    
    def val_dataloader_noisy(self):
        dataset = MuavicSpeechDatasetNormalWhisper(self.__val_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=1,
                                      noise_fn=cfg.noise_fn_val,
                                    )
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[3] for i in self.__val_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPaddingNormalWhisper()
                          )
    def test_dataloader_clean(self):
        dataset = MuavicSpeechDatasetNormalWhisper(self.__test_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=0,
                                    )
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[3] for i in self.__test_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPaddingNormalWhisper()
                          )
    
    def test_dataloader_noisy(self):
        dataset = MuavicSpeechDatasetNormalWhisper(self.__test_dataset, 
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=None,
                                      spec_augment=False,
                                      noise_prob=1,
                                      noise_fn=cfg.noise_fn_test,
                                    )
        length_sorter = LengthBatchSampler(batch_bins=self.cfg.audio_max_length * 8,
                    shapes=[i[3] for i in self.__test_dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=length_sorter,
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPaddingNormalWhisper()
                          )

cfg_yaml = sys.argv[1]
with open(cfg_yaml, 'r') as file:
    dct = yaml.safe_load(file)
    cfg = types.SimpleNamespace(**dct)

print(cfg)
print("audio max length: {}".format(cfg.audio_max_length))

tflogger, checkpoint_callback, callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                                            cfg.check_output_dir, 
                                                                            cfg.train_name, 
                                                                            cfg.train_id,
                                                                            cfg.monitor,)

if 'zh' in cfg.lang:
    audio_transcript_pair_list = load_data(cfg.audio_max_length, cfg.text_max_length, ['zh'], 
                                        include_audio_lens=True)
elif 'en' in cfg.lang:
    audio_transcript_pair_list = load_data(cfg.audio_max_length, cfg.text_max_length, ['en'], 
                                        include_audio_lens=True)

model = WhisperModelModule(cfg, cfg.model_name, cfg.lang, audio_transcript_pair_list['train'], 
                                                          audio_transcript_pair_list['valid'],
                                                          audio_transcript_pair_list['test'],
)

trainer = Trainer(
    precision=16,
    accelerator="gpu",
    max_steps=cfg.num_train_steps,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list,
    num_sanity_val_steps=0, # default is 2 batches, 0 to turn off
    devices=cfg.num_devices,
    val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps), # validate after this number batches
    check_val_every_n_epoch=None, # If None, validation will be done solely based on the number of training batches
    reload_dataloaders_every_n_epochs=1, # shuffle the dataloader after an epoch
    use_distributed_sampler=False, # implemented custom distributed trainer
)

print(cfg)
resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
if os.path.exists(resume_ckpt) and cfg.resume_training: # resume training, don't validate
    trainer.fit(model, ckpt_path=resume_ckpt)
else:
    trainer.validate(model=model, dataloaders=[model.val_dataloader_noisy(), model.val_dataloader_clean(),
                                                model.test_dataloader_noisy(), model.test_dataloader_clean()]) # validate before training
    trainer.fit(model, val_dataloaders=[model.val_dataloader_noisy(), model.val_dataloader_clean(),
                                                 model.test_dataloader_noisy(), model.test_dataloader_clean()])




import os
import cv2
import argparse
from tqdm import tqdm
from scipy.io import wavfile

def count_frames(fids, audio_dir, video_dir):
    total_num_frames = []
    data = []
    for fid, audio_subdir, video_subdir in tqdm(fids):
        wav_fn = os.path.join(audio_dir, audio_subdir, fid + ".wav")
        video_fn = os.path.join(video_dir, video_subdir, fid + ".avi")
        # 确保文件存在，跳过不存在的条目
        if not os.path.isfile(wav_fn):
            continue
        num_frames_audio = len(wavfile.read(wav_fn)[1]) / 3 # 48kHz to 16kHz
        cap = cv2.VideoCapture(video_fn)
        num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_num_frames.append([num_frames_audio, num_frames_video])
        data.append([fid, video_fn, wav_fn, num_frames_video, num_frames_audio])
    return total_num_frames, data

def process_class(cls, speech_type, args, id2text):
    fids = []
    audio_dir = os.path.join(args.base_dir, cls, 'audio', 'hfm', speech_type)
    video_dir = os.path.join(args.base_dir, cls, 'video', 'camera', speech_type)
    
    if not os.path.exists(audio_dir) or not os.path.exists(video_dir):
        return
    
    for subdir in os.listdir(audio_dir):
        subdir_path = os.path.join(audio_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.wav'):
                    fid = filename[:-4]  # 去掉 .wav 后缀
                    fids.append((fid, subdir, subdir))

    num_frames, data = count_frames(fids, audio_dir, video_dir)
    audio_num_frames = [x[0] for x in num_frames]
    video_num_frames = [x[1] for x in num_frames]
    if speech_type == 'normal':
        suffix = f"{cls}_{speech_type}"
    else:
        suffix = f"{cls}"
    cls_output_dir = os.path.join(args.output_dir, suffix)
    os.makedirs(cls_output_dir, exist_ok=True)

    with open(os.path.join(cls_output_dir, f'nframes.audio'), 'w') as fo:
        fo.writelines(f"{x}\n" for x in audio_num_frames)

    with open(os.path.join(cls_output_dir, f'nframes.video'), 'w') as fo:
        fo.writelines(f"{x}\n" for x in video_num_frames)

    with open(os.path.join(cls_output_dir, f"{suffix}.tsv"), 'w') as fo:
        fo.write('/\n')
        for fid, video_path, audio_path, nf_video, nf_audio in data:
            fo.write('\t'.join([fid, video_path, audio_path, str(nf_video), str(nf_audio)]) + '\n')

    with open(os.path.join(cls_output_dir, f"{suffix}.wrd"), 'w') as fo:
        for fid, _, _ in fids:
            fo.write(id2text.get(fid, '') + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process AISHELL6-Whisper data', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_dir', type=str, required=True, 
                        help='Base directory containing the AISHELL6-Whisper data.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory to save results.')
    args = parser.parse_args()

    id2text = {}
    txt_file = os.path.join(args.base_dir, "text_sentence")
    with open(txt_file, 'r') as file:
        for line in file:
            id1, text = line.split(' ', 1)
            id2text[id1.strip()] = text.strip()

    for cls in ['train', 'valid', 'test']:
        for speech_type in ['normal', 'whisper']:
            process_class(cls, speech_type, args, id2text)
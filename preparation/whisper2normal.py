import os
import argparse
from difflib import SequenceMatcher

def load_texts(base_dir):
    id2text = {}
    txt_file = f"{base_dir}/text_sentence"
    with open(txt_file, 'r') as file:
        for line in file:
            id1, text = line.split(' ')
            id2text[id1] = text
    return id2text

def get_audio_paths(base_dir, cls):
    fids_normal = {}
    fids_whisper = {}
    audio_dir = os.path.join(base_dir, cls, 'audio', 'hfm')
    
    for sub2 in os.listdir(audio_dir):
        audio_dir_sub2 = os.path.join(audio_dir, sub2)
        for sub3 in os.listdir(audio_dir_sub2):
            audio_dir_sub3 = os.path.join(audio_dir_sub2, sub3)
            for sub4 in os.listdir(audio_dir_sub3):
                spkid = sub4.split('-')[1]
                path = os.path.join(cls, 'audio', 'hfm', sub2, sub3, sub4)
                
                if '-1_' in sub4:
                    fids_normal.setdefault(spkid, []).append(path)
                elif '-2_' in sub4:
                    fids_whisper.setdefault(spkid, []).append(path)
    
    return fids_normal, fids_whisper

def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def match_whisper_to_normal(fids_whisper, fids_normal, id2text):
    whisper2normal = []
    n2text = []
    
    for key in fids_whisper.keys():
        for relative_path in fids_whisper[key]:
            fid = relative_path.split('/')[-1][:-4]
            text_whisper = id2text.get(fid, "")
            
            max_similarity = 0
            best_match_path = ""
            best_match_text = ""
            
            for relative_path2 in fids_normal.get(key, []):
                fid2 = relative_path2.split('/')[-1][:-4]
                text_normal = id2text.get(fid2, "")
                
                similarity = calculate_similarity(text_normal, text_whisper)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_path = relative_path2
                    best_match_text = text_normal
            
            whisper2normal.append([relative_path, best_match_path])
            n2text.append([best_match_path, best_match_text.strip()])
    
    return whisper2normal, n2text

def write_results(whisper2normal, n2text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'w2n.txt'), 'w') as f:
        for fid, match_fid in whisper2normal:
            f.write(fid + ' ' + match_fid + '\n')
    
    with open(os.path.join(output_dir, 'n2text.txt'), 'w') as f:
        for match_fid, match_text in n2text:
            f.write(match_fid + ' ' + match_text + '\n')

def main(base_dir, output_dir):
    id2text = load_texts(base_dir)
    
    whisper2normal_results = []
    n2text_results = []
    
    for cls in ['train', 'valid', 'test']:
        fids_normal, fids_whisper = get_audio_paths(base_dir, cls)
        
        whisper2normal, n2text = match_whisper_to_normal(fids_whisper, fids_normal, id2text)
        whisper2normal_results.extend(whisper2normal)
        n2text_results.extend(n2text)
    
    write_results(whisper2normal_results, n2text_results, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio data to match whispers to normal speech.")
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the AISHELL6-Whisper data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save results.')
    args = parser.parse_args()

    main(args.base_dir, args.output_dir)
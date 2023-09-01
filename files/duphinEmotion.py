import os
import json
import librosa
import subprocess
import numpy as np
import pandas as pd


class duphinEmotion:
    def __init__(self, pos_model, neg_model, df, audio_path):
        self.pos_model = pos_model
        self.neg_model = neg_model
        self.df = df
        self.audio_path = audio_path
    
    
    def audio_transform(self, start_time, end_time):
        ffmpeg_command = [
            "ffmpeg",
            "-i", self.audio_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c", "copy",
            "temp_audio.wav"
        ]
        
        try:
            subprocess.run(ffmpeg_command, check=True)
            return "temp_audio.wav"
        except subprocess.CalledProcessError as e:
            return None
        
        
    def audio_treatment(self, audio):
        X, sample_rate = librosa.load(audio) # Load o pedacinho do áudio
        
        X_vocal, _ = librosa.effects.hpss(X) # Separar só a voz
        
        max_peak = np.max(np.abs(X_vocal)) # Hard code: normalizar
        ratio = 1 / max_peak
        X = X * ratio
        
        result = np.array([])
        
        # Mel-frequency cepstrum
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        
        # Pre-computed power spectrogram: Mel Spectogram
        pcps = np.mean(librosa.feature.melspectrogram(S=np.abs(librosa.stft(X))**2, sr=sample_rate).T, axis=0)
        result = np.hstack((result, pcps))
        
        # Spectral Contrast: Relative distribution of energies
        contrast = np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(X)), sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
        
        # Constant-Q spectrogram
        C = np.abs(librosa.cqt(y=X, sr=sample_rate))
        onset_env = np.mean(librosa.onset.onset_strength(sr=sample_rate, S=librosa.amplitude_to_db(C, ref=np.max)).T, axis=0)
        result = np.hstack((result, onset_env))
        
        os.remove(audio) # Exclui da memória do sistema a versão *.wav
        
        return result
    
    
    def modelagem(self):
        result_model = {'id':[], 'feel':[]}
        
        for i in self.df.index:
            if self.df['sinal'][i] == 'Blue':
                try:
                    audio_path = self.audio_transform(self.df['start'][i], self.df['end'][i])
                except None:
                    break
                else:
                    output = self.audio_treatment(audio_path)
                    feeling = self.pos_model.predict([output])
                    if feeling == 0:
                        result_model['feel'].append('calma')
                    elif feeling == 1:
                        result_model['feel'].append('alegria')
                    else:
                        result_model['feel'].append('supresa')
                    result_model['id'].append(i)
            elif self.df['sinal'][i] == 'Red':
                try:
                    audio_path = self.audio_transform(self.df['start'][i], self.df['end'][i])
                except None:
                    break
                else:
                    output = self.audio_treatment(audio_path)
                    feeling = self.neg_model.predict([output])
                    if feeling == 0:
                        result_model['feel'].append('desinteresse')
                    elif feeling == 1:
                        result_model['feel'].append('raiva')
                    else:
                        result_model['feel'].append('rejeição')
                    result_model['id'].append(i)
                        
        return pd.DataFrame(data=result_model, index=result_model['id'])
    
    
    def salvacaoJSON(self, output_file):
        doc = pd.concat([self.df, self.modelagem()], axis=1)
        doc_json = json.dumps(doc.to_dict(orient='records'), ensure_ascii=False, indent=2)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(doc_json)
        print(doc_json) # Remover e salvar o arquivo em algum lugar

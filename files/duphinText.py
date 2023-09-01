import pandas as pd
import whisperx
from LeIA import SentimentIntensityAnalyzer

class DuphinText:
    def __init__(self, device, audio_file, batch_size, compute_type, hf_token):
        self.device = device
        self.audio_file = audio_file
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.hf_token = hf_token

        # Carregue os modelos uma vez durante a inicialização
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)
        self.model_a, self.metadata = None, None

    def transcricao(self):
        audio = whisperx.load_audio(self.audio_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        
        return audio, result


    def alinhamento(self, audio, result):
        if self.model_a is None:
            self.model_a, self.metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result_a = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)

        return result_a


    def vozes(self, result_segments):
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(self.audio_file, min_speakers=1, max_speakers=5)
        result_segments = whisperx.assign_word_speakers(diarize_segments, result_segments)

        bd = {
            'id': range(len(result_segments['segments'])),
            'start': [segment['start'] for segment in result_segments['segments']],
            'end': [segment['end'] for segment in result_segments['segments']],
            'voz': [segment['speaker'] for segment in result_segments['segments']],
            'fala': [segment['text'] for segment in result_segments['segments']]
        }

        return pd.DataFrame(data=bd)


    def sinalizador(self, sentence):
        SIA = SentimentIntensityAnalyzer()
        sentiment_dict = SIA.polarity_scores(sentence)
        
        return "Blue" if sentiment_dict['compound'] >= 0.00 else "Red"


    def apendice(self, dataframe):
        sinal = [self.sinalizador(frase) for frase in dataframe['fala']]
        return sinal

import unittest
import pandas as pd
import numpy as np
import joblib
import os

# Importe a classe duphinEmotion da sua implementação real
from ..files.duphinEmotion import duphinEmotion


class TestDuphinEmotion(unittest.TestCase):
    def setUp(self):
        # Crie modelos fictícios de emoção para testar
        self.pos_model = joblib.load("modelo_emocao_positiva.pkl")
        self.neg_model = joblib.load("modelo_emocao_negativa.pkl")

        # Crie um DataFrame de exemplo
        data = {'start': [0, 10, 20],
                'end': [5, 15, 25],
                'sinal': ['Blue', 'Red', 'Blue']}
        self.df = pd.DataFrame(data)

        # Especifique um arquivo de áudio de exemplo
        self.audio_path = "exemplo_audio.wav"


    def test_audio_transform(self):
        emotion_analyzer = duphinEmotion(self.pos_model, self.neg_model, self.df, self.audio_path)
        audio_path = emotion_analyzer.audio_transform(0, 5)
        self.assertIsNotNone(audio_path)
        # Certifique-se de que o arquivo de áudio temporário foi criado


    def test_audio_treatment(self):
        emotion_analyzer = duphinEmotion(self.pos_model, self.neg_model, self.df, self.audio_path)
        audio_path = emotion_analyzer.audio_transform(0, 5)
        features = emotion_analyzer.audio_treatment(audio_path)
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 560)  # Verifique o tamanho das características


    def test_modelagem(self):
        emotion_analyzer = duphinEmotion(self.pos_model, self.neg_model, self.df, self.audio_path)
        result_model = emotion_analyzer.modelagem()
        self.assertIsInstance(result_model, pd.DataFrame)
        self.assertTrue('id' in result_model.columns)
        self.assertTrue('feel' in result_model.columns)
        self.assertEqual(len(result_model), 2)  # Verifique o número de linhas no DataFrame resultante


    def test_salvacaoJSON(self):
        emotion_analyzer = duphinEmotion(self.pos_model, self.neg_model, self.df, self.audio_path)
        result_model = emotion_analyzer.modelagem()
        emotion_analyzer.salvacaoJSON("resultado_emocao_teste.json")
        # Verifique se o arquivo JSON foi criado
        self.assertTrue(os.path.exists("resultado_emocao_teste.json"))

    def tearDown(self):
        # Limpe os recursos após os testes, se necessário
        pass


if __name__ == "__main__":
    unittest.main()

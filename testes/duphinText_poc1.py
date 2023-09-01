import unittest
import pandas as pd
from ..files.duphinText import DuphinText

class TestDuphinText(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.audio_file = "/content/teste.mp3"
        self.batch_size = 16
        self.compute_type = "int8"
        self.HF_TOKEN = "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"

    def test_transcricao(self):
        duphin = DuphinText(self.device, self.audio_file, self.batch_size, self.compute_type, self.HF_TOKEN)
        audio, resultado = duphin.transcricao()
        self.assertIsNotNone(audio)
        self.assertIsNotNone(resultado)

    def test_alinhamento(self):
        duphin = DuphinText(self.device, self.audio_file, self.batch_size, self.compute_type, self.HF_TOKEN)
        audio, resultado = duphin.transcricao()
        linhas = duphin.alinhamento(audio, resultado)
        self.assertIsNotNone(linhas)

    def test_vozes(self):
        duphin = DuphinText(self.device, self.audio_file, self.batch_size, self.compute_type, self.HF_TOKEN)
        audio, resultado = duphin.transcricao()
        linhas = duphin.alinhamento(audio, resultado)
        df_diario = duphin.vozes(linhas)
        self.assertIsInstance(df_diario, pd.DataFrame)
        self.assertTrue('id' in df_diario.columns)
        self.assertTrue('start' in df_diario.columns)
        self.assertTrue('end' in df_diario.columns)
        self.assertTrue('voz' in df_diario.columns)
        self.assertTrue('fala' in df_diario.columns)

    def test_sinalizador(self):
        duphin = DuphinText(self.device, self.audio_file, self.batch_size, self.compute_type, self.HF_TOKEN)
        self.assertEqual(duphin.sinalizador("This is a positive sentence."), "Blue")
        self.assertEqual(duphin.sinalizador("This is a negative sentence."), "Red")

    def test_apendice(self):
        duphin = DuphinText(self.device, self.audio_file, self.batch_size, self.compute_type, self.HF_TOKEN)
        audio, resultado = duphin.transcricao()
        linhas = duphin.alinhamento(audio, resultado)
        df_diario = duphin.vozes(linhas)
        transc = duphin.apendice(df_diario)
        self.assertIsInstance(transc, list)
        self.assertTrue(all(s in ["Blue", "Red"] for s in transc))

if __name__ == "__main__":
    unittest.main()

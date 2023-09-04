from .files.duphinText import DuphinText
from .files.duphinEmotion import duphinEmotion
import joblib

if __name__ == "__main__":
    device = "cpu" # mudar para cuda se tiver gpu / o min custa acurácia
    audio_file = input() # o caminho do áudio para ser analisado ex. C:\Users\Wande\Documents\GitHub\Duphin_Emotion\Duphin_1.0\teste.mp3
    batch_size = 16 # diminuir se estiver custando muito a gpu
    compute_type = "int8" # mudar para float16 se tiver gpu / o min custa acurácia
    HF_TOKEN = "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"
    neg_model = joblib.load('./birdbox/MLP_RED_88.sav')
    pos_model = joblib.load('./birdbox/MLP_BLUE_91.sav')
    
    # Primeira parte: pegar o áudio e transformar em matriz (texto)
    duphinTxt = DuphinText(device, audio_file, batch_size, compute_type, HF_TOKEN)
    audio, resultado = duphinTxt.transcricao()
    linhas = duphinTxt.alinhamento(audio, resultado)
    df_diario = duphinTxt.vozes(linhas)
    transc = duphinTxt.apendice(df_diario)
    
    # Segunda parte: analisar as palavras e tom de voz para detectar qual é a emoção mais próxima sendo expressada
    # Usando como base o modelo de Ekman, 1972 (emocoes basicas)
    duphinEmo = duphinEmotion(pos_model, neg_model, transc, audio_file)
    result_model = duphinEmo.modelagem()
    duphinEmo.salvacaoJSON("resultado_emocao.json") # Salvar no db
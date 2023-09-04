# Duphin_1.0
ML/AI tool que busca detectar sentimentos usando o áudio e texto da fala.

## Arquivos
- ML: modelos treinados com sentimentos positivos e negativos
- files:
    * duphinText: Transforma o áudio em uma matriz
    * duphinEmotion: Usa a fala e analisa o sentimento da voz e adiciona a emoção mais aproximada.
- testes: arquivos de testes
- whisper: modelo da transcrição usado (contudo, deve ser remodelado)
- duphin.py: arquivo principal que chama todas as classes

## Modelo duphin:
Os modelos de previsão de emoções usados no áudio foram treinados com áudios em ingles, no total de 576 (192 arquivos para cada emoção).

### Resultado (Emoção Positiva - acc 91%)
| Emoção       | Precision | Recall | f1-score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (calma)    | 0,93      | 1,00   | 0,96     | 55      |
| 1 (alegria)  | 0,88      | 0,85   | 0,87     | 54      |
| 2 (surpresa) | 0,90      | 0,88   | 0,89     | 64      |

![image](https://github.com/Ingrid-0906/Duphin_1.0/assets/92744210/f83ddc26-6e9e-4460-83e4-7cc7c2b26d4d)

### Resultado (Emoção Negativa - acc 88%)
| Emoção           | Precision | Recall | f1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0 (desinteresse) | 0,89      | 094    | 0,92     | 54      |
| 1 (raiva)        | 0,89      | 0,80   | 0,84     | 64      |
| 2 (rejeição)     | 0,85      | 0,91   | 0,88     | 55      |

![image](https://github.com/Ingrid-0906/Duphin_1.0/assets/92744210/303a3741-6575-4203-9e5c-b437e5706873)

## Pipeline:
![image](https://github.com/Ingrid-0906/Duphin_1.0/assets/92744210/8164be27-67e6-413c-b730-52ff6d55e86d)

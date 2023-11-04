
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Carregando o arquivo CSV em um DataFrame
df = pd.read_csv('C:/Users/gusta/Documents/faculdade/matemática/Filtrados/AlbunsTaylor.csv', delimiter=';')

# Coluna que contem as letras das musicas
letras = df['lyric']  

frase_referencia = input("Digite a frase de referencia: ")

# Inicialize o vetorizador TF-IDF
vectorizer = TfidfVectorizer()

# Aplique o vetorizador nas letras das musicas
tfidf_matrix = vectorizer.fit_transform(letras)

# Calcule a similaridade de cosseno entre a frase de referencia e as letras das musicas
similaridades = cosine_similarity(vectorizer.transform([frase_referencia]), tfidf_matrix)

# Obtenha os indices das musicas mais similares em ordem decrescente
indices_mais_similares = similaridades.argsort()[0][::-1]

# Crie um conjunto para rastrear as musicas ja incluidas
musicas_incluidas = set()

i = 0
# Imprima as 10 musicas mais similares e seus valores de similaridade 
while len(musicas_incluidas) < 10 and i < len(indices_mais_similares):
    indice = indices_mais_similares[i]
    musica = df.iloc[indice]['track_title']
    valor = similaridades[0][indice]
    
    # Verifique se a musica ja foi incluida e, se nao, imprima-a
    if musica not in musicas_incluidas:
        musicas_incluidas.add(musica)
        print(f"{len(musicas_incluidas)}. Musica: {musica}, Similaridade (Cosseno): {valor}")
    
    i += 1

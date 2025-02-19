from fastapi import FastAPI, HTTPException
import os
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from fuzzywuzzy import process
import dask.dataframe as dd

app = FastAPI()

# ✅ 1. Cargar datasets optimizados con Dask
movies_ddf = dd.read_csv(
    'data/movies_dataset.csv',
    blocksize="16MB",
    usecols=['title', 'vote_average', 'popularity', 'release_year', 'revenue', 'budget', 'release_date']
).compute()

# ✅ 2. Filtrar las 20,000 películas más populares
movies_df = movies_ddf.sort_values(by='popularity', ascending=False).head(20000)

# ✅ 3. Cargar y filtrar actores y directores más frecuentes
cast_df = dd.read_csv('data/cast.csv').compute()
crew_df = dd.read_csv('data/crew.csv').compute()

actor_counts = cast_df['name_actor'].value_counts()
actores_frecuentes = actor_counts[actor_counts >= 5].index
cast_df = cast_df[cast_df['name_actor'].isin(actores_frecuentes)]

director_counts = crew_df[crew_df['job_crew'] == 'Director']['name_job'].value_counts()
directores_frecuentes = director_counts[director_counts >= 3].index
crew_df = crew_df[(crew_df['name_job'].isin(directores_frecuentes)) & (crew_df['job_crew'] == 'Director')]

# ✅ 4. Seleccionar y normalizar datos
movies_df = movies_df[['title', 'vote_average', 'popularity', 'release_year', 'revenue', 'budget', 'release_date']]
cast_df = cast_df[['movie_id', 'name_actor']]
crew_df = crew_df[['movie_id', 'name_job', 'job_crew']]

def normalizar_nombre(nombre):
    return unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('ascii').lower()

movies_df['title_normalized'] = movies_df['title'].apply(normalizar_nombre)
cast_df['name_actor_normalized'] = cast_df['name_actor'].apply(normalizar_nombre)
crew_df['name_job_normalized'] = crew_df['name_job'].apply(normalizar_nombre)

# ✅ 5. Manejo de valores NaN
movies_df.fillna({
    'vote_average': movies_df['vote_average'].mean(),
    'popularity': movies_df['popularity'].mean(),
    'release_year': movies_df['release_year'].mode()[0],
}, inplace=True)

movies_df['return'] = np.where(
    movies_df['budget'] != 0,
    movies_df['revenue'] / movies_df['budget'],
    0
)

# ✅ 6. Calcular la matriz de similitud del coseno
features = movies_df[['vote_average', 'popularity', 'release_year', 'return']]
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
cosine_sim = cosine_similarity(csr_matrix(features_normalized), csr_matrix(features_normalized))

# ✅ 7. Función de recomendación
def recomendacion(titulo: str):
    titulo_normalizado = normalizar_nombre(titulo)
    try:
        idx = movies_df[movies_df['title_normalized'] == titulo_normalizado].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# ✅ 8. Endpoints de la API
@app.get('/')
def read_root():
    return {"message": "Bienvenido a la API de recomendación de películas"}

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    mes = mes.lower()
    if mes not in meses:
        raise HTTPException(status_code=400, detail="Mes no válido.")
    
    peliculas_mes = movies_df[movies_df['release_date'].notna()]
    peliculas_mes['release_month'] = pd.to_datetime(peliculas_mes['release_date']).dt.month
    cantidad = peliculas_mes[peliculas_mes['release_month'] == meses[mes]].shape[0]
    
    return {"mensaje": f"{cantidad} películas fueron estrenadas en {mes.capitalize()}"}

@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    dias = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3,
        'viernes': 4, 'sábado': 5, 'domingo': 6
    }
    
    dia = dia.lower()
    if dia not in dias:
        raise HTTPException(status_code=400, detail="Día no válido.")
    
    peliculas_dia = movies_df[movies_df['release_date'].notna()]
    peliculas_dia['release_day'] = pd.to_datetime(peliculas_dia['release_date']).dt.weekday
    cantidad = peliculas_dia[peliculas_dia['release_day'] == dias[dia]].shape[0]
    
    return {"mensaje": f"{cantidad} películas fueron estrenadas en {dia.capitalize()}"}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    return {
        "mensaje": f"La película {pelicula['title'].values[0]} fue estrenada en {pelicula['release_year'].values[0]} con un score de {pelicula['vote_average'].values[0]}"
    }

@app.get('/recomendacion/{titulo}')
def get_recomendacion(titulo: str):
    try:
        recomendaciones = recomendacion(titulo)
        return {"recomendaciones": recomendaciones}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {str(e)}")

# ✅ 9. Servidor FastAPI con Uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Iniciando en puerto: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

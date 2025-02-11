# **Sistema de Recomendación de Películas**

Este proyecto es un sistema de recomendación de películas basado en contenido, desarrollado utilizando **FastAPI** y **Pandas**. El sistema permite realizar consultas sobre películas, actores, directores y obtener recomendaciones basadas en similitud de contenido.

---

## **Tabla de Contenidos**
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Transformaciones de Datos](#transformaciones-de-datos)
3. [Endpoints Disponibles](#endpoints-disponibles)
4. [Bibliotecas Utilizadas](#bibliotecas-utilizadas)
5. [Instalación y Ejecución](#instalación-y-ejecución)
6. [Despliegue](#despliegue)
7. [Contribuciones](#contribuciones)
8. [Licencia](#licencia)

---

## **Descripción del Proyecto**

El sistema de recomendación utiliza datos de películas, actores y directores para proporcionar recomendaciones personalizadas. Los datos se procesan y transforman para garantizar su calidad y consistencia. La API permite realizar consultas sobre:

- Cantidad de películas estrenadas en un mes o día específico.
- Información detallada sobre una película (título, año de estreno, puntuación, etc.).
- Éxito de un actor o director (retorno total, promedio de retorno, etc.).
- Recomendaciones de películas similares.

---

## **Transformaciones de Datos**

Se realizaron las siguientes transformaciones en los datos:

1. **Manejo de Valores Nulos**:
   - Rellenar valores nulos en `revenue` y `budget` con `0`.
   - Eliminar filas con valores nulos en `release_date`.
   - Rellenar valores nulos en `vote_average` y `popularity` con la media.

2. **Formato de Fechas**:
   - Convertir `release_date` al formato `AAAA-mm-dd`.
   - Crear la columna `release_year` extrayendo el año de `release_date`.

3. **Cálculo de Retorno de Inversión**:
   - Crear la columna `return` como `revenue / budget`. Si `budget` es `0`, se asigna `0`.

4. **Eliminación de Columnas Innecesarias**:
   - Eliminar columnas como `video`, `imdb_id`, `adult`, `original_title`, `poster_path`, y `homepage`.

5. **Filtrado de Datos**:
   - En `crew.csv`, filtrar para incluir solo filas donde `job == 'Director'`.

6. **Conversión de Tipos de Datos**:
   - Convertir `movie_id` a `int64` en todos los datasets para garantizar consistencia.

---

## **Endpoints Disponibles**

La API cuenta con los siguientes endpoints:

1. **`/cantidad_filmaciones_mes/{mes}`**:
   - Devuelve la cantidad de películas estrenadas en un mes específico.
   - Ejemplo: `GET /cantidad_filmaciones_mes/enero`.

2. **`/cantidad_filmaciones_dia/{dia}`**:
   - Devuelve la cantidad de películas estrenadas en un día específico.
   - Ejemplo: `GET /cantidad_filmaciones_dia/lunes`.

3. **`/score_titulo/{titulo}`**:
   - Devuelve el título, año de estreno y puntuación de una película.
   - Ejemplo: `GET /score_titulo/Titanic`.

4. **`/votos_titulo/{titulo}`**:
   - Devuelve el título, año de estreno, cantidad de votos y promedio de votaciones de una película.
   - La película debe tener al menos 2000 valoraciones.
   - Ejemplo: `GET /votos_titulo/Titanic`.

5. **`/get_actor/{nombre_actor}`**:
   - Devuelve el éxito de un actor (retorno total, cantidad de películas, promedio de retorno).
   - Ejemplo: `GET /get_actor/Tom Hanks`.

6. **`/get_director/{nombre_director}`**:
   - Devuelve el éxito de un director y detalles de sus películas.
   - Ejemplo: `GET /get_director/Christopher Nolan`.

7. **`/recomendacion/{titulo}`**:
   - Devuelve una lista de 5 películas recomendadas basadas en similitud de contenido.
   - Ejemplo: `GET /recomendacion/Titanic`.

---

## **Bibliotecas Utilizadas**

El proyecto utiliza las siguientes bibliotecas de Python:

- **FastAPI**: Para crear la API.
- **Pandas**: Para la manipulación y transformación de datos.
- **Scikit-learn**: Para calcular la similitud del coseno.
- **Uvicorn**: Para ejecutar el servidor de la API.
- **NumPy**: Para operaciones numéricas.

Puedes instalar las dependencias usando el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## **Instalación y Ejecución**

1. **Clonar el Repositorio**:
   ```bash
   git clone https://github.com/tuusuario/sistema-recomendacion.git
   cd sistema-recomendacion
   ```

2. **Instalar Dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la API**:
   ```bash
   uvicorn main:app --reload
   ```

4. **Acceder a la API**:
   - La API estará disponible en `http://127.0.0.1:8000`.
   - La documentación interactiva (Swagger) estará disponible en `http://127.0.0.1:8000/docs`.

---

## **Despliegue**

La API puede desplegarse en servicios como **Render**, **Railway**, o **Heroku**. Asegúrate de configurar las variables de entorno y seguir las instrucciones del servicio elegido.

---

## **Contribuciones**

¡Las contribuciones son bienvenidas! Si deseas mejorar el proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una rama para tu contribución (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

---

## **Licencia**

Este proyecto está bajo la licencia **MIT**. Para más detalles, consulta el archivo [LICENSE](LICENSE).

---

¡Gracias por usar el Sistema de Recomendación de Películas! 🎬🍿
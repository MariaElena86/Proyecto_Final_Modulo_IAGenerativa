# Tutor Virtual de Machine Learning 
**Autor:** María Elena Carralero
URL del VIDEO: https://drive.google.com/file/d/1fSpsUm2wiLD0_8PoTwJm1z51ieZ6kHmr/view?usp=sharing

Agente experto en **Machine Learning** que responde preguntas sobre los contenidos dado en el modulo de ML que incluye:
- Fundamentos de ML, Feature Engineering, Aprendizaje Supervisado/No Supervisado, Series Temporales,
Optimización y Explicabilidad, Sistemas de Recomendación (7 PDFs, ~180 páginas en total)
- Se indexan en ChromaDB con embeddings locales y se consulta mediante un agente LangGraph que combina recuperación (RAG) y generación con Gemini.

## Stack tecnológico

- LLM: Google Gemini (`gemini-3.5-flash-lite`)
- Embeddings: HuggingFace `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (locales, se ha utilizado este modelo por exceder las cuotas del consumo de la API de Google).
- Base de conocimiento vectorial: ChromaDB (persistente en `data/chroma`)
- Framework de agente: LangGraph (sobre LangChain)
- Entorno de experimentación: Jupyter Notebook (`agent_virtual_ml.ipynb`)
- Interfaz web: Streamlit (`app.py`)

## Estructura del proyecto

```
Proyecto Final/
├── app.py                       # App Streamlit
├── agent_virtual_ml.ipynb         # Notebook principal con el agente
├── README.md
├── data/
│   ├── documents/               # PDFs del curso
│   └── chroma/                  # Base de datos vectorial (generada)
```

## Instalación y ejecución

1. Clonar/descargar el proyecto.

2. Crear un archivo `.env` en la raíz con la variable:
   ```GEMINI_API_KEY=tu_api_key_de_gemini```

3. Instalar dependencias:
  ```pip install python-dotenv langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-google-genai langgraph chromadb streamlit jupyter  ```

4. Revisar documentacion en `data/documents/`.
  ```01_Fundamentos_Machine_Learning, 02_Feature_Engineering, 03_Aprendizaje_no_supervisado,    04_Aprendizaje_Supervisado, 05_Series_Temporales,  06_Optimizacion_y_Explicabilidad, 08_Sistemas_Recomendacion```

5. Abrir el notebook:
   ```jupyter notebook agent_virtual_ml.ipynb```
   Ejecutar las celdas en orden. 
   La primera ejecución crea e indexa los documentos en ChromaDB (puede tardar unos minutos).
6. Revisar que se ha creado la bd en ```data/chroma```
7. Ejecutar la app Streamlit con el comando:```streamlit run app.py```



## System prompt (justificación)

El system prompt (`SYSTEM_PROMPT` en el notebook y en `app.py`) define al agente
como un **Tutor Virtual Pedagógico de Machine Learning** con las siguientes
decisiones de diseño:

- **Respuesta exclusivamente con la documentación del curso**: evita alucinaciones
  y mantiene el agente dentro del dominio (ML del curso). Si no hay información,
  lo indica.
- **Invocación obligatoria de `buscar_documentacion` antes de responder**: fuerza
  el uso de RAG en cada consulta, garantizando que la respuesta se fundamente en
  la base de conocimiento.
- **Citas con documento y página**: trazabilidad y verificabilidad de las
  respuestas (criterio de calidad del RAG).
- **Respuesta en español, clara, pedagógica y estructurada**: adecuada al
  público objetivo (alumnos del curso).


## Flujo RAG / agente (notebook y app)
1. Carga de PDFs (`PyPDFLoader`) y fragmentación en chunks
   (tamaño 800, solapamiento 100).
2. Generación de embeddings locales y almacenamiento en
   ChromaDB (se indexa solo si la colección está vacía).
3. Retriever, búsqueda semántica con `k=3`.
4. Tool `buscar_documentacion`, recupera fragmentos relevantes y los
   devuelve al agente con fuente y página.
5. System prompt, define al tutor pedagógico y obliga a usar la herramienta
   antes de responder.
6. Crear Agente LangGraph: nodos `llm_call` y `tool_node`, con memoria
   (`InMemorySaver`) con un `thread_id` fijo, manteniendo el contexto entre turnos.
7. Consulta: `consultar_tutor` normaliza la respuesta y mide el tiempo.


## Descripción de la Interfaz Streamlit (`app.py`)
- La carga de embeddings/retrieverse inicializa la  primera vez que se consulta, para no bloquear la apertura de la interfaz.
  
**Panel izquierdo:** 
  - Preguntas sugeridas en forma de botones, que puedes pulsar y se adiciona al chat
  - Métricas: Modelo, Consultas, # Documentos y Chunks.
    
**Panel derecho:**
  - Contiene título, chat con historial y tiempo de respuesta por mensaje y entrada de pregunta.


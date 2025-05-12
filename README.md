## Asistente de Consulta Técnica

Este proyecto implementa una aplicación web interactiva que actúa como asistente de consulta técnica. Utiliza Streamlit para la interfaz, Pinecone como almacén vectorial para búsqueda semántica y OpenAI vía LangChain para el procesamiento de lenguaje natural.

---

## 📁 Estructura de carpetas

```
├── app.py
├── requirements.txt
├── .env.example
├── data
│ └── documentos
│ ├── documento1.txt
│ ├── documento2.pdf
│ └── ...
└── src
| ├── loader.py
| ├── pinecone_utils.py
  └── chain.py

```
---

## 🔧 Módulos principales

### 1. loader.py

- **`load_and_split_documents(path, chunk_size, chunk_overlap)`**  
   1. Recorre los archivos en `path`.  
   2. Carga cada archivo con `TextLoader`.  
   3. Los fragmenta en trozos de `chunk_size` caracteres con `chunk_overlap`.  
   4. Devuelve lista de objetos `Document` listos para indexar.

### 2. pinecone_utils.py

- **`get_pinecone_client(api_key=None)`**  
  Inicializa `Pinecone(api_key=…)` usando la clave de entorno o pasada como parámetro.

- **`get_or_create_index(client, name, dimension, cloud, region)`**  
  Verifica si existe el índice `name`; si no, lo crea en modo serverless con las specs dadas. Devuelve la instancia `client.Index(name)`.

### 3. chain.py

- **`build_vectorstore(docs)`**  
   1. Crea un objeto `OpenAIEmbeddings`.  
   2. Llama a `PineconeVectorStore.from_texts(...)` para subir los chunks al índice.  
   3. Devuelve el vectorstore listo para recuperación.

- **`get_qa_chain(vectorstore)`**  
   1. Instancia `ChatOpenAI` con tu clave.  
   2. Usa `RetrievalQA.from_chain_type(...)` para enlazar el LLM con el vectorstore.  
   3. Configura `k=5` resultados y retorno de documentos fuente.

---
## ⚙️ Configuración y ejecución

## 1 Clonar el repositorio

`https://github.com/Ihan-Marroquin/proyecto_3.git`

## 2. Configuración del entorno virtual

2.1. Crea el entorno virtual en la carpeta `venv/`:
   - `python -m venv venv`

2.2. Activalo:
   - Windows:
       - `venv\Scripts\activate`
    
  - Linux/Mac:
       - `source venv/bin/activate`

Tras esto, tu prompt mostrará `(venv)` al inicio.

## 4. Instalación de dependencias
Con el entorno activado, instala todo lo necesario:

```
pip install --upgrade pip
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye:

```
streamlit
openai
langchain
pinecone
python-dotenv
langchain-community
langchain-pinecone
tiktoken
```

## 5. Configurar variables de entorno
Duplica `.env.example` como `.env` y rellena tus claves:

```
OPENAI_API_KEY=tu_openai_key
PINECONE_API_KEY=tu_pinecone_key
PINECONE_ENVIRONMENT=<extraído de tu host>
PINECONE_INDEX_NAME=asistente-tecnico
PINECONE_DIMENSION=1536   # o 3072 según modelo de embeddings
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## 6 Colocar documentos
Pon tus `.txt` en `data/documentos/`. Mínimo 70 archivos, la aplicacion ya cuenta con 70 archivos de muestra

## 7. Ejecutar la aplicación

`streamlit run app.py`

---

## 🥸 Aprendizaje

Aprendí a ver la IA como un proceso completo: desde construir una página simple donde el usuario hace preguntas, hasta organizar mis documentos para que el sistema “entienda” y busque la información más relevante, asi mismo, descubrí que dividir textos largos en partes más pequeñas ayuda al modelo a procesarlos sin perder contexto, y que un buen índice (como el de Pinecone) acelera la búsqueda de esas partes, también enfrenté pequeños tropiezos, como errores al leer archivos o cambiar funciones de librerías, pero eso me enseñó a buscar soluciones en la documentación y a estructurar mi código en módulos claros, además, este proyecto me mostró cómo combinar varias herramientas y prácticas de programación para crear un asistente de IA práctico y fiable.

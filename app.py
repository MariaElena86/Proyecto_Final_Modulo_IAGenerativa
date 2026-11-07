# ============================================================
# 1. IMPORTACIONES
# ============================================================

import os
import time
import uuid
import streamlit as st

from dotenv import load_dotenv
from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.tools import tool
from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    HumanMessage
)

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END
)

from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 2. CONFIGURACIÓN GENERAL
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Tutor Virtual de Machine Learning",
    page_icon="🎓",
    layout="wide"
)

MODEL = "gemini-3.5-flash-lite"

DOCUMENTS_PATH = Path("data/documents")
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "tutor_machine_learning"


# ============================================================
# 3. ESTILOS
# ============================================================

st.markdown("""
<style>
div.stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    border: 1px solid #d0d0d0 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
div.stButton > button:hover {
    border-color: #999 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}
div[data-testid="column"]:first-child {
    border-right: 1px solid #d0d0d0;
    padding-right: 1rem;
    margin-right: 0.5rem;
</style>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 4])

# ============================================================
# 4. ESTADO DE SESIÓN
# ============================================================

def inicializar_estado():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()

    if "app_tutor" not in st.session_state:
        st.session_state.app_tutor = None

    if "agent_api_key" not in st.session_state:
        st.session_state.agent_api_key = None

    if "total_calls" not in st.session_state:
        st.session_state.total_calls = 0


inicializar_estado()

# ============================================================
# 5. CARGA DE DOCUMENTOS PDF
# ============================================================

@st.cache_resource(show_spinner=False)
def cargar_documentos():
    documentos = []
    if not DOCUMENTS_PATH.exists():
        return documentos

    archivos_pdf = list(DOCUMENTS_PATH.glob("*.pdf"))
    for archivo in archivos_pdf:
        loader = PyPDFLoader(str(archivo))
        paginas = loader.load()
        documentos.extend(paginas)

    return documentos


# ============================================================
# 6. DIVISIÓN DE DOCUMENTOS
# ============================================================

@st.cache_resource(show_spinner=False)
def crear_chunks():
    documentos = cargar_documentos()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documentos)
    return chunks


# ============================================================
# 7. BASE DE DATOS VECTORIAL CHROMADB
# ============================================================

@st.cache_resource(show_spinner=False)
def obtener_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    # Si la colección está vacía, indexar los documentos.
    if vectorstore._collection.count() == 0:
        chunks = crear_chunks()
        if not chunks:
            raise ValueError(
                "No se encontraron documentos PDF en data/documents."
            )
        vectorstore.add_documents(documents=chunks)
    return vectorstore

# ============================================================
# 8. RETRIEVER
# ============================================================

@st.cache_resource(show_spinner=False)
def obtener_retriever():
    vectorstore = obtener_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    return retriever

# ============================================================
# 9. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Eres un Tutor Virtual Pedagogico de Machine Learning.
Responde EXCLUSIVAMENTE con la documentacion del curso.

REGLAS:
1. Invoca SIEMPRE 'buscar_documentacion' antes de responder.
2. Usa solo la info recuperada. Si no hay info, responde:
   "Lo siento, ese tema no esta en los materiales del curso."
3. Incluye siempre documento y pagina de cada cita.
4. Responde en espanol, claro, pedagogico y bien estructurado.
"""

# ============================================================
# 10. HERRAMIENTA DE BÚSQUEDA DOCUMENTAL
# ============================================================

@tool
def buscar_documentacion(query: str) -> str:
    """Busca información relevante en la documentación."""
    retriever = obtener_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "NO_INFO: No se encontró información relevante en los apuntes."

    resultados = []
    for doc in docs:
        fuente = doc.metadata.get("source", "Desconocido")
        pagina = doc.metadata.get("page", 0) + 1
        resultados.append(
            f"[Documento: {fuente} | Página: {pagina}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(resultados)


tools = [buscar_documentacion]
tools_by_name = {
    t.name: t for t in tools
}


# ============================================================
# 11. CONSTRUCCIÓN DEL AGENTE LANGGRAPH
# ============================================================

def construir_agente(api_key):
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=api_key,
        max_output_tokens=1024
    )
    llm_with_tools = llm.bind_tools(tools)

    # Nodo del modelo
    def llm_call(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Nodo de herramientas
    def tool_node(state: MessagesState):
        last_message = state["messages"][-1]
        results = []
        for tool_call in last_message.tool_calls:
            tool_func = tools_by_name[tool_call["name"]]
            observation = tool_func.invoke(tool_call["args"])
            results.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": results}

    # Enrutamiento
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return END

    # Construcción del grafo
    grafo = StateGraph(MessagesState)
    grafo.add_node("llm_call", llm_call)
    grafo.add_node("tool_node", tool_node)
    grafo.add_edge(START, "llm_call")   
    grafo.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    grafo.add_edge("tool_node", "llm_call")

    # Memory
    app_tutor = grafo.compile(checkpointer=st.session_state.checkpointer)
    return app_tutor


# Autoconectar al iniciar si existe una API key válida en el entorno
if st.session_state.app_tutor is None:
    _env_key = os.getenv("GEMINI_API_KEY")
    if _env_key:
        try:
            st.session_state.app_tutor = construir_agente(_env_key)
            st.session_state.agent_api_key = _env_key
        except Exception:
            pass

# ============================================================
# 12. FUNCIÓN PARA LIMPIAR RESPUESTAS
# ============================================================

def obtener_texto_limpio(content):
    if isinstance(content, list):
        textos = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and "text" in b
        ]
        return "\n".join(textos)
    return str(content)


# ============================================================
# 13. CONSULTA AL AGENTE y MEDICIÓN DE TIEMPO
# ============================================================

def consultar_tutor(pregunta: str):
    inicio = time.perf_counter()
    resultado = st.session_state.app_tutor.invoke(
        {
            "messages": [HumanMessage(content=pregunta)]
        },
        config={
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }
    )
    respuesta_raw = resultado["messages"][-1].content
    respuesta_limpia = obtener_texto_limpio(
        respuesta_raw
    )
    tiempo_respuesta = time.perf_counter() - inicio
    st.session_state.total_calls += 1
    return respuesta_limpia, tiempo_respuesta


# ============================================================
# 14. INTERFAZ y FLUJO DE CONVERSACIÓN
# ============================================================

with col_right:
    st.title("🎓 Tutor Virtual de Machine Learning")    
    st.divider()
    st.subheader("💬 Conversación")
    for mensaje in st.session_state.messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])
            if (
                mensaje["role"] == "assistant"
                and "tiempo" in mensaje
            ):
                st.caption(
                    f"⏱️ Tiempo de respuesta: "
                    f"{mensaje['tiempo']:.2f} segundos"
                )
    pregunta_usuario = st.chat_input(
        "Escribe tu pregunta sobre Machine Learning..."
    )
    if "pending_question" in st.session_state:
        pregunta_usuario = st.session_state.pop(
            "pending_question"
        )
    if pregunta_usuario:
        # Comprobar que el agente está conectado
        if st.session_state.app_tutor is None:
            st.warning(
                "Primero introduce tu API Key y pulsa "
                "'Conectar tutor' en la barra lateral."
            )
        else:
            # Mostrar pregunta
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": pregunta_usuario
                }
            )
            with st.chat_message("user"):
                st.markdown(pregunta_usuario)

            # Generar respuesta y medir tiempo
            with st.chat_message("assistant"):
                with st.spinner("El tutor está consultando la documentación..."):
                    try:
                        respuesta, tiempo = consultar_tutor(
                            pregunta_usuario
                        )
                        st.markdown(respuesta)
                        st.caption(
                            f"⏱️ Tiempo de respuesta: "
                            f"{tiempo:.2f} segundos"
                        )

                        # Guardar respuesta y tiempo
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": respuesta,
                                "tiempo": tiempo
                            }
                        )
                    except Exception as e:
                        st.error(
                            f"Se produjo un error al consultar "
                            f"el tutor: {e}"
                        )
            st.rerun()

with col_left:    
    st.subheader("💡 Preguntas sugeridas")
    preguntas = [
        "¿Qué es Machine Learning y cuáles son sus tipos principales?",
        "¿Qué es Deep Learning y cuales son sus principios?",
        "¿Cómo funciona el algoritmo K-Means?",
        "¿Podrías darme un ejemplo rápido en código Python del algoritmo que me acabas de explicar?",
        "¿Quién fue el primer presidente de los Estados Unidos?"
    ]
    for idx, pregunta in enumerate(preguntas, 1):
        if st.button(
            f"{pregunta}",
            use_container_width=False,
            key=f"sb_{idx}",
            help=pregunta,
        ):
            st.session_state.pending_question = pregunta
            st.rerun()

    if st.button("🔄 Nueva conversación",use_container_width=False):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    

    st.subheader("📄Documentos")
    try:
        documentos = cargar_documentos()
        archivos = (
                    list(DOCUMENTS_PATH.glob("*.pdf"))
                    if DOCUMENTS_PATH.exists() else []
                )
        for archivo in archivos:
            st.caption(f"✅ {archivo.name}")
    except Exception as e:
        st.error(f"Error: {e}")
    chunks = crear_chunks()

    st.subheader("📊 Métricas")
    st.markdown(f"""        
        - **Modelo:** <small>{MODEL}</small>
        - **Consultas:** {st.session_state.total_calls}
        - **Total Documentos:** {len(archivos)}
        - **Total Páginas:** {len(documentos)}
        - **Total Chunks:** {len(chunks)}
    """,
    unsafe_allow_html=True,)   


# ============================================================

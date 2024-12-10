import os, uuid
from flask import Flask, request, render_template, session, redirect, url_for, flash

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)
app.secret_key = "your-secret-key"

# Global dictionary to hold sessions: { session_id: {...} }
# Each session stores 'name', 'conversational_chain', 'chat_history'
global_sessions = {}

BOT_TEMPLATE = '''
<div class="d-flex align-items-start mb-3">
    <img src="https://cdn-icons-png.flaticon.com/512/170/170579.png" class="rounded-circle me-2" style="height:40px;width:40px;">
    <div class="p-2 bg-secondary text-white rounded">
        {{msg}}
    </div>
</div>
'''

USER_TEMPLATE = '''
<div class="d-flex justify-content-end mb-3">
    <div class="p-2 bg-primary text-white rounded" style="max-width:60%;">
        {{msg}}
    </div>
    <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512"
         class="rounded-circle ms-2" style="height:40px;width:40px;">
</div>
'''

def prepare_and_split_docs(pdf_paths):
    split_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=10000,
            chunk_overlap=2000,
            disallowed_special=(),
            separators=["\n\n", "\n", " "]
        )
        split_docs.extend(splitter.split_documents(documents))
    return split_docs

def ingest_into_vectordb(split_docs, session_id):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = f'vectorstore/{session_id}_db_faiss'
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    return db

def get_conversation_chain(retriever):
    llm = Ollama(model="llama3.2-vision")
    contextualize_q_system_prompt = (
        """You are an intelligent assistant designed to answer questions
        based on provided documents. When answering, rely solely on the
        content of the documents retrieved, and aim to provide concise,
        relevant, and well-reasoned answers. If multiple documents provide related
        information, synthesize it into a coherent response. If the documents do not
        contain information on the topic, respond with 'The documents do not contain
        relevant information on this question.' Always cite specific parts of the documents
        where applicable and maintain an objective and factual tone. where applicable and maintain an objective and factual tone. If you provide lists, use proper markdown bullet points. For example:
        - First item
        - Second item
        - Third item"""
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        """You are an intelligent assistant designed to answer questions based solely on the provided documents. When responding:
            - Rely only on the given documents. If no relevant information is found, respond with: "The documents do not contain relevant information on this question."
            - Synthesize all provided information if multiple documents discuss related topics.
            - Highlight the key point in your answer.
            - Use an objective and factual tone without citing specific parts of the documents.
            - If you provide lists, format them in markdown:
                - Example item
                - Another example item
            Your goal is to deliver concise, relevant, and well-reasoned answers strictly derived from the given documents.""" 
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def calculate_similarity_score(answer: str, context_docs: list) -> float:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    context_texts = [doc.page_content for doc in context_docs]
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    context_embeddings = model.encode(context_texts, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)
    max_score = similarities.max().item()
    return max_score


@app.route('/')
def home():
    return redirect(url_for('sessions_dashboard'))

@app.route('/sessions', methods=['GET', 'POST'])
def sessions_dashboard():
    if request.method == 'POST':
        # Handle creating a new session
        session_name = request.form.get('session_name', 'Untitled Session')
        new_session_id = str(uuid.uuid4())
        global_sessions[new_session_id] = {
            'name': session_name,
            'conversational_chain': None,
            'chat_history': []
        }
        flash("New session created!", "success")
        return redirect(url_for('sessions_dashboard'))

    return render_template('sessions.html', sessions=global_sessions)

@app.route('/sessions/<session_id>', methods=['GET', 'POST'])
def chat(session_id):
    if session_id not in global_sessions:
        flash("Session does not exist!", "danger")
        return redirect(url_for('sessions_dashboard'))

    data = global_sessions[session_id]
    chat_history = data.get('chat_history', [])

    if request.method == 'POST':
        # Could be file upload or user message
        if 'pdf_files' in request.files:
            uploaded_files = request.files.getlist('pdf_files')
            pdf_paths = []
            for f in uploaded_files:
                if f.filename.endswith('.pdf'):
                    os.makedirs('uploads', exist_ok=True)
                    save_path = os.path.join('uploads', f.filename)
                    f.save(save_path)
                    pdf_paths.append(save_path)
            if pdf_paths:
                split_docs = prepare_and_split_docs(pdf_paths)
                vector_db = ingest_into_vectordb(split_docs, session_id)
                retriever = vector_db.as_retriever()
                conversational_chain = get_conversation_chain(retriever)
                data['conversational_chain'] = conversational_chain
                data['chat_history'] = []
                global_sessions[session_id] = data
                flash("Documents processed and vector database created!", "success")
            return redirect(url_for('chat', session_id=session_id))

        user_input = request.form.get('user_input')
        if user_input and data['conversational_chain']:
            conversational_chain = data['conversational_chain']
            response = conversational_chain.invoke({"input": user_input}, 
                                                   config={"configurable": {"session_id": session_id}})
            context_docs = response.get('context', [])
            chat_history.append({"user": user_input, "bot": response['answer'], "context_docs": context_docs})
            data['chat_history'] = chat_history
            global_sessions[session_id] = data
            return redirect(url_for('chat', session_id=session_id))
        else:
            flash("Either no documents have been processed yet, or no user input was provided.", "warning")
            return redirect(url_for('chat', session_id=session_id))

    return render_template('chat.html',
                           session_id=session_id,
                           session_name=data['name'],
                           chat_history=chat_history,
                           user_template=USER_TEMPLATE,
                           bot_template=BOT_TEMPLATE)

@app.route('/sessions/<session_id>/rename', methods=['POST'])
def rename_session(session_id):
    if session_id not in global_sessions:
        flash("Session not found", "danger")
        return redirect(url_for('sessions_dashboard'))
    new_name = request.form.get('new_name', 'Untitled Session')
    global_sessions[session_id]['name'] = new_name
    flash("Session renamed successfully!", "success")
    return redirect(url_for('sessions_dashboard'))

@app.route('/sessions/<session_id>/delete', methods=['POST'])
def delete_session(session_id):
    if session_id in global_sessions:
        del global_sessions[session_id]
        flash("Session deleted successfully!", "success")
    else:
        flash("Session not found", "danger")
    return redirect(url_for('sessions_dashboard'))

@app.route('/sessions/<session_id>/clear')
def clear_chat(session_id):
    if session_id in global_sessions:
        global_sessions[session_id]['chat_history'] = []
        flash("Chat history cleared!", "success")
    else:
        flash("Session not found", "danger")
    return redirect(url_for('chat', session_id=session_id))

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port='5040', host='0.0.0.0')

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

cache_base_dir = os.path.join(current_dir, "cache")
os.makedirs(cache_base_dir, exist_ok=True)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from agno.embedder.google import GeminiEmbedder
from typing import List
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from Agents.search_agent import Search
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from Agents.answer_grader_agent import AnswerGrader
from Agents.hallucinator_agent import HallucinationGrader
from Agents.grader_agent import Grader
from Agents.question_rewriter import QuestionRewriter
from Agents.abstractor_agent import Abstractor
from pprint import pprint
from langgraph.graph import END
from Agents.reflectionAgent import IntrospectiveAgent
# Additional imports for document loading
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
import json
import pandas as pd
import pathlib
import hashlib
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_dir = os.path.join(cache_base_dir, "llm_cache", f"map_cache_{hashed_llm}")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=cache_dir),
    )


set_llm_cache(GPTCache(init_gptcache))


class LoadDocuments:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_extension = pathlib.Path(file_path).suffix.lower()

    def load_documents(self):
        """Load documents based on file extension"""
        if self.file_extension == '.csv':
            return self._load_csv()
        elif self.file_extension == '.pdf':
            return self._load_pdf()
        elif self.file_extension in ['.txt', '.md']:
            return self._load_text()
        elif self.file_extension in ['.docx', '.doc']:
            return self._load_word()
        elif self.file_extension == '.json':
            return self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.file_extension}")

    def _load_csv(self):
        """Load CSV files with general format handling"""
        df = pd.read_csv(self.file_path)
        documents = []
        questions = []
        translations = []

        for _, row in df.iterrows():
            # General CSV format - combine all columns into content
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
            documents.append(Document(
                page_content=content,
                metadata={"row_index": row.name, "source": self.file_path}
            ))
            questions.append("General information query")
            translations.append(content[:200] + "..." if len(content) > 200 else content)

        return documents, questions, translations

    def _load_pdf(self):
        """Load PDF files"""
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        documents = []
        questions = []
        translations = []

        for i, page in enumerate(pages):
            documents.append(Document(
                page_content=page.page_content,
                metadata={"page": i + 1, "source": self.file_path}
            ))
            questions.append(f"Information from page {i + 1}")
            translations.append(page.page_content[:200] + "..." if len(page.page_content) > 200 else page.page_content)

        return documents, questions, translations

    def _load_text(self):
        """Load text/markdown files"""
        loader = TextLoader(self.file_path, encoding='utf-8')
        docs = loader.load()
        documents = []
        questions = []
        translations = []

        for doc in docs:
            # Split large text files into chunks
            content = doc.page_content
            if len(content) > 2000:
                chunks = [content[i:i+2000] for i in range(0, len(content), 1800)]
                for j, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"chunk": j + 1, "source": self.file_path}
                    ))
                    questions.append(f"Information from text chunk {j + 1}")
                    translations.append(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            else:
                documents.append(Document(
                    page_content=content,
                    metadata={"source": self.file_path}
                ))
                questions.append("Text document information")
                translations.append(content[:200] + "..." if len(content) > 200 else content)

        return documents, questions, translations

    def _load_word(self):
        """Load Word documents"""
        try:
            loader = UnstructuredWordDocumentLoader(self.file_path)
            docs = loader.load()
            documents = []
            questions = []
            translations = []

            for doc in docs:
                documents.append(Document(
                    page_content=doc.page_content,
                    metadata={"source": self.file_path}
                ))
                questions.append("Word document information")
                translations.append(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)

            return documents, questions, translations
        except Exception as e:
            raise ValueError(f"Error loading Word document: {str(e)}")

    def _load_json(self):
        """Load JSON files"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        questions = []
        translations = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                content = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
                documents.append(Document(
                    page_content=content,
                    metadata={"item": i + 1, "source": self.file_path}
                ))
                questions.append(f"JSON item {i + 1} information")
                translations.append(content[:200] + "..." if len(content) > 200 else content)
        elif isinstance(data, dict):
            content = json.dumps(data, indent=2)
            documents.append(Document(
                page_content=content,
                metadata={"source": self.file_path}
            ))
            questions.append("JSON document information")
            translations.append(content[:200] + "..." if len(content) > 200 else content)
        else:
            content = str(data)
            documents.append(Document(
                page_content=content,
                metadata={"source": self.file_path}
            ))
            questions.append("JSON data information")
            translations.append(content)

        return documents, questions, translations
    
class MyEmbeddings(EmbeddingFunction):
    def __init__(self, model: str = None):
        self.embedder = GeminiEmbedder()
        # Create embedding cache directory
        self.cache_dir = os.path.join(cache_base_dir, "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "embedding_cache.json")
        self._cache = self._load_cache()

    def _load_cache(self):
        """Load cache from file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._cache:
                embeddings.append(self._cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)
        
        # Batch process uncached texts
        if uncached_texts:
            # Process in smaller batches to avoid timeout
            batch_size = 5
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                for j, text in enumerate(batch):
                    embedding = self.embedder.get_embedding(text)
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self._cache[text_hash] = embedding
                    embeddings[uncached_indices[i+j]] = embedding
        
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self._cache:
            return self._cache[query_hash]
        
        embedding = self.embedder.get_embedding(query)
        self._cache[query_hash] = embedding
        self._save_cache()  # Save cache after each update
        return embedding
    
        
class ADAPTIVE_RAG:
    def __init__(self, model, api_key, k, file_path):
        self.load_documents = LoadDocuments(file_path)
        self.chat_memory = []
        
        # Create vectorstore directory inside cache
        self.vectorstore_dir = os.path.join(cache_base_dir, "vectorstore", "chroma_db")
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # Check if vectorstore already exists to avoid reprocessing
        if self._vectorstore_exists():
            print("Loading existing vectorstore...")
            self.embd = MyEmbeddings()
            self.vectorstore = Chroma(
                collection_name="rag-chroma",
                embedding_function=self.embd,
                persist_directory=self.vectorstore_dir
            )
        else:
            print("Creating new vectorstore...")
            self.documents, self.questions, self.translations = self.load_documents.load_documents()
            self.embd = MyEmbeddings()
            # Use SemanticChunker for better semantic understanding
            self.text_splitter = SemanticChunker(self.embd)
            self.doc_splits = self.text_splitter.split_documents(self.documents)
            
            self.vectorstore = Chroma.from_documents(
                documents=self.doc_splits,
                collection_name="rag-chroma",
                embedding=self.embd,
                persist_directory=self.vectorstore_dir
            )
        
        self.compressor = CohereRerank(model="rerank-english-v3.0")
        
        # Use async retriever with smaller k for faster retrieval
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": min(k, 3)})
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever
        )
        self.model = model
        self.api_key = api_key
        self.k = k
        self.llm = ChatGroq(model=model, api_key=api_key)
        
        self.generator_prompt = """
        You are an AI assistant for question-answering tasks. Use the provided context along with the previous chat history to deliver a precise and concise response. If the information is insufficient or unclear, acknowledge that you don't know. Keep the answer brief (three sentences or less) while maintaining clarity and relevance.

        """
        self.human_prompt = """
            Inputs:

            Question: {question}
            Context: {context}
            Chat History: {chat_history}
            Output:
            Answer:
        """
        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.generator_prompt), 
                ("human", self.human_prompt)
            ]
        )
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        self.recursion_limit = 7
        self.recursion_counter = 0
        
        self.instropection_agent = IntrospectiveAgent(
            model_id=self.model,
        )

    def _vectorstore_exists(self):
        """Check if vectorstore already exists"""
        chroma_files = ['chroma.sqlite3', 'index']
        return all(os.path.exists(os.path.join(self.vectorstore_dir, f)) for f in chroma_files)
        
    def retrieve(self, state):
        question = state["question"]
        documents = self.compression_retriever.invoke(question)   
        return {"documents": documents, "question": question}
    def abstraction(self, state):
        content = state["documents"]
        abstractor = Abstractor(self.model)
        extractions = abstractor.abstract(content)
        print("extracted info: ", extractions)
        return {"documents": content, "question": state["question"], "extractions": extractions}
    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question, "chat_history": self.chat_memory})
        if len(self.chat_memory) < 5:
            self.chat_memory.append(generation)
        else:
            self.chat_memory.pop(0)
            self.chat_memory.append(generation)
        if state.get("extractions") is not None:
            return {"documents": documents, "question": question,"extractions": state["extractions"], "generation": generation}
        return {"documents": documents, "question": question, "generation": generation}
        
    def fast_generate(self, state):
        """Fast generation without abstraction for simple queries"""
        question = state["question"]
        documents = state["documents"]
        
        # Use simpler prompt for faster processing
        fast_prompt = f"Context: {documents}\n\nQuestion: {question}\n\nAnswer:"
        generation = self.llm.invoke(fast_prompt).content
        
        if len(self.chat_memory) < 5:
            self.chat_memory.append(generation)
        else:
            self.chat_memory.pop(0)
            self.chat_memory.append(generation)
            
        return {"documents": documents, "question": question, "generation": generation}
        
    def grade_documents(self, state):
        """Optimized document grading with early stopping"""
        question = state["question"]
        print("quest")
        documents = state["documents"]
        filtered_docs = []
        
        # Process only top 3 documents for speed
        top_docs = documents[:3] if len(documents) > 3 else documents
        
        for d in top_docs:
            if len(filtered_docs) >= 2:  # Early stopping - we have enough good docs
                break
            grader = Grader(self.model)
            grade = grader.grade_documents(question, d.page_content)
            if grade == "yes":
                filtered_docs.append(d)
                
        return {"documents": filtered_docs, "question": question}
        
    def transform_query(self, state):
        question = state["question"]
        documents = state["documents"]
        questionRewriter = QuestionRewriter(self.model)
        better_question = questionRewriter.re_write_question(question)
        return {"documents": documents, "question": better_question}
        
    def web_search(self, state):
        question = state["question"]
        search = Search(self.k)
        docs = search.web_search(question)
        return {"documents": docs, "question": question}
        
    def decide_to_generate(self, state):
        question = state["question"]
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        if not filtered_documents:
            return "transform_query"
        else:
            return "generate"
            
    def grade_generation_v_documents_and_question(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        print("---CHECK HALLUCINATIONS---")
        hallucinationGrader = HallucinationGrader(self.model)
        grade = hallucinationGrader.grade_hallucinations(documents, generation)
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            answerGrader = AnswerGrader(self.model)
            grade = answerGrader.grade_answer(question, generation)
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        
    def introspective_agent_response(self, state):
        question = state["question"]
        extracted_info = state.get("extractions")
        retrieved_docs = state["documents"]
        
        final_prompt = f"""
            This is the question given by the user : {question}
            These are the most relevant retrieved documents : {retrieved_docs[0]} 
        """
        response = self.instropection_agent.chat(final_prompt)
        return {"generation": str(response), "question": question, "extractions": extracted_info, "documents": retrieved_docs}
        
    def track_recursion_and_retrieve(self, state):
        """
        Track the number of recursions when going to 'transform_query'.
        End the workflow if the recursion limit is reached.
        """
        if self.recursion_counter < self.recursion_limit:
            self.recursion_counter += 1
            return "retrieve"
        else:
            return "end_due_to_limit"

    def end_due_to_limit(self, state):
        """
        End the workflow because the recursion limit was reached.
        """
        print("Recursion limit reached. Ending workflow.")
        return END
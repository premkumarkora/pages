import os
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass
import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import tempfile
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from langchain.schema import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = ""
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = ""
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'databases' not in st.session_state:
        st.session_state.databases = {}

init_session_state()

# Define three student collections.
DatabaseType = Literal["academic", "faq", "campus"]
PERSIST_DIRECTORY = "db_storage"

@dataclass
class CollectionConfig:
    name: str
    description: str
    collection_name: str  # This will be used as Qdrant collection name

# Collection configurations for student support
COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "academic": CollectionConfig(
        name="Academic Resources",
        description="Academic documents, course materials, research articles, and textbooks.",
        collection_name="academic_collection"
    ),
    "faq": CollectionConfig(
        name="Student FAQs",
        description="Frequently asked questions, guidance, and policies for students.",
        collection_name="faq_collection"
    ),
    "campus": CollectionConfig(
        name="Campus & Administrative Support",
        description="Information on campus services, administrative support, and student affairs.",
        collection_name="campus_collection"
    )
}

def initialize_models():
    """Initialize OpenAI models and Qdrant client"""
    if (st.session_state.openai_api_key and 
        st.session_state.qdrant_url and 
        st.session_state.qdrant_api_key):
        
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.llm = ChatOpenAI(temperature=0)
        
        try:
            # Initialize Qdrant client with session state credentials
            client = QdrantClient(
                url=st.session_state.qdrant_url,
                api_key=st.session_state.qdrant_api_key
            )
            
            # Test connection
            client.get_collections()
            vector_size = 1536  
            st.session_state.databases = {}
            for db_type, config in COLLECTIONS.items():
                try:
                    client.get_collection(config.collection_name)
                except Exception:
                    # Create collection if it doesn't exist
                    client.create_collection(
                        collection_name=config.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                
                st.session_state.databases[db_type] = Qdrant(
                    client=client,
                    collection_name=config.collection_name,
                    embeddings=st.session_state.embeddings
                )
            
            return True
        except Exception as e:
            st.error(f"Failed to connect to Qdrant: {str(e)}")
            return False
    return False

def process_document(file) -> List[Document]:
    """Process uploaded PDF document"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
            
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        return texts
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

def create_routing_agent() -> Agent:
    """Creates a routing agent using phidata framework for student queries"""
    return Agent(
        model=OpenAIChat(
            id="gpt-4o",
            api_key=st.session_state.openai_api_key
        ),
        tools=[],
        description="""You are a query routing expert. Your only job is to analyze student queries and determine which database they should be routed to.
You must respond with exactly one of these three options: 'academic', 'faq', or 'campus'. The student's question is: {question}""",
        instructions=[
            "Follow these rules strictly:",
            "1. For questions about course content, academic materials, research articles, or textbooks → return 'academic'",
            "2. For questions about common student queries, guidance, policies, or how-to instructions → return 'faq'",
            "3. For questions about campus services, administrative support, or student affairs → return 'campus'",
            "4. Return ONLY the database name, no other text or explanation",
            "5. If you're not confident about the routing, return an empty response"
        ],
        markdown=False,
        show_tool_calls=False
    )

def route_query(question: str) -> Optional[DatabaseType]:
    """Route query by comparing vector similarity scores across all student collections.
    Falls back to LLM routing if needed."""
    try:
        best_score = -1
        best_db_type = None
        all_scores = {}  # Store all scores for debugging
        
        # Search each database and compare relevance scores
        for db_type, db in st.session_state.databases.items():
            results = db.similarity_search_with_score(
                question,
                k=3
            )
            
            if results:
                avg_score = sum(score for _, score in results) / len(results)
                all_scores[db_type] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_db_type = db_type
        
        confidence_threshold = 0.5
        if best_score >= confidence_threshold and best_db_type:
            st.success(f"Using vector similarity routing: {best_db_type} (confidence: {best_score:.3f})")
            return best_db_type
            
        st.warning(f"Low confidence scores (below {confidence_threshold}), falling back to LLM routing")
        
        # Fallback to LLM routing
        routing_agent = create_routing_agent()
        response = routing_agent.run(question)
        
        db_type = (response.content
                  .strip()
                  .lower()
                  .translate(str.maketrans('', '', '`\'"')))
        
        if db_type in COLLECTIONS:
            st.success(f"Using LLM routing decision: {db_type}")
            return db_type
            
        st.warning("No suitable database found, will use web search fallback")
        return None
        
    except Exception as e:
        st.error(f"Routing error: {str(e)}")
        return None

def create_fallback_agent(chat_model: BaseLanguageModel):
    """Create a LangGraph agent for web research."""
    
    def web_research(query: str) -> str:
        """Web search with result formatting."""
        try:
            search = DuckDuckGoSearchRun(num_results=5)
            results = search.run(query)
            return results
        except Exception as e:
            return f"Search failed: {str(e)}. Providing answer based on general knowledge."

    tools = [web_research]
    
    agent = create_react_agent(model=chat_model,
                             tools=tools,
                             debug=False)
    
    return agent

def query_database(db: Qdrant, question: str) -> tuple[str, list]:
    """Query the chosen student database and return answer and relevant documents"""
    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        relevant_docs = retriever.get_relevant_documents(question)

        if relevant_docs:
            # Updated prompt focusing on student support and academic context
            retrieval_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant specialized in student support.
Answer student queries using the provided academic or administrative context.
Ground your responses strictly in the context to avoid hallucinations.
If the context does not fully answer the question, clearly acknowledge the limitations."""),
                ("human", "Here is the context:\n{context}"),
                ("human", "Question: {input}"),
                ("assistant", "I'll provide an answer based on the above context."),
                ("human", "Please provide your answer:"),
            ])
            combine_docs_chain = create_stuff_documents_chain(st.session_state.llm, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            response = retrieval_chain.invoke({"input": question})
            return response['answer'], relevant_docs
        
        raise ValueError("No relevant documents found in the selected student database")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try rephrasing your question.", []

def _handle_web_fallback(question: str) -> tuple[str, list]:
    st.info("No relevant documents found. Searching web for additional academic context...")
    fallback_agent = create_fallback_agent(st.session_state.llm)
    
    with st.spinner('Researching...'):
        agent_input = {
            "messages": [
                HumanMessage(content=f"Research and provide a detailed answer for the student query: '{question}'")
            ],
            "is_last_step": False
        }
        
        try:
            response = fallback_agent.invoke(agent_input, config={"recursion_limit": 100})
            if isinstance(response, dict) and "messages" in response:
                answer = response["messages"][-1].content
                return f"Web Search Result:\n{answer}", []
                
        except Exception:
            fallback_response = st.session_state.llm.invoke(question).content
            return f"Web search unavailable. General response: {fallback_response}", []

def main():
    """Main application function for the Student Support RAG Assistant."""
    st.set_page_config(page_title="RAG-based AI Assistant for Student Support", page_icon="🎓")
    st.title("🎓 RAG-based AI Assistant for Student Support")
    
    # Sidebar for API keys and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key
        api_key = st.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=st.session_state.openai_api_key,
            key="api_key_input"
        )
        
        # Qdrant Configuration
        qdrant_url = st.text_input(
            "Enter Qdrant URL:",
            value=st.session_state.qdrant_url,
            help="Example: https://your-cluster.qdrant.tech"
        )
        
        qdrant_api_key = st.text_input(
            "Enter Qdrant API Key:",
            type="password",
            value=st.session_state.qdrant_api_key
        )
        
        # Update session state
        if api_key:
            st.session_state.openai_api_key = api_key
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        if qdrant_api_key:
            st.session_state.qdrant_api_key = qdrant_api_key
            
        # Initialize models if all credentials are provided
        if (st.session_state.openai_api_key and 
            st.session_state.qdrant_url and 
            st.session_state.qdrant_api_key):
            if initialize_models():
                st.success("Connected to OpenAI and Qdrant successfully!")
            else:
                st.error("Failed to initialize. Please check your credentials.")
        else:
            st.warning("Please enter all required credentials to continue")
            st.stop()

        st.markdown("---")

    st.header("Academic Document Upload")
    st.info("Upload academic PDFs to populate the student support databases with structured content for improved, context-aware responses.")
    
    # Create tabs for each student collection
    tabs = st.tabs([config.name for config in COLLECTIONS.values()])
    for (collection_type, config), tab in zip(COLLECTIONS.items(), tabs):
        with tab:
            st.write(config.description)
            uploaded_files = st.file_uploader(
                f"Upload PDF documents to {config.name}",
                type="pdf",
                key=f"upload_{collection_type}",
                accept_multiple_files=True  
            )
            
            if uploaded_files:
                with st.spinner('Processing documents...'):
                    all_texts = []
                    for uploaded_file in uploaded_files:
                        texts = process_document(uploaded_file)
                        all_texts.extend(texts)
                    
                    if all_texts:
                        db = st.session_state.databases[collection_type]
                        db.add_documents(all_texts)
                        st.success("Documents processed and added to the database!")

    # Query section
    st.header("Ask Your Question")
    st.info("Enter your student query below to get answers grounded in academic or administrative context.")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner('Finding answer...'):
            # Route the question to the best matching collection
            collection_type = route_query(question)
            
            if collection_type is None:
                # Use web search fallback directly if routing fails
                answer, _ = _handle_web_fallback(question)
                st.write("### Answer (from web search fallback)")
                st.write(answer)
            else:
                st.info(f"Routing question to: {COLLECTIONS[collection_type].name}")
                db = st.session_state.databases[collection_type]
                answer, _ = query_database(db, question)
                st.write("### Answer")
                st.write(answer)

if __name__ == "__main__":
    main()

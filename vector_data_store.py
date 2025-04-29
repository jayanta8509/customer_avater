import bs4
import os
import asyncio
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Set user agent environment variable if not already set
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


async def add_website_data(documents, save_path="website_data"):
    """
    Add website data to the vector store and save it.
    
    Args:
        documents: Single URL string or list of URLs to scrape
        save_path: Path where to save the vector store (default: 'website_data')
    
    Returns:
        The loaded vector store
    """
    # Initialize vector store with fixed dimension size
    embed_dimension = len(embeddings.embed_query("initialize vector store"))
    index = faiss.IndexFlatL2(embed_dimension)
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Convert single URL to a list if needed
    if isinstance(documents, str):
        web_paths = [documents]
    else:
        web_paths = documents

    # Use run_in_executor to make the blocking call asynchronous
    loop = asyncio.get_event_loop()
    
    # Define a function to perform the loading work
    def load_documents():
        try:
            loader = WebBaseLoader(
                web_paths=web_paths,
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header", "content", "main")
                    )
                ),
            )
            
            try:
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                all_splits = text_splitter.split_documents(docs)
                
                # Add documents to the vector store
                vector_store.add_documents(all_splits)
            except ValueError as e:
                print(f"Error parsing website content: {str(e)}")
                # Try a more basic approach without the strainer
                basic_loader = WebBaseLoader(
                    web_paths=web_paths,
                    bs_kwargs={}  # No strainer, just load everything
                )
                try:
                    docs = basic_loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    all_splits = text_splitter.split_documents(docs)
                    vector_store.add_documents(all_splits)
                    print("Used fallback loader successfully")
                except Exception as inner_e:
                    print(f"Fallback loader also failed: {str(inner_e)}")
                    # Create a simple document with the URL if all else fails
                    from langchain_core.documents import Document
                    for url in web_paths:
                        vector_store.add_documents([Document(page_content=f"Website: {url}", metadata={"source": url})])
                    print("Added minimal URL information to vector store")
            
            # Save the vector store to disk
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            vector_store.save_local(save_path)
            
            # Load and return the saved vector store
            return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            # If an error occurs, still try to return an existing saved store or an empty one
            if os.path.exists(save_path):
                return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            return vector_store
    
    # Run the blocking operation in a thread pool
    return await loop.run_in_executor(None, load_documents)


async def search_website_data(query, save_path="website_data", k=5):
    """
    Search for relevant documents in the saved vector store.
    
    Args:
        query: The search query
        save_path: Path where the vector store is saved
        k: Number of documents to return
    
    Returns:
        List of relevant documents
    """
    if not os.path.exists(save_path):
        return []
    
    # Use run_in_executor to make the blocking call asynchronous
    loop = asyncio.get_event_loop()
    
    def load_and_search():
        # Load the saved vector store
        loaded_vector_store = FAISS.load_local(
            save_path, embeddings, allow_dangerous_deserialization=True
        )
        
        # Search for similar documents
        results = loaded_vector_store.similarity_search(query, k=k)
        return results
    
    # Run the blocking operation in a thread pool
    return await loop.run_in_executor(None, load_and_search)


# # Example usage
# if __name__ == "__main__":
#     # Example 1: Store data from a single website
#     single_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
#     print(f"Loading data from: {single_url}")
#     vector_db = add_website_data(single_url, save_path="single_website_data")
#     print("Single website data stored successfully!")

#     # Example 2: Store data from multiple websites
#     multiple_urls = [
#         "https://lilianweng.github.io/posts/2023-06-23-agent/",
#         "https://python.langchain.com/docs/get_started/introduction"
#     ]
#     print(f"\nLoading data from {len(multiple_urls)} websites")
#     vector_db = add_website_data(multiple_urls, save_path="multiple_websites_data")
#     print("Multiple websites data stored successfully!")

#     # Example search using the vector_db directly
#     print("\nSearching for 'agent architecture' in the vector_db directly:")
#     results = vector_db.similarity_search("agent architecture", k=4)
#     for i, result in enumerate(results):
#         print(f"\nResult {i+1}:")
#         print(f"Source: {result.metadata.get('source', 'Unknown')}")
#         print(f"Content: {result.page_content[:150]}...")  # First 150 chars

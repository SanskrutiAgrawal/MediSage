# test_rag_agent.py

import logging
from rag_agent import MedicalRAG
from config import Config

# Set up basic logging to see what the agent is doing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_test():
    """
    Initializes the MedicalRAG agent and processes a test query.
    """
    print("--- RAG Agent Test Initializing ---")
    
    # 1. Load the configuration we have carefully prepared
    print("Loading configuration...")
    try:
        config = Config()
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # 2. Initialize the MedicalRAG agent
    print("Initializing MedicalRAG agent...")
    try:
        rag_agent = MedicalRAG(config)
        print("MedicalRAG agent initialized successfully.")
    except Exception as e:
        print(f"Error initializing MedicalRAG agent: {e}")
        return

    # 3. Define a query that is relevant to the documents you ingested
    #    (Change this query to match your PDF content)
    query = "What are the deep learning techniques for brain tumor diagnosis?"
    
    print(f"\n--- Processing Query ---\nQuery: '{query}'\n")

    # 4. Call the process_query method
    try:
        result = rag_agent.process_query(query)
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Print the results in a readable format
    print("\n--- Agent Response Received ---")
    
    if result:
        print("\n[Final Response]:")
        # The actual response content might be nested inside an AIMessage object
        response_content = result.get("response", "No response found.")
        if hasattr(response_content, 'content'):
             print(response_content.content)
        else:
             print(response_content)

        print("\n[Confidence Score]:")
        print(result.get("confidence", "N/A"))

        print("\n[Retrieved Sources]:")
        sources = result.get("sources", [])
        if sources:
            for i, source in enumerate(sources):
                print(f"  {i+1}. Title: {source['title']}")
                # print(f"     Path: {source['path']}") # You can uncomment this for more detail
        else:
            print("No sources were cited.")
    else:
        print("The agent did not return a result.")
        
    print("\n--- Test Complete ---")


if __name__ == "__main__":
    run_test()
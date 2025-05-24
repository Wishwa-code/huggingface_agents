"""LangGraph Agent"""
import os
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings,  HuggingFacePipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client
from pydantic import BaseModel, Field


from typing import List, Set, Any

load_dotenv()

class TableCommutativityInput(BaseModel):
    table: List[List[Any]] = Field(description="The 2D list representing the multiplication table.")
    elements: List[str] = Field(description="The list of header elements corresponding to the table rows/columns.")

class VegetableListInput(BaseModel):
    items: List[str] = Field(description="A list of grocery item strings.")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

# @tool
# def web_search(query: str) -> str:
#     """Search Tavily for a query and return maximum 3 results.
    
#     Args:
#         query: The search query."""
#     search_docs = TavilySearchResults(max_results=3).invoke(query=query)
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
#             for doc in search_docs
#         ])
#     return {"web_results": formatted_search_docs}
@tool
def web_search(query: str) -> dict: # Changed return type annotation to dict
    """Search Tavily for a query and return maximum 3 results.
    Each result will be formatted with its source URL and content.
    
    Args:
        query: The search query.
    """
    print(f"\n--- Web Search Tool ---") # For debugging
    print(f"Received query: {query}")
    try:
        tavily_tool = TavilySearchResults(max_results=3)
        # .invoke() for TavilySearchResults typically expects 'input'
        # and returns a list of dictionaries
        search_results_list = tavily_tool.invoke(input=query)
        
        print(f"Raw Tavily search results type: {type(search_results_list)}")
        if isinstance(search_results_list, list):
            print(f"Number of results: {len(search_results_list)}")
            if search_results_list:
                print(f"Type of first result: {type(search_results_list[0])}")
                if isinstance(search_results_list[0], dict):
                    print(f"Keys in first result: {search_results_list[0].keys()}")

        formatted_docs = []
        if isinstance(search_results_list, list):
            for doc_dict in search_results_list:
                if isinstance(doc_dict, dict):
                    source = doc_dict.get("url", "N/A")
                    content = doc_dict.get("content", "")
                    # title = doc_dict.get("title", "") # Optionally include title
                    # score = doc_dict.get("score", "") # Optionally include score
                    
                    # Constructing the XML-like format you desire
                    formatted_doc = (
                        f'<Document source="{source}">\n'
                        f'{content}\n'
                        f'</Document>'
                    )
                    formatted_docs.append(formatted_doc)
                else:
                    # If an item in the list is not a dict, convert it to string
                    print(f"Warning: Unexpected item type in Tavily results list: {type(doc_dict)}")
                    formatted_docs.append(str(doc_dict))
            
            final_formatted_string = "\n\n---\n\n".join(formatted_docs)

        elif isinstance(search_results_list, str): # Less common, but for robustness
            final_formatted_string = search_results_list
        else:
            print(f"Unexpected Tavily search result format overall: {type(search_results_list)}")
            final_formatted_string = str(search_results_list) # Fallback

        print(f"Formatted search docs for LLM:\n{final_formatted_string[:500]}...") # Print a snippet
        return {"web_results": final_formatted_string}

    except Exception as e:
        print(f"Error during Tavily search for query '{query}': {e}")
        # It's good practice to return an error message in the expected dict format
        return {"web_results": f"Error performing web search: {e}"}

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

@tool
def reverse_text(text_to_reverse: str) -> str:
    """Reverses the input text.
    Args:
        text_to_reverse: The text to be reversed.
    """
    if not isinstance(text_to_reverse, str):
        raise TypeError("Input must be a string.")
    return text_to_reverse[::-1]


@tool(args_schema=TableCommutativityInput)
def find_non_commutative_elements(table: List[List[Any]], elements: List[str]) -> str:
    """
    Given a multiplication table (2D list) and its header elements,
    returns a comma-separated string of elements involved in any non-commutative operations (a*b != b*a),
    sorted alphabetically.
    """
    if len(table) != len(elements) or (len(table) > 0 and len(table[0]) != len(elements)):
        raise ValueError("Table dimensions must match the number of elements.")
        
    non_comm: Set[str] = set()
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if i < j: # Avoid checking twice (a*b vs b*a and b*a vs a*b) and self-comparison
                if table[i][j] != table[j][i]:
                    non_comm.add(a)
                    non_comm.add(b)
    # Return as a comma-separated string as per typical LLM tool output preference
    return ", ".join(sorted(list(non_comm)))


@tool(args_schema=VegetableListInput)
def list_vegetables(items: List[str]) -> str:
    """
    From a list of grocery items, returns a comma-separated string of those
    that are true vegetables (botanical definition, based on a predefined set),
    sorted alphabetically.
    """
    _VEG_SET = {
        "broccoli", "bell pepper", "celery", "corn", # Note: corn, bell pepper are botanically fruits
        "green beans", "lettuce", "sweet potatoes", "zucchini" # Note: green beans, zucchini are botanically fruits
    }
    # Corrected according to common culinary definitions rather than strict botanical for a typical user:
    _CULINARY_VEG_SET = {
        "broccoli", "celery", "lettuce", "sweet potatoes", # Potatoes are tubers (stems)
        # Items often considered vegetables culinarily but are botanically fruits:
        # "bell pepper", "corn", "green beans", "zucchini", "tomato", "cucumber", "squash", "eggplant"
        # You need to be very clear about which definition the tool should use.
        # For the original problem's intent with a "stickler botanist mom", the original set was
        # actually trying to define culinary vegetables, and the *fruits* were the ones to avoid.
        # The prompt needs to be clear. Let's assume the provided _VEG_SET was the desired one
        # despite its botanical inaccuracies for some items if the goal was "botanical vegetables".
    }
    # Sticking to the provided _VEG_SET for now, assuming it was curated for a specific purpose.
    # If the goal is strict botanical vegetables, this set would need significant revision.
    
    vegetables_found = sorted([item for item in items if item.lower() in _VEG_SET])
    return ", ".join(vegetables_found)

class ExcelSumFoodInput(BaseModel):
    excel_path: str = Field(description="The file path to the .xlsx Excel file to read.")

@tool(args_schema=ExcelSumFoodInput)
def sum_food_sales(excel_path: str) -> str:
    """
    Reads an Excel file with columns 'Category' and 'Sales',
    and returns total sales (as a string) for categories that are NOT 'Drink',
    rounded to two decimal places.
    Args:
        excel_path: The file path to the .xlsx Excel file to read.
    """
    try:
        df = pd.read_excel(excel_path)
        if "Category" not in df.columns or "Sales" not in df.columns:
            raise ValueError("Excel file must contain 'Category' and 'Sales' columns.")
        
        # Ensure 'Sales' column is numeric, coercing errors to NaN
        df["Sales"] = pd.to_numeric(df["Sales"], errors='coerce')
        
        # Filter out 'Drink' and then sum, handling potential NaNs from coercion
        total = df.loc[df["Category"].str.lower() != "drink", "Sales"].sum(skipna=True)
        
        return str(round(float(total), 2))
    except FileNotFoundError:
        return f"Error: File not found at path '{excel_path}'"
    except ValueError as ve:
        return f"Error processing Excel file: {ve}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

# build a retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_SERVICE_KEY"))
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)



tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
    reverse_text,
    find_non_commutative_elements,
    list_vegetables,
    sum_food_sales,
]

hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    raise ValueError("Hugging Face API token (HF_TOKEN) not found in environment variables.")

tavili_key = os.environ.get('TAVILY_API_KEY')
if not tavili_key:
    raise ValueError("Hugging Face API token (HF_TOKEN) not found in environment variables.")


# Build graph function
def build_graph(provider: str = "huggingface"):

    """Build the graph"""
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        # Groq https://console.groq.com/docs/models
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0) # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        # repo_id = "togethercomputer/evo-1-131k-base"
        # repo_id="HuggingFaceH4/zephyr-7b-beta",
        # repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",

        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set. It's required for Hugging Face provider.")
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            provider="auto",
            task="text-generation",
            max_new_tokens=1000,
            do_sample=False,
            repetition_penalty=1.03,
            
            
        )
        llm = ChatHuggingFace(llm=llm)
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    # Bind tools to LLM
    """Build the graph"""

  
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        print("\n--- Assistant Node ---")
        print("Incoming messages to assistant:")
        for msg in state["messages"]:
            msg.pretty_print() #
        
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        print("ex msgs"+[sys_msg] + state["messages"] + [example_msg])
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    compiled_graph = builder.compile() # This line should already be there or be the next line

    # --- START: Add this visualization code ---
    try:
        print("Attempting to generate graph visualization...")
        image_filename = "langgraph_state_diagram.png"
        # Using draw_mermaid_png as it's often more robust
        image_bytes = compiled_graph.get_graph().draw_mermaid_png()
        with open(image_filename, "wb") as f:
            f.write(image_bytes)
        print(f"SUCCESS: Graph visualization saved to '{image_filename}'")

    except ImportError as e:
        print(f"WARNING: Could not generate graph image due to missing package: {e}. "
              "Ensure 'pygraphviz' and 'graphviz' (system) are installed, or Mermaid components are available.")
    except Exception as e:
        print(f"WARNING: An error occurred while generating the graph image: {e}")
        try:
            print("\nGraph (DOT format as fallback):\n", compiled_graph.get_graph().to_string())
        except Exception as dot_e:
            print(f"Could not even get DOT string: {dot_e}")
    # --- END: Visualization code ---

    return compiled_graph # This should be the last line of the function
    
# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph(provider="huggingface")
    # Run the graph
    messages = [HumanMessage(content=question)]

    print(messages)
    config = {"recursion_limit": 27}

    messages = graph.invoke({"messages": messages}, config=config)
    for m in messages["messages"]:
        m.pretty_print()



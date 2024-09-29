from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

def create_llm(model_name: str = "gpt-3.5-turbo"):
    """Creates an LLM instance with the specified model."""
    return ChatOpenAI(model=model_name)

def get_map_chain(llm):
    """Creates the map chain for summarization."""
    map_prompt = hub.pull("rlm/map-prompt")
    return map_prompt | llm | StrOutputParser()

def get_reduce_chain(llm):
    """Creates the reduce chain for summarization."""
    reduce_template = """
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary
    of the main themes.
    """
    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    return reduce_prompt | llm | StrOutputParser()

def length_function(documents, llm):
    """Calculates the total number of tokens in the given documents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

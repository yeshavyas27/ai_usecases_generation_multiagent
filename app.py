import streamlit as st
import os
import uuid
import sqlite3
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

# Initialize Tavily client
if 'TAVILY_API_KEY' not in os.environ:
    st.error("TAVILY_API_KEY environment variable not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()
tavily = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])

# Define Pydantic models
class Queries(BaseModel):
  queries: List[str]

class Sources(BaseModel):
    data: str
    url: str
    id: int

class CitedUseCase(BaseModel):
    answer: str = Field(..., description="Detailed use case description")
    citations: List[int] = Field(..., description="Source IDs")

class CitedUseCaseWithAssets(CitedUseCase):
    assets: List[str]

class UseCases(BaseModel):
    usecases: List[CitedUseCase]

# Define AgentState
class AgentState(TypedDict):
    input_company_industry: str
    content: List[Sources]
    draft: List[CitedUseCaseWithAssets]
    source_id_start: int

# Initialize Mistral model
model = ChatMistralAI(model="mistral-small-latest", temperature=0)


RESEARCH_INDUSTRY_PROMPT = """
You are an expert at identifying relevant industries given a company name or industry name.
Generate a list of search queries that will gather key focus areas for the company or industry (e.g., operations, supply chain, customer experience, etc.)
Different products related to the industry or company should also be included.
Generate a list of search queries that will gather any relevant information.
Only generate 3 queries max.

"""

RESEARCH_AI_USECASES_PROMPT = """
You are an expert in finding loopholes in various industries which can be resolved by Artificial Intelligence (AI), Machine Learning (ML) and Automation.
Refer to reports and insights on AI and digital transformation from industry-specific sources to find potential solutions.
It could improve efficiency, solve problems and innvoate. Generate a list of search queries that will gather any relevant information given the data. A
Only generate 5 queries max.

"""

WRITER_PROMPT = """
You are an AI expert who can identify areas usecases of Artificial Intelligence (AI), Machine Learning (ML) and Automation in a given company/industry.
You are given data on possible usecases in a specific industry. Identify the best top 3 usecases and provide details for each including the problem and it's solution using AI, ML or automation. Do not explicitly use the word problem and solution. Write a paragraph for each usecase.
"""

RESEARCH_DATASETS_FOR_USECASES_PROMPT = """
You are an expert at searching relevant datasets/ databases or relevant projects for a solving problem or usecase, which can be used for AI, ML and automation.
Generate a list of search queries that will gather any relevant information to this.
Only generate 3 queries max.
"""



def research_and_identify_industry_node(state: AgentState):
  queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_INDUSTRY_PROMPT),
        HumanMessage(content=f"Here is the company/industry: {state['input_company_industry']}.")
        ])
  content =  state.get('content', [])
  source_id_start = state['source_id_start']
  for q in queries.queries:
      response = tavily.search(query=q, max_results=2)
      for r in response['results']:
        source_id_start = source_id_start + 1
        content.append(Sources(data=r['content'], url=r['url'], id=source_id_start))

  return {"content": content}

def research_relevant_ai_usecases_node(state: AgentState):
  queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_AI_USECASES_PROMPT),
        HumanMessage(content=f"Here is company/industry: {state['input_company_industry']}. Some relevant data to the industry: {' '.join([source.data for source in state['content']])}")
        ])
  content = state['content']
  source_id_start = content[-1].id
  for q in queries.queries:
      response = tavily.search(query=q, max_results=2)
      for r in response['results']:
        source_id_start = source_id_start + 1
        content.append(Sources(data=r['content'], url=r['url'], id=source_id_start))

  return {"content": content}

def generate_draft_node(state: AgentState):
  usecases = model.with_structured_output(UseCases).invoke([
        SystemMessage(content=WRITER_PROMPT),
        HumanMessage(content=f"Here is company/industry: {state['input_company_industry']}. Some relevant data to the industry and possible usecases: {' '.join([f' Source: {source.id} Information:{source.data} ' for source in state['content']])}")
        ])
  return {"draft": usecases}

def research_assets_for_usecases_node(state: AgentState):
  # loop through the list and find relevant datasets
  content = state['content']
  draft = state['draft']

  usecases_with_assets = []
  for usecase in draft.usecases:
    queries = model.with_structured_output(Queries).invoke([
          SystemMessage(content=RESEARCH_DATASETS_FOR_USECASES_PROMPT),
          HumanMessage(content=f"The problem domain or usecase that I am trying to solve for is {usecase.answer}")
          ])
    urls = []

    for q in queries.queries:
        response = tavily.search(query=q, max_results=2, include_domains=["kaggle.com/datasets", "huggingface.co/datasets", "github.com"])
        for r in response['results']:
          urls.append(r['url'])
    usecases_with_assets.append(
        CitedUseCaseWithAssets(
            answer=usecase.answer,
            citations=usecase.citations,
            assets=urls
        )
    )

  return {"draft": usecases_with_assets}

# Set up LangGraph with caching
@st.cache_resource
def setup_graph():
    conn = sqlite3.connect('checkpoints.db', check_same_thread=False)
    memory = SqliteSaver(conn)
    
    builder = StateGraph(AgentState)
    builder.add_node("research_industry", research_and_identify_industry_node)
    builder.add_node("research_ai_usecases", research_relevant_ai_usecases_node)
    builder.add_node("generate_draft", generate_draft_node)
    builder.add_node("research_assets", research_assets_for_usecases_node)
    
    builder.set_entry_point("research_industry")
    builder.add_edge("research_industry", "research_ai_usecases")
    builder.add_edge("research_ai_usecases", "generate_draft")
    builder.add_edge("generate_draft", "research_assets")
    builder.add_edge("research_assets", END)
    
    return builder.compile(checkpointer=memory)

# Streamlit UI
st.set_page_config(page_title="AI Solution Explorer", layout="wide")
st.title("🔍 AI Solution Explorer")
st.markdown("Identify AI/ML opportunities for any industry or company")

with st.form("main_form"):
    industry = st.text_input("Enter a company/industry:", 
                           placeholder="E.g., 'Healthcare' or 'Tesla'")
    submitted = st.form_submit_button("Analyze")

if submitted and industry:
    # Initialize graph and state
    graph = setup_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    with st.status("Researching...", expanded=True) as status:
        # Execute the graph
        st.write("🔍 Analyzing industry landscape...")
        graph.invoke({
            'input_company_industry': industry,
            'source_id_start': 0
        }, config)
        
        # Get final state
        state = graph.get_state(config)
        content = state.values.get("content", [])
        draft = state.values.get("draft", [])
        
        status.update(label="Analysis complete!", state="complete")
    
    # Display results
    if draft:
        st.subheader("🚀 Identified AI Opportunities")
        for idx, usecase in enumerate(draft, 1):
            with st.expander(f"Opportunity #{idx}: {usecase.answer[:80]}..."):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Solution Overview**\n{usecase.answer}")
                
                with cols[1]:
                    st.markdown("**References**")
                    for cit in usecase.citations:
                        source = next((s for s in content if s.id == cit), None)
                        if source:
                            st.markdown(f"- [Source {cit}]({source.url})")
                    
                    if usecase.assets:
                        st.markdown("**Recommended Datasets/Assets**")
                        for asset in usecase.assets:
                            st.markdown(f"- [Dataset]({asset})")
    else:
        st.warning("No use cases found. Try a different input.")
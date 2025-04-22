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
You are an expert in market and industry research.

Given a company or industry name, generate 3 highly specific and relevant search queries that will help uncover:
- The company/industry’s key focus areas (e.g., operations, innovation, supply chain, customer experience)
- Products or services related to the company/industry

Guidelines:
- Use domain-specific language
- Avoid generic phrases
- Include diverse business angles if possible

Example (for Tesla):
1. Tesla energy storage solutions 2024 overview
2. Tesla customer satisfaction in EV sector
3. Tesla gigafactory supply chain logistics

Now, generate 3 search queries for the given input.


"""

RESEARCH_AI_USECASES_PROMPT = """
You are an expert in identifying inefficiencies, gaps, or untapped opportunities across industries that can be addressed using Artificial Intelligence (AI), Machine Learning (ML), or Automation.

Your task is to generate a list of 5 precise and relevant search queries that will help uncover:
- Industry-specific pain points or inefficiencies
- Opportunities for AI/ML/automation-driven transformation
- Case studies, digital transformation reports, or expert insights

Use domain-relevant language, and ensure queries explore how AI can improve processes, reduce costs, or drive innovation.

Example (for logistics industry):
1. AI in last-mile delivery optimization 2024
2. Warehouse automation trends logistics industry
3. Machine learning for demand forecasting in supply chain
4. Digital transformation case studies logistics 2023
5. Operational bottlenecks solved by AI logistics

Now, generate 5 search queries based on the given industry or business domain.


"""

WRITER_PROMPT = """
You are an expert in Artificial Intelligence (AI), Machine Learning (ML), and Automation, skilled at identifying impactful use cases within specific industries or companies.

You are provided with data describing potential AI/ML/automation applications in a given industry. From this, select the three most promising or transformative use cases.

For each use case, write one clear, well-structured paragraph that:
- Describes the context or challenge being addressed (without using the word "problem")
- Explains how AI, ML, or automation is applied
- Highlights the potential value or outcome

Do not use labels like “Problem” or “Solution.” Just write three coherent paragraphs, one per use case.

"""

RESEARCH_DATASETS_FOR_USECASES_PROMPT = """
You are an expert in finding relevant datasets, databases, and related AI/ML/automation projects that support solving specific problems or implementing use cases.

Your task is to generate up to 3 well-formed search queries that will help discover:
- Datasets suitable for training or evaluating models
- Open-source projects or research initiatives related to the use case
- Domain-specific resources (e.g., finance, healthcare, logistics, etc.)

Use precise technical language and include keywords that reflect the use case’s domain, objective, or data type.

Example (for fraud detection in finance):
1. Financial transaction fraud detection benchmark datasets
2. Open-source projects on anomaly detection in banking
3. Real-world datasets for supervised learning in fraud analytics

Now, generate 3 search queries for the given use case.

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
                        source = next((s for s in content if s.id == cit - 1), None)
                        if source:
                            st.markdown(f"- [Source {cit}]({source.url})")
                    
                    if usecase.assets:
                        st.markdown("**Recommended Datasets/Assets**")
                        for asset in usecase.assets:
                            st.markdown(f"- [Dataset]({asset})")
    else:
        st.warning("No use cases found. Try a different input.")

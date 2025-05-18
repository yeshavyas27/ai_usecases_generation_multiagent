import streamlit as st
import os
import uuid
import sqlite3
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

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
    answer: str = Field(
        ...,
        description="All the data of a usecase, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the usecase.",
    )
    title: str = Field(
        ...,
        description="The title of the usecase.",
    )

class CitedUseCaseWithAssets(CitedUseCase):
    assets: List[str]
    assets_ids: List[int]
    total_score: int | None
    feedback: str | None

class FeedbackScoreUseCase(BaseModel):
  total_score: int
  feedback: str

class UseCases(BaseModel):
    usecases: List[CitedUseCase]

class Industries(BaseModel):
  industries_sectors_and_products: List[str]

# Define AgentState
class AgentState(TypedDict):
    user_input: str
    is_company: bool
    industries: Industries
    assets: List[str]
    draft: UseCases
    content: List[Sources]
    source_id_start: int
    revision: int
    company_industry_context: List[Sources]

# Initialize Gemini model
model =  ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)


RESEARCH_COMPANY_PROMPT = """
Based solely on the provided company data, identify the company's primary industry and sector.
- Use only information explicitly stated in the data; do not infer or assume details not present.
- Be as specific as possible. For example, if the company manufactures automotive parts, state 'automotive parts manufacturing' instead of the broader 'automotive' industry.
- Avoid using general or umbrella terms like 'beauty' or 'technology' if more precise descriptions (e.g., 'cosmetic materials manufacturing' or 'semiconductor equipment') are available in the data.
Reason and think step by step to decide the most relevant industry/ sectors/ products, and generate at most 3 answers. Make sure to use specific terms which are most relevant to what the company does.
"""

RESEARCH_INDUSTRY_PROMPT = """
Based on the data provided about the industry, sector or domain, identify other closely related sub domains. It should be related to the given industry/ domain provided.
Use only information explicitly stated in the data; do not infer or assume details not present.
Find sepecifc areas for example, if 'healthcare' is provided, state 'mental health', 'dermatology', 'emergency and critical care' and so on.
Reason and think step by step to decide the most relevant sub domains, and generate at most 3 answers.
"""

WRITER_PROMPT = """
You are an expert in Artificial Intelligence (AI), Machine Learning (ML), and Automation, skilled at identifying impactful use cases within specific industries and domains.

You are provided with data describing potential AI/ML/automation applications in a given industry.
From this, select the most promising or transformative use cases. Each usecase should be unique as compared to the other ones.

For each use case, write one clear, well-structured paragraph that:
- Describes the context or challenge being addressed (without using the word "problem")
- Explains how AI, ML, or automation is applied
- Highlights the potential value or outcome

Write one paragraph per use case. Reason and only pick the usecases which are most relevant and unique to the industry provided.

"""

REVIEWER_PROMPT = """
You are an expert reviewer who rates the relevancy of an ai usecase to the industry,it's sources and it's feasiblity given information about assets. You are given:

- `AI Usecase`: A proposed AI use case for a specific industry.
- `References Content`: Background content from which this use case may have been derived.
- `Assets Content`: A collection of web-scraped datasets and open-source AI/ML projects.
- `Related Industries`: The target industries for the AI use case.

Your task is to **critically assess** the use case across **three dimensions**:

1. **Industry Relevance (Score out of 10)**
   Does the use case directly and meaningfully address real challenges or opportunities in the target industry/industries?

2. **Source Relatedness (Score out of 5)**
   How well is the use case grounded in the provided reference materials and industry context?

3. **Feasibility via Assets (Score out of 5)**
   How realistically can this use case be implemented using the provided assets (datasets, models, APIs, etc.)?

Provide scores for each dimension, a brief justification, and calculate the **total score out of 20**. Then, offer **critical feedback and suggestions for improvement**.

---

**Example 1: High-Score Use Case (18/20)**
Related Industries: Agriculture, Food Safety, Environmental Science
AI Use Case:
Predicting crop contamination from soil sensors and satellite imagery using CNN models. Detects early signs of heavy metal absorption in leafy greens, alerts farmers pre-harvest, and integrates with irrigation systems to flush toxins.

Reference Content:
- FAO 2024 report on $12B annual losses from crop contamination
- EU Regulation 2023/1945 mandating real-time soil monitoring
- Case study: Walmart's blockchain system for produce traceability

Assets Content:
- USDA's 10TB OpenAgData repository
- Pretrained ResNet-50 on PlantVillage dataset
- AgriOS farm management API

1. Industry Relevance: 9/10
   Strongly aligned with food safety and environmental sustainability. Tangible ROI and regulatory impact.

2. Source Relatedness: 4/5
   Well-supported by regulatory and economic sources, though blockchain link could be stronger.

3. Feasibility via Assets: 5/5
   All components are available in the open-source ecosystem.

**Total Score: 18/20**
**Feedback:** Add traceability integration using blockchain frameworks. Consider deploying through AgriOS-compatible IoT devices.

---

**Example 2: Medium-Score Use Case (12/20)**
Related Industries: Retail, Urban Planning, Transportation
AI Use Case:
Optimizing last-mile delivery by analyzing social media posts to predict package receipt likelihood.

Reference Content:
- McKinsey report on $18B losses in last-mile logistics
- Singapore's Smart City Mobility Index
- FTC guidelines on data privacy

Assets Content:
- Geotagged Twitter dataset (1M posts)
- OSRM routing engine
- Anonymized UPS logs (2018–2020)

1. Industry Relevance: 6/10
   Addresses a real cost center, but approach is speculative and privacy-sensitive.

2. Source Relatedness: 3/5
   General logistics pain point covered; weak link to social signals.

3. Feasibility via Assets: 3/5
   Data and tools exist, but real-time NLP pipelines are unproven for this task.

**Total Score: 12/20**
**Feedback:** Validate signal strength from social media. Start with offline pilots. Explore alternative customer behavior signals.

---

**Example 3: Low-Score Use Case (7/20)**
Related Industries: Aerospace, Automotive, Telecommunications
AI Use Case:
Using quantum reinforcement learning for autonomous drone swarms in 6G tower maintenance.

Reference Content:
- Ericsson 6G whitepaper
- FAA drone regulations (2025)
- Tesla Bot v2 specs

Assets Content:
- OpenStreetMap 3D maps
- Qiskit quantum framework
- DJI Drone SDK

1. Industry Relevance: 3/10
   Highly futuristic, minimal current industry alignment or readiness.

2. Source Relatedness: 2/5
   References speak to general trends, not the specific use case.

3. Feasibility via Assets: 2/5
   Quantum RL is highly experimental; real-world readiness is low.

**Total Score: 7/20**
**Feedback:** Scope down to classical RL with single drones. Seek telecom partnerships for gradual validation.

"""

RESEARCH_BASED_ON_FEEDBACK = """
Generate search queries to gather information to improve the ai usecase for a given industry based on the feedback provided. Reason step by step and generate at most search queries.

"""
IMPROVE_USECASE_PROMPT = """
You are an expert in Artificial Intelligence (AI), Machine Learning (ML), and Automation with deep knowledge of identifying high-impact, domain-specific use cases tailored to particular industries.

You will be provided with:
- Existing AI use cases (to avoid duplication)
- Reviewer feedback on how to improve or evolve those ideas
- Target industries for relevance
- Background content suggesting potential AI opportunities

Your task is to generate a **new, unique AI use case** that:
- Is clearly distinct from all existing use cases
- Is highly relevant to the specified industries
- Draws only from the given information and context

In your response, write **one well-structured paragraph** that:
- Introduces the industry context or challenge (do not use the word “problem”)
- Describes how AI, ML, or automation is applied
- Articulates the potential impact or value delivered

"""


import re

def extract_domain(url):
    pattern = r'^(?:https?:\/\/)?(?:www\.)?([^\/:?#]+)'
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    return None

def is_company(state: AgentState):
  if state["is_company"]:
    return "research_about_company"
  else:
    return "research_about_industry"

def research_company_node(state: AgentState):

  content =  []
  source_id_start = state['source_id_start']
  company_name = state["user_input"]

  queries = [company_name, f"What industry does {company_name} work in", f"What are the product offerings of {company_name}"]
  for q in queries:
    response = tavily.search(query=q, max_results=2)
    for r in response['results']:
      source_id_start = source_id_start + 1
      content.append(Sources(data=r['content'], url=r['url'], id=source_id_start))

  industries = model.with_structured_output(Industries).invoke([
      SystemMessage(content=RESEARCH_COMPANY_PROMPT),
      HumanMessage(content=f"Company data/ information: {' '.join([source.data for source in content])}")
      ])

  return {"industries": industries, "company_industry_context": content}

def research_industry_node(state: AgentState):
  content =  state.get('content', [])
  source_id_start = state['source_id_start']
  industries = state["user_input"]

  queries = [f"{user_input} industry", f"Product offerings in {user_input} industry", f"Trends and Sub domains in {user_input} industry"]
  for q in queries:
    response = tavily.search(query=q, max_results=2)
    for r in response['results']:
      source_id_start = source_id_start + 1
      content.append(Sources(data=r['content'], url=r['url'], id=source_id_start))

  industries = model.with_structured_output(Industries).invoke([
      SystemMessage(content=RESEARCH_INDUSTRY_PROMPT),
      HumanMessage(content=f"Industry/ Sector data: {' '.join([source.data for source in content])}")
      ])

  return {"industries": industries, "company_industry_context": content}


def research_and_generate_ai_usecases(state: AgentState):

  content = state.get('content', [])
  source_id_start = state['source_id_start']
  industries = state["industries"]
  draft = state.get('draft', None)
  company_industry_context = state.get('company_industry_context', None)

  queries = []
  for industry in industries.industries_sectors_and_products:
    queries.append(f"products using AI(artififcial intelligence), ML(machine learning) in {industry} ")
    queries.append(f"AI(artificial intelligence), ML(machine learning) trends in {industry} ")
    queries.append(f"AI for {industry} site:springer.com")
    queries.append(f"AI driven innovation in {industry}")

  for q in queries:
    response = tavily.search(query=q, max_results=3)
    for r in response['results']:
      source_id_start = source_id_start + 1
      content.append(Sources(data=r['content'], url=r['url'], id=source_id_start))

  usecases = model.with_structured_output(UseCases).invoke([
        SystemMessage(content=WRITER_PROMPT),
        HumanMessage(content=f"Here are the related industries/ domains/product offerings: {industries} and context for industries: {' '.join([source.data for source in company_industry_context])}. Relevant data of the industry and possible usecases: {' '.join([f' Source: {source.id} Information:{source.data} ' for source in content])}")
        ])
  return {"content": content, "draft": usecases}


def research_assets_for_usecases_node(state: AgentState):
  # loop through the list and find relevant datasets
  content = state['content']
  draft = state['draft']
  source_id_start = content[-1].id
  usecases_with_assets = []
  for usecase in draft.usecases:
    query = usecase.title
    urls = []

    response = tavily.search(query=query, max_results=3, include_domains=["kaggle.com/datasets", "huggingface.co/datasets", "github.com"])
    assets_ids = []
    for r in response['results']:
      urls.append(r['url'])
      source_id_start = source_id_start + 1
      content.append(Sources(data=r['content'], url=r['url'], id=source_id_start))
      assets_ids.append(source_id_start)

    usecases_with_assets.append(
        CitedUseCaseWithAssets(
            answer=usecase.answer,
            citations=usecase.citations,
            assets=urls,
            title=usecase.title,
            assets_ids=assets_ids,
            total_score=None,
            feedback=None
        )
    )

  return {"draft": usecases_with_assets}

def review_usecases_node(state: AgentState):
  # usecases below 12 need to be improved
  # check for all good, if yes then end, else go to improve
  draft = state['draft']
  content = state['content']
  industries = state['industries']
  company_industry_context = state['company_industry_context']

  for i in range(len(draft)):
    usecase = draft[i]
    references_content = [content[citation_id - 1].data for citation_id in usecase.citations]
    ai_usecase = usecase.answer
    assets_content = [content[asset_id - 1].data for asset_id in usecase.assets_ids]
    feedback_score = model.with_structured_output(FeedbackScoreUseCase).invoke([SystemMessage(content=REVIEWER_PROMPT), HumanMessage(content=f"AI Usecase: {ai_usecase} \n References Content: {references_content} \n Assets Content: {assets_content} \n Related Industries: {industries} and context for industries: {' '.join([source.data for source in company_industry_context])}")])
    usecase.total_score = feedback_score.total_score
    usecase.feedback = feedback_score.feedback

  return {"draft": draft}



def improve_usecases_node(state: AgentState):
  # replace the usecases below 12, and then send control back to rate usecases
  # search for relevant information based on the critique given, search for more  sources, generate new draft, and then find new assets
  revision = state["revision"] + 1
  draft = state['draft']
  content = state['content']
  industries = state['industries']
  source_id_start = content[-1].id
  company_industry_context = state['company_industry_context']

  for i in range(len(draft)):
    if draft[i].total_score < 14:
      remove_usecase = draft[i]
      # generate search queries to find relevant content to improve the given usecase based on the feedback provided
      queries = []
      for industry in industries.industries_sectors_and_products:
        queries.append(f"{industry} AI innovation market report")
        queries.append(f"robotics and automation for {industry}")
        queries.append(f"generative ai (genai) for {industry}")

      old_sources = [extract_domain(source.url) for source in content]
      for q in queries:
        response = tavily.search(query=q, max_results=2, exclude_domains=old_sources)
        for r in response['results']:
          source_id_start = source_id_start + 1
          source_obj = Sources(data=r['content'], url=r['url'], id=source_id_start)
          content.append(source_obj)
      # given the existing usecases which is the intitial draft, improve the score by suggesting a new unique idea relevant to the industry (make sure the ideas do not overlap)
      # update the usecase
      updated_usecase = model.with_structured_output(CitedUseCase).invoke([SystemMessage(content=IMPROVE_USECASE_PROMPT), HumanMessage(content=f"Exsting AI Usecases: {','.join([usecase.title for usecase in draft if usecase.total_score >= 12])} \n Related Industries: {industries} and context for industries: {''.join([source.data for source in company_industry_context])}\n Feedback: {remove_usecase.feedback} \n Reference data: {' '.join([f' Source: {source.id} Information:{source.data} ' for source in content])}")])
      if updated_usecase:
        remove_usecase.answer = updated_usecase.answer
        remove_usecase.citations = updated_usecase.citations
        remove_usecase.title = updated_usecase.title
        remove_usecase.total_score = 14 # dummy number so the newly generated one is also considered while generating the other new one

        # find new assets related to title
        query = remove_usecase.title
        urls = []

        response = tavily.search(query=query, max_results=3, include_domains=["kaggle.com/datasets", "huggingface.co/datasets", "github.com"])
        assets_ids = []
        for r in response['results']:
          urls.append(r['url'])
          source_id_start = source_id_start + 1
          content.append(Sources(data=r['content'], url=r['url'], id=source_id_start))
          assets_ids.append(source_id_start)

        remove_usecase.assets = urls
        remove_usecase.assets_ids = assets_ids
      else:
        draft.pop(i)

  return {"draft": draft, "revision": revision, "content": content}

def is_all_good(state: AgentState):
  # end if no usecases with score below 14, else go to improve usecases
  if state["revision"] > 5:
    return "stop"
  draft = state['draft']
  for usecase in draft:
    if usecase.total_score < 14:
      return "continue"
  return "stop"


# Set up LangGraph with caching
@st.cache_resource
def setup_graph():
    conn = sqlite3.connect('checkpoints.db', check_same_thread=False)
    memory = SqliteSaver(conn)
    
    builder = StateGraph(AgentState)
    builder.add_node("research_company",research_company_node)
    builder.add_node("research_industry",research_industry_node)
    builder.add_node("research_and_generate_ai_usecases",research_and_generate_ai_usecases)
    builder.add_node("research_assets_for_usecases",research_assets_for_usecases_node)
    builder.add_node("review_usecases",review_usecases_node)
    builder.add_node("improve_usecases",improve_usecases_node)
        
    builder.add_conditional_edges(START, is_company, {"research_about_company": "research_company", "research_about_industry": "research_industry"})
    builder.add_edge("research_company", "research_and_generate_ai_usecases")
    builder.add_edge("research_industry", "research_and_generate_ai_usecases")
    builder.add_edge("research_and_generate_ai_usecases", "research_assets_for_usecases")
    builder.add_edge("research_assets_for_usecases", "review_usecases")
    builder.add_conditional_edges("review_usecases", is_all_good, {"continue": "improve_usecases", "stop": END})
    builder.add_edge("improve_usecases","review_usecases" )
    
    return builder.compile(checkpointer=memory)
mapper = {"Industry": False, "Company": True}
# Streamlit UI
st.set_page_config(page_title="AI Usecases Researcher", layout="wide")
st.title("🔍 AI Applications Explorer")
st.markdown("Identify AI/ML opportunities for any industry or company")

with st.form("main_form"):
    user_input = st.text_input("Enter a company/industry:", 
                           placeholder="E.g., 'Healthcare' or 'Tesla'")
    input_type = st.selectbox("Is this a company or an industry?", 
                              options=["Industry", "Company"])
    st.write(f"The user chose {input_type}")
    submitted = st.form_submit_button("Analyze")

if submitted and user_input:
    # Initialize graph and state
    graph = setup_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    with st.status("Researching...", expanded=True) as status:
        # Execute the graph
        st.write("🔍 Analyzing industry landscape...")
        graph.invoke({
            'user_input': user_input,
            "is_company": mapper[input_type],
            "source_id_start": 0,
            "revision": 1

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
            with st.expander(f"UseCase #{idx}: {usecase.title} (Score: {usecase.total_score})"):
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

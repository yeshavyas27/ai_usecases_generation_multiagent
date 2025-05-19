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
You are an expert reviewer who rates the relevancy of an AI usecase to the industry, its sources, and its feasibility given information about assets. You are given:

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

## EXAMPLE 1: LOW SCORE (Specialty Food Manufacturing)

AI Usecase: An AI system that analyzes social media trends to predict future flavor preferences for artisanal cheese production, automatically adjusting milk fermentation parameters and aging conditions to optimize for anticipated consumer taste preferences six months in advance.

References Content: 
The specialty cheese market is growing at 3.5% annually, with increasing consumer interest in artisanal products. Traditional cheese production relies on master cheesemakers who make decisions based on experience and sensory evaluation. Temperature control systems for cheese aging have seen technological improvements, with IoT sensors becoming more common in aging caves. Some large dairy processors are exploring data analytics for quality control in commodity cheese production.

Assets Content: 
OpenFoodFacts dataset containing nutritional information for packaged foods; TensorFlow implementation of sentiment analysis for product reviews; GitHub repository with basic time series forecasting models for agricultural commodities.

Related Industries: 'specialty food manufacturing', 'artisanal cheese production', 'dairy processing', 'food technology'

EVALUATION:

Industry Relevance: 5/10
While flavor innovation and production optimization are relevant to artisanal cheese manufacturing, the proposed predictive social media analysis for automated fermentation control misunderstands fundamental aspects of the industry. Artisanal cheese production values traditional methods and cheesemaker expertise. The six-month prediction timeframe is problematic given cheese aging varies from weeks to years, and the direct correlation between social media trends and successful cheese flavor profiles is unproven in this craft-focused segment.

Source Relatedness: 2/5
The references mention growth in the specialty cheese market and some technology adoption for temperature monitoring, but nothing supports the core premise of social media trend analysis for cheese production. The references actually emphasize traditional production methods and the role of master cheesemakers, contradicting the fully automated approach proposed.

Feasibility via Assets: 1/5
The provided assets are severely misaligned with the proposed use case. OpenFoodFacts contains nutritional data but not production parameters or flavor profiles. The sentiment analysis model isn't designed for predictive flavor trends. The time series forecasting for agricultural commodities focuses on pricing/supply predictions, not consumer taste preferences. None of the assets address fermentation parameter control systems.

Total Score: 8/20

Feedback: This use case demonstrates a fundamental misalignment with industry realities and available assets. Artisanal cheese production is inherently craft-based, and the value proposition centers on traditional methods and expert knowledge. A more feasible approach would be an AI system that supports (rather than replaces) master cheesemakers by providing insights on optimal aging conditions based on historical production data, or quality assessment tools that complement human expertise. The proposed assets are insufficient for implementation, requiring significant additional data collection specific to cheese production parameters and flavor development during aging.

## EXAMPLE 2: MEDIUM SCORE (Precision Agricultural Equipment Manufacturing)

AI Usecase: A computer vision system integrated into harvesting equipment that identifies crop diseases in real-time during harvest operations, providing immediate alerts to equipment operators and generating field maps of disease incidence. The system also adapts harvesting parameters (cutting height, speed) to maximize yield when processing affected areas, and tags potentially contaminated harvest batches for separate processing.

References Content:
Global precision agriculture equipment market projected to reach $12.8 billion by 2025, with smart equipment adoption growing at 14.2% annually. John Deere has invested $305 million in computer vision and machine learning for agricultural applications since 2017. Studies show early disease detection can reduce crop losses by 30-60%. Agricultural equipment manufacturers face increasing pressure to demonstrate sustainability metrics and reduce pesticide usage through targeted interventions. Computer vision has achieved 87-92% accuracy in identifying common crop diseases in controlled testing environments.

Assets Content:
PlantDisease-Net: An open-source dataset containing 50,000 labeled images of 38 crop diseases across 12 plant species with varied lighting and growth stage conditions. AgEquipment-SDK: A software development kit for interfacing with major agricultural equipment control systems, including APIs for adjusting harvester operational parameters. HarvestOptimizer: A reinforcement learning framework for optimizing harvesting operations based on crop conditions and equipment capabilities.

Related Industries: 'precision agricultural equipment manufacturing', 'smart farming solutions', 'agricultural technology', 'harvest automation'

EVALUATION:

Industry Relevance: 8/10
The use case directly addresses significant industry challenges in precision agriculture equipment manufacturing. Real-time disease detection during harvest has clear economic value through reduced crop losses, improved processing decisions, and enhanced sustainability metrics. The integration of detection with automated harvester parameter adjustment aligns with market trends toward smarter equipment. However, it lacks consideration of the economic trade-offs of implementation costs versus benefits for specific crop types.

Source Relatedness: 4/5
The use case is well-grounded in the provided references, which confirm substantial investment in computer vision technology by major manufacturers, quantify the potential benefits of early disease detection, and establish technical feasibility with specific accuracy metrics. The references directly support the market demand and technological trajectory described in the use case.

Feasibility via Assets: 3/5
The PlantDisease-Net dataset provides a strong foundation for the computer vision component, though domain adaptation would be needed for real-time, in-field conditions during harvest. The AgEquipment-SDK offers practical paths to implementation for harvester parameter control. However, the HarvestOptimizer framework would require significant modification to incorporate disease detection as an input parameter for operational adjustments, and the system lacks assets addressing the segregation and tracking of potentially contaminated batches.

Total Score: 15/20

Feedback: This use case demonstrates strong alignment with industry needs and available references, with moderate feasibility given the assets. To improve, consider adding specific ROI calculations for different crop types and equipment scales to strengthen the business case. The technical implementation would benefit from addressing environmental variability (dust, lighting conditions during harvest) that could affect computer vision performance. Consider expanding the scope to include post-harvest data integration with farm management systems for long-term disease management strategies. Additional assets related to batch tracking and contamination management would strengthen the implementation pathway.

## EXAMPLE 3: HIGH SCORE (Clinical Trial Management Software)

AI Usecase: An AI-powered predictive analytics platform for clinical trial management that uses natural language processing to continuously analyze patient-reported outcomes, clinical notes, and physiological monitoring data to identify early warning signs of adverse events or trial protocol deviations. The system automatically stratifies risk levels, prioritizes patients for intervention, suggests protocol adjustments to clinical teams, and optimizes site monitoring resource allocation based on predicted risk factors.

References Content:
According to industry reports, 85% of clinical trials fail to retain enough patients, with adverse events being a primary reason for dropout. Patient monitoring in decentralized clinical trials increased by 50% during 2020-2022, generating 3x more data points per patient than traditional site-based trials. NLP techniques have demonstrated 92% accuracy in extracting adverse event information from unstructured clinical notes in controlled studies. Regulatory bodies including FDA and EMA have issued guidance supporting risk-based monitoring approaches that leverage predictive analytics. A meta-analysis of 230 failed clinical trials found that 40% could have been salvaged with earlier identification of protocol issues or patient risks.

Assets Content:
MIMIC-IV-NLP: A de-identified clinical database containing 350,000+ patient notes with annotated adverse event mentions, medication effects, and symptom progressions from multiple therapeutic areas. ClinicalBERT: A domain-specific language model pre-trained on 2 million clinical documents with specific fine-tuning for adverse event recognition and medical entity extraction. TrialRisk-ML: An open-source machine learning framework specifically designed for time-series analysis of patient monitoring data with implementations of early warning prediction algorithms validated in oncology and cardiovascular trials. REDCap-Connect: API infrastructure for secure, compliant integration with common clinical trial electronic data capture systems.

Related Industries: 'clinical trial management software', 'pharmaceutical research technology', 'healthcare informatics', 'drug development platforms', 'regulatory technology for life sciences'

EVALUATION:

Industry Relevance: 10/10
This use case addresses critical pain points in clinical trial management with significant financial and patient safety implications. The focus on early detection of adverse events directly targets the 85% failure rate in patient retention noted in the references. The multi-modal approach combining patient-reported outcomes, clinical notes, and monitoring data aligns with the industry trend toward decentralized trials generating diverse data types. The risk stratification and resource optimization components address operational efficiency challenges faced by clinical trial managers working across multiple sites.

Source Relatedness: 5/5
The use case is exceptionally well-grounded in the provided references. It directly applies the demonstrated 92% NLP accuracy for adverse event extraction, responds to the regulatory guidance supporting risk-based monitoring, and addresses the documented 40% of failed trials that could benefit from earlier risk identification. The approach leverages the 3x increase in patient data points noted in the references to enable the proposed predictive capabilities.

Feasibility via Assets: 4/5
The combination of assets forms a comprehensive foundation for implementation. MIMIC-IV-NLP provides the necessary training data for adverse event recognition. ClinicalBERT offers a domain-adapted language model specifically aligned with the NLP requirements. TrialRisk-ML directly supports the time-series analysis needed for monitoring data, with validation in relevant therapeutic areas. REDCap-Connect enables practical integration with existing clinical workflows. The only limitation is the potential need for additional assets addressing regulatory compliance workflows for suggested protocol adjustments.

Total Score: 19/20

Feedback: This use case demonstrates excellent alignment with industry needs, reference materials, and available assets. It addresses a high-value problem with clear ROI through improved patient retention and trial success rates. To further strengthen the implementation pathway, consider expanding the assets to include a regulatory workflow engine for managing the protocol adjustment suggestion process in compliance with GCP requirements. The system could also benefit from incorporating site-specific regulatory requirements for multi-region trials. Consider exploring federated learning approaches to address patient data privacy concerns while maintaining predictive power across trial sites.

## EXAMPLE 4: MEDIUM-HIGH SCORE (Industrial HVAC Systems Manufacturing)

AI Usecase: A predictive maintenance system for commercial HVAC systems that uses acoustic anomaly detection combined with thermal imaging analysis to identify failing components before complete breakdown. The system continuously monitors compressor and air handler sound signatures using edge-deployed deep learning models, correlates acoustic patterns with thermal imagery from periodic maintenance scans, and generates maintenance recommendations with specific component replacement forecasts and energy efficiency impact projections.

References Content:
Unplanned HVAC system downtime costs commercial building operators an average of $5,000-$50,000 per incident depending on facility type. Traditional maintenance schedules miss 30% of developing component failures. Energy consumption increases by 15-30% as components degrade prior to complete failure. Acoustic monitoring has shown 87% accuracy in detecting refrigerant leaks and compressor issues when tested in laboratory settings. Thermal imaging is currently used in 60% of commercial HVAC maintenance processes but typically only during scheduled maintenance visits rather than continuous monitoring.

Assets Content:
HVAC-SoundDB: A dataset containing 120,000 labeled audio samples of normal and abnormal operating sounds from 15 models of commercial HVAC systems across various load conditions. ThermalVisionPro: Computer vision models pre-trained on 50,000 thermal images of mechanical and electrical components with temperature anomaly detection capabilities. EnergyImpactCalculator: An analytics framework for projecting energy consumption impacts of component degradation based on historical performance data. BuildingIoT-Edge: Low-power edge computing framework optimized for continuous sensor monitoring in building systems with pre-built connectors to building management systems.

Related Industries: 'industrial HVAC systems manufacturing', 'commercial building automation', 'facilities management technology', 'predictive maintenance systems'

EVALUATION:

Industry Relevance: 9/10
The use case directly addresses significant pain points in commercial HVAC maintenance with clear financial implications. Preventing unplanned downtime ($5,000-$50,000 per incident) and reducing energy waste (15-30% from component degradation) presents a compelling value proposition. The combination of continuous acoustic monitoring with thermal imaging enhances existing maintenance practices rather than replacing them. The system aligns with industry trends toward condition-based maintenance and energy efficiency optimization in commercial buildings.

Source Relatedness: 4/5
The use case is well-grounded in the provided references, directly addressing the 30% of component failures missed by traditional maintenance and leveraging the demonstrated 87% accuracy of acoustic monitoring. The approach intelligently combines the periodic thermal imaging (currently used in 60% of maintenance processes) with continuous acoustic monitoring to create a more comprehensive solution than either method alone. However, the references don't specifically address the correlation between acoustic and thermal indicators, which is a key aspect of the proposed system.

Feasibility via Assets: 4/5
The assets provide a strong foundation for implementation. HVAC-SoundDB offers comprehensive training data for the acoustic models across various HVAC systems and conditions. ThermalVisionPro directly supports the thermal analysis component. The EnergyImpactCalculator enables the energy efficiency projections that enhance the business case. BuildingIoT-Edge addresses the critical implementation challenge of deploying models to existing building infrastructure. The primary limitation is the lack of assets specifically addressing the fusion of acoustic and thermal data for improved prediction accuracy.

Total Score: 17/20

Feedback: This use case demonstrates strong alignment with industry needs and available assets. The dual-modality approach combining acoustic and thermal analysis represents an innovative advancement beyond current maintenance practices. To strengthen the proposal, consider including validation methodology for the correlation between acoustic anomalies and thermal patterns, as this relationship isn't established in the references. The implementation would benefit from adding a feedback loop that incorporates actual maintenance findings to continuously improve prediction accuracy. Consider expanding the business case to quantify ROI based on facility type and size, since downtime costs vary significantly ($5,000-$50,000).

## EXAMPLE 5: LOW-MEDIUM SCORE (Beauty E-commerce Platform)

AI Usecase: A virtual makeup try-on system for beauty e-commerce that uses generative AI to create photorealistic simulations of how skincare products would affect a customer's skin over time with continued use. Customers upload a selfie and the system generates a timeline of expected results showing progressive improvements in skin texture, tone, and specific conditions like acne or hyperpigmentation at 2-week intervals over a 3-month projected usage period.

References Content:
Beauty e-commerce platforms experience 38% higher cart abandonment rates than other retail categories, with uncertainty about product effectiveness cited as a primary reason. Virtual makeup try-on technologies have increased conversion rates by 30% for color cosmetics but have shown limited impact for skincare products. Consumer surveys indicate 72% of skincare purchasers don't believe product claims about long-term benefits. The skincare segment represents 45% of beauty e-commerce revenue but only 15% of virtual try-on technology implementations. Clinical trials for skincare products typically measure results at 4, 8, and 12-week intervals.

Assets Content:
FaceMesh-AR: A 3D facial mapping technology that creates detailed skin topology maps from selfie images with texture and tone analysis. BeautyGAN: A generative adversarial network trained on before/after images from makeup application but not specifically optimized for skincare effects. ProductClaims-NLP: A natural language processing model that extracts measurable benefit claims from product descriptions. Clinical-Imaging-Small: A limited dataset of 500 before/after images from skincare clinical trials with controlled lighting and standardized photography.

Related Industries: 'beauty e-commerce', 'virtual try-on technology', 'skincare product marketing', 'cosmetics retail platforms'

EVALUATION:

Industry Relevance: 7/10
The use case addresses a significant pain point in beauty e-commerce - the high abandonment rate (38%) specifically linked to uncertainty about skincare product effectiveness. The focus on skincare rather than color cosmetics targets an underserved segment (45% of revenue but only 15% of virtual try-on implementations). The approach directly responds to consumer skepticism about long-term benefits (72% disbelief in claims). However, the use case assumes consumers' primary concern is visualizing long-term results, when immediate texture/feel and potential irritation might be equally important factors in purchase decisions.

Source Relatedness: 3/5
The use case builds on the referenced success of virtual try-on for color cosmetics (30% conversion increase) and attempts to extend this to skincare. The time intervals proposed (2-week increments over 3 months) generally align with the clinical trial measurement points mentioned (4, 8, and 12 weeks). However, the references don't provide evidence that visualizing progressive results would address the specific concerns causing cart abandonment for skincare products.

Feasibility via Assets: 2/5
The assets present significant limitations for implementing this use case. While FaceMesh-AR provides the necessary foundation for skin analysis, BeautyGAN is trained for makeup application rather than progressive skincare effects. The Clinical-Imaging-Small dataset (500 images) is vastly insufficient for training a generative model that can accurately predict personalized skincare results across diverse skin types, conditions, and product interactions. ProductClaims-NLP could help extract benefit timelines but doesn't address the core technical challenge of realistic skin progression visualization.

Total Score: 12/20

Feedback: This use case addresses a valid industry need but faces substantial feasibility challenges with the available assets. The most critical limitation is the insufficient clinical imagery dataset for training a reliable generative model - predictive visualization of skincare benefits requires extensive before/after documentation across diverse skin types and conditions. To improve feasibility, consider pivoting to a hybrid approach that combines actual clinical trial imagery with more modest personalization, rather than fully generative predictions. The business case would be strengthened by validating that visualization of progressive results (rather than immediate effects or ingredient education) would actually increase conversion rates for skincare products. Consider starting with a narrower scope focusing on a specific skin concern (e.g., acne) where before/after patterns might be more consistent and predictable.
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

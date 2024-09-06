#Authenticate
from google.oauth2 import service_account
keyfile = 'tokencreator.json'
#Service Account
credentials = service_account.Credentials.from_service_account_file(keyfile)

# Initialize Vertex AI SDK
import vertexai

PROJECT_ID = "argolis-project-340214"  # @param {type:"string"}

LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials )

from dataclasses import dataclass
from typing import Literal

import streamlit as st
import graphviz
import pandas as pd
from langchain_google_vertexai import VertexAI

from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate


from google.cloud import aiplatform
print(f"Vertex AI SDK version: {aiplatform.__version__}")


# You will need to change these variables
#connectionUrl = "neo4j+s://f28aa3c1.databases.neo4j.io:7687"
#connectionUrl = "neo4j+s://9973068e.databases.neo4j.io"
#connectionUrl = "neo4j+s://3d859915.databases.neo4j.io"
connectionUrl = "neo4j+s://67d25b41.databases.neo4j.io"
username = "neo4j"
#password = "BAZuPrEv4CexTEzucDbJt019MuwfX91kVWv1lRaGNcA"
#password = "5Jh5cGnMJB7tS-UVOPsAyQKov0DcjcoO833b4BuXm0A"
#password = "h0_sQEMslVxhe-bEuB_4uO3vTU4SBBmiiAc7D_5ccFw"


from neo4j import GraphDatabase
driver = GraphDatabase.driver(connectionUrl, auth=(username, password))
driver.verify_connectivity()
def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())

graph = Neo4jGraph(
    url=connectionUrl, username=username, password=password
    )

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
You are a helpful AI assistant for Rocket Mortgage who can respond to any queries asked by employees of Rocket  about Rocket's customers.
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
The provided information is the only information expected of you. So no need to apologize if you do not have all info.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:

# Hello bot
match (a)-[r]->(b) where b.fileName is null and a.fileName is null  
return "No. of relationships in Sarah's customer 360 graph: ", count(*)," - Welcome."

# Where did Sarah build her career?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'BUILT_CAREER_IN'
RETURN b.id

# What is Sarah's occupation?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'HAS_OCCUPATION'
RETURN b.id

# What degree does Sarah hold?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'HOLDS'
RETURN b.id

# What company has Sarah used before?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'USED_COMPANY'
RETURN b.id

# What company does Sarah trust?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'TRUSTED_COMPANY'
RETURN b.id

# What all has Sarah invested in?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'INVESTED_IN'
RETURN b.id

# What has Sarah secured?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'SECURED'
RETURN b.id

# What all has Sarah Purchased?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'PURCHASED'
RETURN b.id

# Which location does Sarah feel drawn to?
match (a)-[r]->(b) where b.fileName is null and a.fileName is null and type(r) = 'FEELS_DRAWN_TO'
RETURN b.id


The question is:
{question}"""


CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)



llm = VertexAI(model_name = "gemini-pro", temperature=0, max_output_tokens=8192)

chain = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True, cypher_prompt=CYPHER_GENERATION_PROMPT
)

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str


# Create a graphlib graph object
def showgraph(query):
    df = run_query(query)
    graph = graphviz.Digraph()
    graph.attr(size='8,5')
    graph.attr(fontsize='20')
    for index, row in df.iterrows():
        node1 = row['loc1']
        node2 = row['loc2']
        relation=row['loc3']
        print("Node1:",node1, "Node2:",node2)
        graph.edge(node2,node1, label=relation)
    return graph

def load_css():
    with open("styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "interaction" not in st.session_state:
        st.session_state.interaction = 0
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0

def on_click_callback():
    # Extract the input text from the request
    human_prompt = st.session_state.human_prompt
    print(human_prompt)
    if human_prompt == "":
        human_prompt = "No input received"

    if (st.session_state.interaction  == 0):
        restart = True
        st.session_state.interaction += 1
        augmented_prompt = human_prompt

    else:
        st.session_state.interaction += 1
        restart = False
        augmented_prompt = human_prompt
    llm_response =  chain.run(augmented_prompt)
    print("Interaction:", st.session_state.interaction)
            
    print(llm_response)

    st.session_state.history.append(
    Message("human", human_prompt)
    )
    st.session_state.history.append(
    Message("ai", llm_response)
    )
    return "Clicked"


load_css()

initialize_session_state()

sample_questions = """
Sample Questions:
- Where did Sarah build her career?
- What is Sarah's occupation?
- What degree does Sarah hold?
- What company has Sarah used before?
- What company does Sarah trust?
- What all has Sarah invested in?
- What has Sarah secured?
- What all has Sarah Purchased?
- Which location does Sarah feel drawn to? """

st.title("Sarah")
st.header("Customer 360 Profile")
st.graphviz_chart(showgraph("match (a)-[r]->(b) where b.fileName is null and a.fileName is null  return a.id as loc1,b.id as loc2, type(r) as loc3"))
#st.graphviz_chart(showgraph("MATCH p=(i:ItemLocation)-[:CONTAINS*]->(n2:ItemLocation) where i.locationId = 'DC1'  match (n1)-[]->(n2:ItemLocation) RETURN distinct n1.itemLocationDescription as loc1, n2.itemLocationDescription as loc2"))
#st.header("DC2 - Direct DC - Supply Network")
#st.graphviz_chart(showgraph("MATCH p=(i:ItemLocation)-[:CONTAINS*]->(n2:ItemLocation) where i.locationId = 'DC2'  match (n1)-[]->(n2:ItemLocation) RETURN distinct n1.itemLocationDescription as loc1, n2.itemLocationDescription as loc2"))
#st.header("DC3 - XDock  - Supply Network")
#st.graphviz_chart(showgraph("MATCH p=(i:ItemLocation)-[:CONTAINS*]->(n2:ItemLocation) where i.locationId = 'DC3'  match (n1)-[]->(n2:ItemLocation) RETURN distinct n1.itemLocationDescription as loc1, n2.itemLocationDescription as loc2"))
#st.graphviz_chart(showgraph("MATCH (n1:Location)-[]->(n2:Item) RETURN n1.locationId as loc1, n2.itemNumber as loc2"))

st.header(sample_questions)

hide_streamlit_style = '''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
'''

st.markdown(hide_streamlit_style, unsafe_allow_html=True)



chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

with chat_placeholder:
    Interaction = 0
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row 
    {'' if chat.origin == 'ai' else 'row-reverse'}">
    <img class="chat-icon" src="{
        'bot.png' if chat.origin == 'ai' 
                  else 'human.png'}"
         width=32 height=32>
    <div class="chat-bubble
    {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
        &#8203;{chat.message}
        """
        st.markdown(div, unsafe_allow_html=True)
    
    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("***Chat*** - _Press Enter to submit_")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="Hello bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback, 
    )



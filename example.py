%pip install --upgrade pip setuptools wheel
%pip install tiktoken --only-binary=:all:

# Install required libraries
%pip install -qU \
    langchain==0.3.* \
    langchain_openai==0.3.* \
    langchain_community \
    unstructured[md]==0.17.* \
    langgraph==0.4.* \
    websockets==15.0.*

import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass("Enter your OpenAI API key: ")

import websockets

# URL for the WebSocket server (make sure the ticketing system is running)
WS_URL = "ws://localhost:3000/ws"

# This async function connects to the WebSocket and listens for ticket updates
async def listen_for_ticket_updates():
    print("Starting connection")
    # Establish a connection to the WebSocket server
    async with websockets.connect(WS_URL) as websocket:
        print("WebSocket connection established.")
        try:
            # Keep listening for messages from the server
            while True:
                message = await websocket.recv()  # Wait for a new message
                print(f"Ticket update received! Ticket ID: {message}")  # Print the update
        except websockets.ConnectionClosed:
            print("WebSocket connection closed.")
        except Exception as e:
            print(f"WebSocket error: {e}")

# To run the async function in a notebook cell, use 'await' (Jupyter supports this)
await listen_for_ticket_updates()

import time
import requests

while True:
    response = requests.get("http://localhost:3000/api/tickets")
    tickets = response.json()
    print(tickets)  # Or process as needed
    time.sleep(10)  # Wait 10 seconds before polling again

from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
import requests

# Load the OpenAPI specification from the running ticketing system
root = "http://localhost:3000"
api_spec_url = f"{root}/api/docs/openapi.json"

# Download and parse the OpenAPI spec
response = requests.get(api_spec_url)
data = response.json()
data['servers'] = [{'url': root}]
openapi_spec = reduce_openapi_spec(data, dereference=False)

# Show the OpenAPI spec details
print('Servers:', openapi_spec.servers)
print('Descriptions:', openapi_spec.description)
print('Endpoints:')
for endpoint in openapi_spec.endpoints:
    print(endpoint)

from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi import planner
from langchain_openai import ChatOpenAI

requests_wrapper = RequestsWrapper()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

agent = planner.create_openapi_agent(
    api_spec=openapi_spec,
    requests_wrapper=requests_wrapper,
    llm=llm,
    verbose=True,
    allow_dangerous_requests=True,
    handle_parsing_errors=True,
    allow_operations=['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
)

# Example: Close a ticket by ID 
result = agent.invoke('Close ticket 1a2b3c4d-0001-0000-0000-000000000001 category to "Maintenance"')
print(result)

import json

# This async function connects to the WebSocket and listens for ticket updates
# Once a ticket update is received, it yields it for processing.
async def listen_for_ticket_updates():
    print("Starting connection")
    # Establish a connection to the WebSocket server
    async with websockets.connect(WS_URL) as websocket:
        print("WebSocket connection established.")
        try:
            # Keep listening for messages from the server
            while True:
                message = await websocket.recv()  # Wait for a new message
                yield json.loads(message)
        except websockets.ConnectionClosed:
            print("WebSocket connection closed.")
        except Exception as e:
            print(f"WebSocket error: {e}")

# To run the async function in a notebook cell, use 'await' (Jupyter supports this)
async for message in listen_for_ticket_updates():
    ticket_id = message.get('ticketId')
    update_type = message.get('updateType')
    
    if update_type == 'created':
        print(f'Categorizing ticket: {ticket_id}')
        agent.invoke(f"""
Based only on the ticket information, categorize the ticket into one of the following categories:
                
- Mechanical
- Quality
- Maintenance
- Technical
                
Ticket ID: {ticket_id}
""".strip())

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

loader = DirectoryLoader("./support-info")
docs = loader.load()

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

vector_store.as_retriever().invoke("Machine won't start.")




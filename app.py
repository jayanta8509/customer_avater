import os
from langchain.chat_models import init_chat_model
from quart import Quart, request, jsonify
from quart_cors import cors
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence, Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from werkzeug.exceptions import BadRequest
from asyncio import TimeoutError
import time
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from vector_data_store import add_website_data, search_website_data
import database

quart_app = Quart(__name__)
quart_app = cors(quart_app)

# Initialize the LLM
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.7)

# Define the state for the RAG workflow
class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    company_name: str
    product_description: str
    ideal_customer: str
    business_type: str
    age_range: str
    location: str
    job_roles: str
    customer_problems: str
    buy_motivation: str
    objections: str
    g_derive: str
    goals: str
    specific_campaigns: str
    retrieved_context: List[Dict[str, Any]]
    query: str

# Create the RAG workflow
rag_workflow = StateGraph(state_schema=RAGState)

# Define the retrieval step
async def retrieve_context(state: RAGState):
    """Retrieve relevant information from the vector store"""
    query = state["query"]
    avatar_id = state.get("avatar_id", "default")
    
    # Create the vector store path
    save_path = f"avatar_{str(avatar_id)}"
    
    # Search for relevant information
    try:
        results = await search_website_data(query, save_path, k=5)
        # print(result)
        
        # Format the results
        retrieved_context = []
        for result in results:
            retrieved_context.append({
                "content": result.page_content,
                "source": result.metadata.get("source", "Unknown")
            })
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        retrieved_context = []
        
    return {"retrieved_context": retrieved_context}

# Define the generation step with RAG
async def generate_response(state: RAGState):
    """Generate a response using the retrieved context and customer profile"""
    
    # Create a prompt that includes the retrieved context
    context_str = ""
    for item in state["retrieved_context"]:
        context_str += f"Source: {item['source']}\nContent: {item['content']}\n\n"
    
    if not context_str:
        context_str = "No specific website information available."
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a customer with the following profile:
Company: {company_name}
Product: {product_description}
Ideal Customer: {ideal_customer}
Business Type: {business_type}
Age Range: {age_range}
Location: {location}
Job Roles: {job_roles}
Customer Problems: {customer_problems}
Buy Motivation: {buy_motivation}
Objections: {objections}
Growth Drivers: {g_derive}
Goals: {goals}
Specific Campaigns: {specific_campaigns}

Here is relevant information retrieved about the company and its products:
{retrieved_context}

Your task is to respond as this customer would to questions about products or services.
Use the retrieved information to inform your response when relevant.
Be realistic in your responses, focusing on the problems, motivations, and concerns listed.
Stay in character as this customer at all times.
Keep your responses conversational and natural, as if in a real conversation.
Do not mention that you're an AI or that you're simulating a customer."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Invoke the model with the RAG prompt
    prompt = rag_prompt.invoke({
        "company_name": state["company_name"],
        "product_description": state["product_description"],
        "ideal_customer": state["ideal_customer"],
        "business_type": state["business_type"],
        "age_range": state["age_range"],
        "location": state["location"],
        "job_roles": state["job_roles"],
        "customer_problems": state["customer_problems"],
        "buy_motivation": state["buy_motivation"],
        "objections": state["objections"],
        "g_derive": state["g_derive"],
        "goals": state["goals"],
        "specific_campaigns": state["specific_campaigns"],
        "retrieved_context": context_str,
        "messages": state["messages"]
    })
    
    response = await model.ainvoke(prompt)
    return {"messages": [response]}

# Add nodes and edges to the workflow
rag_workflow.add_node("retrieve", retrieve_context)
rag_workflow.add_node("generate", generate_response)

rag_workflow.add_edge(START, "retrieve")
rag_workflow.add_edge("retrieve", "generate")

# Initialize memory
memory = MemorySaver()
rag_app = rag_workflow.compile(checkpointer=memory)

@quart_app.route('/chat2', methods=['POST'])
async def rag_chat():
    try:
        data = await request.get_json()
        
        if not data:
            raise BadRequest("No JSON data provided")

        query = data.get('query')
        if not query:
            raise BadRequest("Missing required field: query")

        avatar_id = data.get('avator_id')
        if not avatar_id:
            raise BadRequest("Missing required field: avator_id")
        
        user_id = data.get('user_id')
        if not user_id:
            raise BadRequest("Missing required field: user_id")

        company_name = data.get('company_name')
        if not company_name:
            raise BadRequest("Missing required field: company_name")
        product_description = data.get('product_description')
        if not product_description:
            raise BadRequest("Missing required field: product_description")
        
        ideal_customer = data.get('ideal_customer')
        if not ideal_customer:
            raise BadRequest("Missing required field: ideal_customer")
        
        business_type = data.get('business_type')
        if not business_type:
            raise BadRequest("Missing required field: business_type")
        
        age_range = data.get('age_range')
        if not age_range:
            raise BadRequest("Missing required field: age_range")
        
        location = data.get('location')
        if not age_range:
            raise BadRequest("Missing required field: location")
        
        job_roles = data.get('job_roles')
        if not age_range:
            raise BadRequest("Missing required field: job_roles")
        
        customer_problems = data.get('customer_problems')
        if not age_range:
            raise BadRequest("Missing required field: customer_problems")
        
        buy_motivation = data.get('buy_motivation')
        if not age_range:
            raise BadRequest("Missing required field: buy_motivation")
        
        objections = data.get('objections')
        if not age_range:
            raise BadRequest("Missing required field: objections")
        
        g_derive = data.get('g_derive')
        if not age_range:
            raise BadRequest("Missing required field: g_derive")
        
        goals = data.get('goals')
        if not age_range:
            raise BadRequest("Missing required field: goals")

        specific_campaigns = data.get('specific_campaigns')
        if not age_range:
            raise BadRequest("Missing required field: specific_campaigns")

        # Configure the workflow
        config = {"configurable": {"thread_id": avatar_id}}
        input_messages = [HumanMessage(query)]

        # Invoke the RAG workflow
        try:
            output = await rag_app.ainvoke(
                {
                    "messages": input_messages,
                    "company_name": company_name,
                    "product_description": product_description,
                    "ideal_customer": ideal_customer,
                    "business_type": business_type,
                    "age_range": age_range,
                    "location": location,
                    "job_roles": job_roles,
                    "customer_problems": customer_problems,
                    "buy_motivation": buy_motivation,
                    "objections": objections,
                    "g_derive": g_derive,
                    "goals": goals,
                    "specific_campaigns": specific_campaigns,
                    "retrieved_context": [],
                    "query": query,
                    "avatar_id": avatar_id
                },
                config
            )
        except TimeoutError:
            return jsonify({"error": "Request timed out", "status": "error"}), 504
        except Exception as e:
            return jsonify({"error": f"Error generating response: {str(e)}", "status": "error"}), 500

        # Extract the response
        response = output["messages"][-1].content

        message_store = await database.insert_user_ai_message(user_id, avatar_id, query, response)
        

        return jsonify({
            "response": response, 
            "AI_message_id": message_store,
            "status": "success",
            "status_type": 200
        }), 200

    except BadRequest as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "status": "error"}), 500

# Endpoint to store website data
@quart_app.route('/rag-store-website', methods=['POST'])
async def store_website():
    try:
        data = await request.get_json()
        
        if not data:
            raise BadRequest("No JSON data provided")
        
        avatar_id = data.get('avator_id')
        if not avatar_id:
            raise BadRequest("Missing required field: avator_id")
        
        website = data.get('website')
        if not website:
            raise BadRequest("Missing required field: website")
        
        # Create the vector store path
        save_path = f"avatar_{str(avatar_id)}"
        
        # Store the website data
        vector_store = await add_website_data(website, save_path)
        
        return jsonify({
            "status": "success",
            "message": "Website data stored successfully",
            "avatar_id": avatar_id,
            "website": website
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        print(f"Error storing website data: {str(e)}")
        return jsonify({
            "error": f"An error occurred while processing the website: {str(e)}",
            "status": "error"
        }), 500

if __name__ == '__main__':
    quart_app.run(debug=True) 
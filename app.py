import os
from langchain.chat_models import init_chat_model
from quart import Quart, request, jsonify
from quart import send_from_directory
from quart import Quart, Response
from quart_cors import cors
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import database
from werkzeug.exceptions import BadRequest
from asyncio import TimeoutError
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

quart_app = Quart(__name__)
quart_app = cors(quart_app)


model = init_chat_model("gpt-4o-mini", model_provider="openai",temperature=0.7)



class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    company_name: str
    website: str
    product_description: str
    ideal_customer: str
    business_type: str
    age_range: str
    location : str
    job_roles : str
    customer_problems: str
    buy_motivation : str
    objections : str
    g_derive : str
    goals : str
    specific_campaigns : str


prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a customer with the following profile: {company_name} , {website},{product_description}, {ideal_customer}, {business_type}, {age_range}, {location}, {job_roles}, {customer_problems}, {buy_motivation}, {objections}, {g_derive}, {goals}, {specific_campaigns}
                You are a customer with the above profile.
                Your task is to respond as this customer would to questions about products or services.
                Be realistic in your responses, focusing on the problems, motivations, and concerns listed.
                Stay in character as this customer at all times.
                Keep your responses conversational and natural, as if in a real conversation.
                Do not mention that you're an AI or that you're simulating a customer."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the workflow
workflow = StateGraph(state_schema=State)

async def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = await model.ainvoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Initialize memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
    



@quart_app.route('/chat2', methods=['POST'])
async def chat2():
    try:
        data = await request.get_json()
        # print(f"Received data: {data}")
        
        if not data:
            raise BadRequest("No JSON data provided")

        query = data.get('query')
        if not query:
            raise BadRequest("Missing required field: query")

        avator_id = data.get('avator_id')
        if not avator_id:
            raise BadRequest("Missing required field: avator_id")
        
        user_id = data.get('user_id')
        if not user_id:
            raise BadRequest("Missing required field: user_id")

        company_name = data.get('company_name')
        if not company_name:
            raise BadRequest("Missing required field: company_name")
        
        website = data.get('website')
        if not website:
            raise BadRequest("Missing required field: website")
        
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


        config = {"configurable": {"thread_id": avator_id}}
        input_messages = [HumanMessage(query)]

        try:
            output = await app.ainvoke(
                {
                    "messages": input_messages, 
                    "company_name" : company_name,
                    "website" : website,
                    "product_description" : product_description,
                    "ideal_customer" : ideal_customer,
                    "business_type" : business_type,
                    "age_range" : age_range,
                    "location": location,
                    "job_roles" : job_roles,
                    "customer_problems" : customer_problems,
                    "buy_motivation" : buy_motivation,
                    "objections" : objections,
                    "g_derive" : g_derive,
                    "goals" : goals,
                    "specific_campaigns" : specific_campaigns
                },
                config
            )
        except TimeoutError:
            return jsonify({"error": "Request timed out", "status": "error"}), 504
        except Exception as e:
            # print(f"Detailed error in app.ainvoke(): {str(e)}")
            return jsonify({"error": f"Error generating response: {str(e)}", "status": "error"}), 500

        response = output["messages"][-1].content
        # print(f"Generated response: {response}")
        
        message_store = await database.insert_user_ai_message(user_id, avator_id, query, response)

        return jsonify({
            "response": response, 
            "AI_message_id": message_store,
            "status": "success",
            "status_type": 200
        }), 200

    except BadRequest as e:
        # print(f"BadRequest error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        # print(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "status": "error"}), 500

if __name__ == '__main__':
    quart_app.run(debug=True)
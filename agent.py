import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define Operational Tools
# ---------------------------------------------------------------------------
@tool
def calculate_feed_consumption(animal_type: str, head_count: int, days: int) -> str:
    """
    Calculates exact total feed requirements based on stock count and duration.
    Daily intake rates:
    - rabbit: 0.12 kg/day
    - broiler: 0.15 kg/day
    - layer: 0.11 kg/day
    """
    rates = {
        "rabbit": 0.12,
        "broiler": 0.15,
        "layer": 0.11,
    }
    species = animal_type.lower().strip()
    if species not in rates:
        return f"Unknown animal type '{animal_type}'. Supported types: {list(rates.keys())}"

    daily_per_head = rates[species]
    total_kg = round(head_count * daily_per_head * days, 2)
    standard_50kg_bags = round(total_kg / 50.0, 1)

    return (
        f"Feed calculation for {head_count} {species}s over {days} days:\n"
        f"- Daily rate per head: {daily_per_head} kg\n"
        f"- Total feed required: {total_kg} kg\n"
        f"- Equivalent 50kg bags: {standard_50kg_bags} bags"
    )


@tool
def check_equipment_telemetry(device_id: str) -> str:
    """
    Retrieves current operating telemetry and maintenance alerts for on-site machinery.
    """
    telemetry_db = {
        "PUMP-TX8": {
            "status": "OPERATIONAL",
            "operating_hours": 248,
            "last_service": "2026-08-10",
            "oil_level": "WARNING_LOW",
            "notes": "Approaching 250h maintenance threshold. Service required in 2 hours."
        },
        "GEN-45KVA": {
            "status": "STANDBY",
            "operating_hours": 82,
            "fuel_level": "94%",
            "battery_voltage": "26.8V",
            "notes": "Ready for emergency auto-transfer."
        }
    }
    
    dev = device_id.upper().strip()
    if dev in telemetry_db:
        return str(telemetry_db[dev])
    return f"Device '{device_id}' not found in active telemetry registry."


tools = [calculate_feed_consumption, check_equipment_telemetry]

# ---------------------------------------------------------------------------
# 2. Multi-Provider Setup with Fallbacks (.with_fallbacks)
# ---------------------------------------------------------------------------
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

# Primary: Gemini 3.8 Flash (set max_retries=0 so it immediately yields to fallback on 429)
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=gemini_key,
    timeout=10.0,
    max_retries=0,
    temperature=0.0,
)

# Backup: Groq (runs on custom LPUs, fast inference and generous rate limits)
backup_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.0,
)

# Bind the tools to both models
primary_with_tools = primary_llm.bind_tools(tools)
backup_with_tools = backup_llm.bind_tools(tools)

# Automatic Failover: If Gemini throws a 429, 503, or times out, Groq takes over instantly
llm_with_tools = primary_with_tools.with_fallbacks([backup_with_tools])

# ---------------------------------------------------------------------------
# 3. Define Graph State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ---------------------------------------------------------------------------
# 4. Define Nodes
# ---------------------------------------------------------------------------
def agent_node(state: AgentState):
    """Executes the LLM (with fallback failover) with current message history."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools=tools)

# ---------------------------------------------------------------------------
# 5. Build and Compile the StateGraph
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node(
    "agent", 
    agent_node, 
    retry=RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)
)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# ---------------------------------------------------------------------------
# 6. Test Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    system_message = SystemMessage(
        content=(
            "You are an autonomous farm operations engineer. "
            "Use your tools to query machine states and compute precise resource requirements. "
            "Never guess arithmetic or machine health; always invoke the appropriate tool."
        )
    )

    query = (
        "We have 450 rabbits that need feeding for the next 21 days. "
        "Also, check if the irrigation pump PUMP-TX8 needs immediate servicing. "
        "Summarize both results clearly."
    )

    print(f"\nUser Query: {query}\n" + "-" * 60)

    initial_state = {
        "messages": [
            system_message,
            HumanMessage(content=query),
        ]
    }

    for event in graph.stream(initial_state, stream_mode="values"):
        latest_message = event["messages"][-1]
        sender = latest_message.__class__.__name__

        if sender == "AIMessage" and latest_message.tool_calls:
            print(f"\n[Agent Action] Calling Tools:")
            for call in latest_message.tool_calls:
                print(f"  -> {call['name']}({call['args']})")
        elif sender == "ToolMessage":
            print(f"\n[Tool Output] {latest_message.name}:")
            print(f"  {latest_message.content}")
        elif sender == "AIMessage" and not latest_message.tool_calls:
            content = latest_message.content
            if isinstance(content, list):
                clean_text = "".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            else:
                clean_text = str(content)

            print(f"\n[Final Answer]\n{clean_text}")

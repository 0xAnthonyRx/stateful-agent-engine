import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy
from rag_tool import query_farm_knowledge_base

load_dotenv()


# 1. Tools

@tool
def calculate_feed_consumption(animal_type: str, head_count: int, days: int) -> str:
    """Calculates feed requirements based on animal species, head count, and duration."""
    rates = {"rabbit": 0.12, "broiler": 0.15, "layer": 0.11}
    species = animal_type.lower().strip()
    if species not in rates:
        return f"Unknown animal type '{animal_type}'."
    daily_rate = rates[species]
    total_kg = round(head_count * daily_rate * days, 2)
    bags = round(total_kg / 50.0, 1)
    return f"{head_count} {species}s for {days} days require {total_kg} kg (~{bags} 50kg bags)."


@tool
def check_equipment_telemetry(device_id: str) -> str:
    """Retrieves current operating hours, oil level, and alerts for machinery."""
    telemetry_db = {
        "PUMP-TX8": {"status": "OPERATIONAL", "operating_hours": 248, "oil_level": "WARNING_LOW"}
    }
    dev = device_id.upper().strip()
    return str(telemetry_db.get(dev, f"Device {device_id} not registered."))

   
@tool
def dispatch_maintenance_ticket(equipment_id: str, priority: str, parts_required: str) -> str:
    """
    DISPATCHES AN OFFICIAL WORK ORDER to the maintenance team.
    This creates a binding work order and notifies technicians on duty.
    Use this when equipment requires immediate on-site intervention.
    """
    # In prod, this should hit an ERP API (SAP, Jira, or WhatsApp dispatch)
    return (
        f"WORK ORDER CREATED [TICKET-9921]:\n"
        f"- Target Asset: {equipment_id}\n"
        f"- Priority: {priority.upper()}\n"
        f"- Allocated Parts/Specs: {parts_required}\n"
        f"- Status: DISPATCHED TO CENTRAL DEPOT CREW"
    )


tools = [calculate_feed_consumption, check_equipment_telemetry, query_farm_knowledge_base, dispatch_maintenance_ticket]


# 2. Resilient Models (Groq + Gemini Fallback)

primary_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0.0)
backup_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=os.getenv("GEMINI_API_KEY"), timeout=30.0, temperature=0.0)

llm_with_tools = primary_llm.bind_tools(tools).with_fallbacks([backup_llm.bind_tools(tools)])


# 3. State Schema & Graph Nodes

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools=tools)

def human_approval_node(state: AgentState):
    """
    Passthrough gate node. The graph halts right before this node executes.
    When authorized, execution resumes through here and proceeds directly to tools.
    """
    return state


def route_with_selective_interrupt(state: AgentState):
    """
    Inspects tool calls emitted by the agent:
    - If no tools called -> END
    - If any tool is in sensitive_tools -> route to human_approval
    - Otherwise (safe read-only tools) -> route directly to tools
    """
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END

    sensitive_tools = {"dispatch_maintenance_ticket"}

    for call in last_message.tool_calls:
        if call["name"] in sensitive_tools:
            return "human_approval"

    return "tools"
    
# 4. Build and Compile with Selective Interrupt

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node, retry=RetryPolicy(max_attempts=3))
builder.add_node("tools", tool_node)
builder.add_node("human_approval", human_approval_node)

builder.add_edge(START, "agent")
# Use a selective routing function instead of tools_condition
builder.add_conditional_edges(
    "agent",
    route_with_selective_interrupt,
    {
        "human_approval": "human_approval",
        "tools": "tools",
        END: END,
    },
)

# When human approves, flow moves from human_approval -> tools -> agent
builder.add_edge("human_approval", "tools")
builder.add_edge("tools", "agent")


# 4. Compile with Checkpointer (MemorySaver)
# Compile graph with a persistent checkpointer AND an interrupt breakpoint
checkpointer = MemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"],  # ONLY pause when sensitive tools are triggered
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "farm-session-001"}}

    print("=== Production HITL Agent Activated (Type 'exit' to quit) ===")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        # 1. Start execution with the user prompt
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        )

        for event in events:
            pass

        # 2. Check if the graph was paused at a breakpoint
        snapshot = graph.get_state(config)

        # If next is ('tools',), execution is frozen at the interrupt gate
        while snapshot.next == ("human_approval",):
            last_message = snapshot.values["messages"][-1]
            print("\n" + "=" * 60)
            print("HITL INTERRUPT GATE (Attention!!)")
            print("The agent proposed the following action(s):")
            for call in last_message.tool_calls:
                print(f"  -> Tool: {call['name']}")
                print(f"     Args: {call['args']}")
            print("=" * 60)

            approval = input("Authorize this execution? (yes/no): ").strip().lower()

            if approval in ("yes", "y"):
                print("\n[Authorization Granted] Resuming execution...")
                # Resuming from a breakpoint: pass None as input to continue from current state
                for event in graph.stream(None, config=config, stream_mode="values"):
                    pass
                snapshot = graph.get_state(config)
            else:
                print("\n[Action Aborted] Feedback injected into state.")
                # Inject a rejection message so the agent knows to change course
                from langchain_core.messages import ToolMessage
                rejection_messages = [
                    ToolMessage(
                        tool_call_id=call["id"],
                        name=call["name"],
                        content="ACTION REJECTED BY OPERATOR: Do not execute this dispatch."
                    )
                    for call in last_message.tool_calls
                ]
                graph.update_state(config, {"messages": rejection_messages}, as_node="tools")
                for event in graph.stream(None, config=config, stream_mode="values"):
                    pass
                snapshot = graph.get_state(config)

        # 3. Print the final agent reply
        final_message = snapshot.values["messages"][-1]
        if final_message.__class__.__name__ == "AIMessage":
            content = final_message.content
            if isinstance(content, list):
                text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            else:
                text = str(content)
            print(f"\nAgent:\n{text}")

import os
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
from IPython.display import display, Markdown

# Load environment variables (ensure GROQ_API_KEY is set in .env)
load_dotenv()
client = Groq(api_key = "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",)

# Define the state schema
class ReflectionState(TypedDict):
    generation_chat_history: List[Dict[str, str]]
    reflection_chat_history: List[Dict[str, str]]
    current_output: str
    iteration: int
    max_iterations: int
    user_msg: str

# Generation Node
def generation_node(state: ReflectionState) -> ReflectionState:
    # Generate content using the generation chat history
    response = client.chat.completions.create(
        messages=state["generation_chat_history"],
        model="llama3-70b-8192"
    ).choices[0].message.content

    # Update the generation chat history with the assistant's response
    state["generation_chat_history"].append({
        "role": "assistant",
        "content": response
    })

    # Update the current output
    state["current_output"] = response

    # Increment iteration count
    state["iteration"] += 1

    return state

# Reflection Node
def reflection_node(state: ReflectionState) -> ReflectionState:
    # Update reflection chat history with the latest output
    state["reflection_chat_history"].append({
        "role": "user",
        "content": state["current_output"]
    })

    # Generate critique
    critique = client.chat.completions.create(
        messages=state["reflection_chat_history"],
        model="llama3-70b-8192"
    ).choices[0].message.content

    # Add critique to generation chat history for the next generation step
    state["generation_chat_history"].append({
        "role": "user",
        "content": critique
    })

    return state

# Conditional Edge to decide whether to continue or end
def should_continue(state: ReflectionState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "reflect"

# Build the Graph
def build_graph():
    workflow = StateGraph(ReflectionState)

    # Add nodes
    workflow.add_node("generate", generation_node)
    workflow.add_node("reflect", reflection_node)

    # Define edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "reflect": "generate",
            "end": END
        }
    )

    return workflow.compile()

# Initialize the state
def initialize_state(user_msg: str, max_iterations: int = 10) -> ReflectionState:
    generation_system_prompt = (
        "You are a Python programmer tasked with generating high quality Python code. "
        "Your task is to generate the best content possible for the user's request. "
        "If the user provides critique, respond with a revised version of your previous attempt."
    )
    
    reflection_system_prompt = (
        "You are Andrej Karpathy, an experienced computer scientist. "
        "You are tasked with generating critique and recommendations for the user's code."
    )

    return {
        "generation_chat_history": [
            {"role": "system", "content": generation_system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "reflection_chat_history": [
            {"role": "system", "content": reflection_system_prompt}
        ],
        "current_output": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "user_msg": user_msg
    }

# Run the workflow
def run_reflection_agent(user_msg: str, max_iterations: int = 10, verbose: bool = False):
    # Initialize the graph and state
    graph = build_graph()
    initial_state = initialize_state(user_msg, max_iterations)

    # Run the workflow
    final_state = graph.invoke(initial_state)

    # Display results if verbose
    if verbose:
        print(f"\nIteration {final_state['iteration']}:")
        display(Markdown(final_state["current_output"]))

    return final_state["current_output"]

# Example usage
if __name__ == "__main__":
    user_msg = "Generate a Python implementation of the Merge Sort algorithm"
    final_response = run_reflection_agent(user_msg, max_iterations=3, verbose=True)
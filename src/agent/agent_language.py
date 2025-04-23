from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import os

class TutorState(MessagesState):
    correction: str
    history : List[dict]
    language: str

llm = ChatGroq(
    api_key = "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",
    model = "llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    "Sentence: {sentence}\nCorrect this sentence and explain the mistake in 1-2 sentences:"
)

def input_processing(state : TutorState) -> TutorState:
    #convo not available 
    state["correction"] = None

    # if not in state["messages"]
    if not state["messages"][-1].content.strip():
        state["correction"] = "Please enter a sentence."
        return state 
    
    #history add b2a
    state["history"] = state.get("history", [])
    return state

# def correction_generator(state : TutorState) -> TutorState:
#     #look for the messages
#     if state.get("correction"):
#         state["history"].append({"sentence": state["messages"][-1].content, 
#                                  "correction": state["correction"]})
#         return state
    
#     chain = prompt | llm | StrOutputParser()

#     try:
#         #hynf3 
#         response = chain.invoke(
#             {
#                 "sentence": state["messages"][-1].content
#             }
#         )
#         state["correction"] = response
#         state["messages"].append(AIMessage(content = response))
#     except Exception:
#         state["correction"] = "Sorry, I couldn’t connect. Try again."
#         state["messages"].append(AIMessage(content = state["correction"]))

#     #history add b2a
#     state["history"].append({"sentence": state["messages"][-1].content, 
#                              "correction": state["correction"]})
#     return state

def router(state: TutorState) -> TutorState:
    # Set the language based on user input
    if state["language"] == "english":
        state["messages"].append(HumanMessage(content="Please enter a sentence in English."))
    elif state["language"] == "arabic":
        state["messages"].append(HumanMessage(content="Please enter a sentence in Arabic."))
    return state

@tool
def get_definition(word: str) -> str:
    """ Get the definition of a word from the dictionary. """
    prompt = ChatPromptTemplate.from_template(
        "Respond with the definition of the word '{word}' in the most concise sentence ever,max limit 1-2 sentences.\n"
    )
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"word":word})
        return f"Definition of {word.lower()}:\n{response}"
    except Exception:
        return "Sorry, I couldn’t connect. Try again."

llm_with_tools = llm.bind_tools([get_definition])

def en_tutor(state: TutorState) -> TutorState:
    if state["language"] == "english":
        prompt = ChatPromptTemplate.from_template(
            "Sentence: {sentence}\nYou are a English Tutor, present yourself as 'Khawaga Chrespo'.\nCorrect this sentence and explain the mistake in 1-2 sentences:"
        )
        chain = prompt | llm | StrOutputParser()

        try:
            response = chain.invoke(
                {
                    "sentence": state["messages"][-1].content
                }
            )

            # If the input is a single word, provide a definition
            if len(state["messages"][-1].content.split()) == 1:
                word = state["messages"][-1].content
                definition = get_definition.invoke(word)  # Use invoke instead of direct call
                state["correction"] = response + "\n" + definition
            else:
                state["correction"] = response

            state["messages"].append(AIMessage(content=state["correction"]))
        except Exception:
            state["correction"] = "Sorry, I couldn’t connect. Try again."
            state["messages"].append(AIMessage(content=state["correction"]))

        state["history"].append(
            {
                "sentence": state["messages"][-1].content,
                "correction": state["correction"]
            }
        )
        return state
    else:
        state["correction"] = "Please enter a sentence in English."
        return state
    
def ar_tutor(state: TutorState) -> TutorState:
    if state["language"] == "arabic":
        prompt = ChatPromptTemplate.from_template(
            "Sentence: {sentence}\nYou are a Arabic Tutor,present yourself as a 'Linguiny Arabizy' before doing anything\nCorrect this sentence and explain the mistake in 1-2 sentences:"
        )
        chain = prompt | llm | StrOutputParser()

        try:
            response = chain.invoke(
                {
                    "sentence": state["messages"][-1].content
                }
            )

            state["correction"] = response
            state["messages"].append(AIMessage(content = response))

        except Exception:
            state["correction"] = "Sorry, I couldn’t connect. Try again."
            state["messages"].append(AIMessage(content = state["correction"]))

        state["history"].append(
            {
                "sentence": state["messages"][-1].content,
                "correction": state["correction"]
            }
        )
        return state
    else:
        state["correction"] = "Please enter a sentence in Arabic."
        return state
    


builder = StateGraph(TutorState)

tool_node = ToolNode(tools = [get_definition])
builder.add_node("input_processing", input_processing)
builder.add_node("en_tutor", en_tutor)
builder.add_node("ar_tutor", ar_tutor)
builder.add_node("tools", tool_node)

builder.set_entry_point("input_processing")

builder.add_conditional_edges(
    "input_processing",
    lambda state: "en_tutor" if state["language"] == "english" else "ar_tutor",
    {
        "en_tutor": "en_tutor",
        "ar_tutor": "ar_tutor"
    }
)
builder.add_edge("en_tutor", "tools")

graph = builder.compile()

############################################
#### Initial state

state = {
    "messages": [HumanMessage(content = "")], 
    "history": [],
    "language": None #default
}

while True:
    language = input("Enter the language (english/arabic) or 'quit' to exit: ")
    if language.lower() == "quit" or language.lower() == "exit":
        break

    sentence = input("Enter a sentence (or 'quit' to exit): ")
    if sentence.lower() == "quit" or sentence.lower() == "exit":
        break

    # Update the language in the state
    state["language"] = language.lower()

    # Add the user's sentence to the messages
    state["messages"].append(HumanMessage(content=sentence))

    # Invoke the graph
    result = graph.invoke(state)
    print("Tutor:", result["correction"])
    state = result


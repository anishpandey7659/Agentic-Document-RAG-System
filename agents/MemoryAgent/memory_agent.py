# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import InMemorySaver
# from langchain_core.runnables import RunnableConfig
# from typing import Annotated
# from typing_extensions import TypedDict
# from operator import add

# from test.retrive_test import rag


# class State(TypedDict):
#     query: str
#     foo: str
#     bar: Annotated[list[str], add]
#     answer: str
#     context: str


# def test_workflow(state: State) -> dict:
#     """Test workflow that runs the RAG agent with a given query."""

#     result = rag.run(state["query"])

#     return {
#         **state,
#         "answer": result["answer"],
#         "context": str(result.get("sources", ""))
#     }


# workflow = StateGraph(State)

# workflow.add_node("test_workflow", test_workflow)

# workflow.add_edge(START, "test_workflow")
# workflow.add_edge("test_workflow", END)

# checkpointer = InMemorySaver()

# graph = workflow.compile(checkpointer=checkpointer)

# config: RunnableConfig = {
#     "configurable": {
#         "thread_id": "1"
#     }
# }

# result = graph.invoke(
#     {
#         "query": "Hello, how are you?",
#         "foo": "",
#         "bar": []
#     },
#     config=config
# )

# print(result)

# python -m agents.MemoryAgent.memory_agent
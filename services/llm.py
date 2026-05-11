from pipeline import retrieve_and_answer,answer_normal
from services import list_available_documents
from agents.Orchestration.router.router_agent import should_retrieve
from  Model_Memory_store.memory.memory_manager import memory,conv_id



if __name__ == "__main__":

    query = "What is the recent political news in nepal"

    is_retrieve = should_retrieve(query)
    full_answer = ""
    sources = None

    memory.add_message(
        conv_id,
        role="user",
        content=query
    )
    if not is_retrieve:
        print("Answer:")
        for token in answer_normal(query,stream=True):
            print(token, end="", flush=True)
            full_answer += token
    else:
        print("Answer:")

        for event in retrieve_and_answer(query=query, stream=True):
            if event["type"] == "sources":
                sources = event["sources"]

            elif event["type"] == "token":
                print(event["token"], end="", flush=True)
                full_answer += event["token"]

    print("\n")

    if sources:
        print("Sources:", sources)

    memory.add_message(
        conv_id,
        role="assistant",
        content=full_answer,
        metadata={"sources": sources} if sources else {}
    )

# run using python -m services.llm


# [ANSWER]
# {'answer': 'Agentic\u202fAI is used to perform tasks\u202fautonomously rather than just responding to prompts.  For example, an AI agent can:\n\n*\u202fDetect a policy violation in an expense report, generate an audit trail, notify the manager and kick off corrective procurement workflows\u202f—all without human intervention.  \n*\u202fPlan, act, observe results and adapt to achieve specific goals, enabling goal‑oriented, independent operation.  \n*\u202fSupport multi‑agent collaboration, autonomous code generation and execution, and enforce governance frameworks as part of core enterprise infrastructure.  \n\nThese capabilities are shifting AI from a “help‑you” role to a “work‑for‑you” role across enterprises.', 
# 'sources': [{'rank': 1, 'score': 0.929, 'text': 'Agentic AI: State of the Field May 2026 What Is Agentic AI? Unlike traditional generative AI models that simply respond to user prompts, agentic AI systems can plan, act, observe results, and adapt as necessary to achieve a specific goal working independently and autonomously. AI systems now dont just respond they act. A real-world example: an AI agent autonomously flagged a policy violation in an expense report, generated an audit trail, notified the relevant manager, and initiated corrective procurement workflows all without a single human intervention. Market Growth The agentic AI market expanded from 7.6 billion in 2025 to a', 'doc_id': 'doc-86e44eda', 'chunk_id': 0}, 
# {'rank': 2, 'score': 0.8849, 'text': 'be made autonomously through agentic AI by 2028. The field is moving away from passive conversational models towards active, autonomous, and goal-oriented AI agents, requiring robust, scalable, and secure infrastructure. Websites and APIs are being redesigned to be machine-readable for automated agents traditional SEO and human-centric UIUX are no longer sufficient. New job titles are emerging: Agent Supervisor, Agent QA Lead, AI Ops Manager, Chief AI Officer signaling that agentic AI is maturing into core enterprise IT infrastructure. Summary 2026 marks the year agentic AI moved from pilot projects to core enterprise infrastructure, with multi-agent collaboration, autonomous coding, and governance', 'doc_id': 'doc-86e44eda', 'chunk_id': 8}, 
# {'rank': 3, 'score': 0.8678, 'text': 'marks the year agentic AI moved from pilot projects to core enterprise infrastructure, with multi-agent collaboration, autonomous coding, and governance frameworks as the defining themes. The shift is from AI that helps you to AI that works for you. Sources: Salesforce, Gartner, National Interest, Techzine, Kersai, Epsilla, DevOps.com May 2026', 'doc_id': 'doc-86e44eda', 'chunk_id': 9}]}
import os
import asyncio
from langgraph.graph import StateGraph, START, END
from langchain_together import ChatTogether
from langfuse.callback import CallbackHandler
from email_handlers import (
    EmailState,
    read_email,
    classify_email,
    handle_spam,
    drafting_response,
    notify_mr_wayne,
    route_email,
)

async def main():
    # Initialize model
    model = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=0)

    # Bind model to handlers that require it
    classify_email.model = model
    drafting_response.model = model

    # Build the state graph
    email_graph = StateGraph(EmailState)
    email_graph.add_node("read_email", read_email)
    email_graph.add_node("classify_email", classify_email)
    email_graph.add_node("handle_spam", handle_spam)
    email_graph.add_node("drafting_response", drafting_response)
    email_graph.add_node("notify_mr_wayne", notify_mr_wayne)

    email_graph.add_edge(START, "read_email")
    email_graph.add_edge("read_email", "classify_email")
    email_graph.add_conditional_edges(
        "classify_email", route_email,
        {"spam": "handle_spam", "legitimate": "drafting_response"}
    )
    email_graph.add_edge("handle_spam", END)
    email_graph.add_edge("drafting_response", "notify_mr_wayne")
    email_graph.add_edge("notify_mr_wayne", END)

    compiled = email_graph.compile()

    # Example emails
    def make_email(sender, subj, body):
        return {"sender": sender, "subject": subj, "body": body}

    legit = make_email("Joker", "Found you Batman!", "I know you're Batman. I have proof!")
    spam = make_email("Crypto bro", "Best investment of 2025", "Buy my ALT coin now!")

    print("\nProcessing legitimate email...")
    compiled.invoke({"email": legit, "is_spam": None, "spam_reason": None,
                     "email_category": None, "draft_response": None, "messages": []})

    print("\nProcessing spam email...")
    compiled.invoke({"email": spam, "is_spam": None, "spam_reason": None,
                     "email_category": None, "draft_response": None, "messages": []})

    # Langfuse tracing
    langfuse = CallbackHandler()
    compiled.invoke(
        input={"email": legit, "is_spam": None, "spam_reason": None,
               "email_category": None, "draft_response": None, "messages": []},
        config={"callbacks": [langfuse]}
    )

    # Draw graph
    compiled.get_graph().draw_mermaid_png()

if __name__ == "__main__":
    os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "YOUR_API_KEY")
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "YOUR_PUBLIC_KEY")
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "YOUR_SECRET_KEY")
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    asyncio.run(main())

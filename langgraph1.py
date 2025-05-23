import os
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_together import ChatTogether
from langfuse.callback import CallbackHandler

# Async wrapper
async def main():

    class EmailState(TypedDict):
        email: Dict[str, Any]           
        is_spam: Optional[bool]         
        spam_reason: Optional[str]      
        email_category: Optional[str]   
        draft_response: Optional[str]   
        messages: List[Dict[str, Any]]  

    model = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", 
                         temperature=0)

    class EmailState(TypedDict):
        email: Dict[str, Any]
        is_spam: Optional[bool]
        draft_response: Optional[str]
        messages: List[Dict[str, Any]]

    # Define nodes
    def read_email(state: EmailState):
        email = state["email"]
        print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
        return {}

    def classify_email(state: EmailState):
        email = state["email"]

        prompt = f"""
    As Alfred the butler of Mr wayne and it's SECRET identity Batman, analyze this email and determine if it is spam or legitimate and should be brought to Mr wayne's attention.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam.
    answer with SPAM or HAM if it's legitimate. Only reurn the answer
    Answer :
        """
        messages = [HumanMessage(content=prompt)]
        response = model.invoke(messages)

        response_text = response.content.lower()
        print(response_text)
        is_spam = "spam" in response_text and "ham" not in response_text

        if not is_spam:
            new_messages = state.get("messages", []) + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content}
            ]
        else :
            new_messages = state.get("messages", [])

        return {
            "is_spam": is_spam,
            "messages": new_messages
        }

    def handle_spam(state: EmailState):
        print(f"Alfred has marked the email as spam.")
        print("The email has been moved to the spam folder.")
        return {}

    def drafting_response(state: EmailState):
        email = state["email"]

        prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    Draft a brief, professional response that Mr. Wayne can review and personalize before sending.
        """

        messages = [HumanMessage(content=prompt)]
        response = model.invoke(messages)

        new_messages = state.get("messages", []) + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]

        return {
            "draft_response": response.content,
            "messages": new_messages
        }

    def notify_mr_wayne(state: EmailState):
        email = state["email"]

        print("\n" + "="*50)
        print(f"Sir, you've received an email from {email['sender']}.")
        print(f"Subject: {email['subject']}")
        print("\nI've prepared a draft response for your review:")
        print("-"*50)
        print(state["draft_response"])
        print("="*50 + "\n")

        return {}

    # Define routing logic
    def route_email(state: EmailState) -> str:
        if state["is_spam"]:
            return "spam"
        else:
            return "legitimate"
    # Create the graph
    email_graph = StateGraph(EmailState)

    # Add nodes
    email_graph.add_node("read_email", read_email) # the read_email node executes the read_mail function
    email_graph.add_node("classify_email", classify_email) # the classify_email node will execute the classify_email function
    email_graph.add_node("handle_spam", handle_spam) #same logic 
    email_graph.add_node("drafting_response", drafting_response) #same logic
    email_graph.add_node("notify_mr_wayne", notify_mr_wayne) # same logic

    # Add edges
    email_graph.add_edge(START, "read_email") # After starting we go to the "read_email" node

    email_graph.add_edge("read_email", "classify_email") # after_reading we classify

    # Add conditional edges
    email_graph.add_conditional_edges(
        "classify_email", # after classify, we run the "route_email" function"
        route_email,
        {
            "spam": "handle_spam", # if it return "Spam", we go the "handle_span" node
            "legitimate": "drafting_response" # and if it's legitimate, we go to the "drafting response" node
        }
    )

    # Add final edges
    email_graph.add_edge("handle_spam", END) # after handling spam we always end
    email_graph.add_edge("drafting_response", "notify_mr_wayne")
    email_graph.add_edge("notify_mr_wayne", END) # after notifyinf Me wayne, we can end  too

    # Compile the graph
    compiled_graph = email_graph.compile()

     # Example emails for testing
    legitimate_email = {
        "sender": "john.smith@example.com",
        "subject": "Question about your services",
        "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
    }

    spam_email = {
        "sender": "Crypto bro",
        "subject": "The best investment of 2025",
        "body": "Mr Wayne, I just launched an ALT coin and want you to buy some !"
    }
    # Process legitimate email
    print("\nProcessing legitimate email...")
    legitimate_result = compiled_graph.invoke({
        "email": legitimate_email,
        "is_spam": None,
        "draft_response": None,
        "messages": []
    })

    # Process spam email
    print("\nProcessing spam email...")
    spam_result = compiled_graph.invoke({
        "email": spam_email,
        "is_spam": None,
        "draft_response": None,
        "messages": []
    }) 

    # Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
    langfuse_handler = CallbackHandler()

    # Process legitimate email
    legitimate_result = compiled_graph.invoke(
        input={"email": legitimate_email, "is_spam": None, "spam_reason": None, "email_category": None, "draft_response": None, "messages": []},
        config={"callbacks": [langfuse_handler]}
    )

    compiled_graph.get_graph().draw_mermaid_png(output_file_path='./compiled_graph.png')


# Entry point
if __name__ == "__main__":
    os.environ["TOGETHER_API_KEY"] = "e6f6c6102549624c7b189476f2aee0398f7b69748a22191423ef852d86fcc31f"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-ff390ffa-108c-4d03-9739-2f57b7be26b8" 
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-b5c0e3f5-6704-4d6a-b0f7-1ddf5f03996b"
    os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
    asyncio.run(main())
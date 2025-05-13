from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import HumanMessage

class EmailState(TypedDict):
    email: Dict[str, Any]
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_category: Optional[str]
    draft_response: Optional[str]
    messages: List[Dict[str, Any]]


def read_email(state: EmailState) -> Dict[str, Any]:
    email = state['email']
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")
    return {}


def classify_email(state: EmailState) -> Dict[str, Any]:
    email = state['email']
    prompt = (
        f"""
As Alfred the butler of Mr. Wayne (Batman), analyze this email and determine if it is spam or legitimate.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

Answer with exactly 'SPAM' or 'HAM'.
Answer: """
    )
    response = classify_email.model.invoke([HumanMessage(content=prompt)])
    text = response.content.strip().upper()
    is_spam = text == 'SPAM'

    new_messages = state.get('messages', [])
    if not is_spam:
        new_messages += [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response.content},
        ]
    return {'is_spam': is_spam, 'messages': new_messages}


def handle_spam(state: EmailState) -> Dict[str, Any]:
    print("Alfred has marked the email as spam and moved it to the spam folder.")
    return {}


def drafting_response(state: EmailState) -> Dict[str, Any]:
    email = state['email']
    prompt = (
        f"""
As Alfred the butler, draft a brief, professional response for Mr. Wayne to review.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

Response: """
    )
    response = drafting_response.model.invoke([HumanMessage(content=prompt)])
    new_messages = state.get('messages', []) + [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': response.content},
    ]
    return {'draft_response': response.content, 'messages': new_messages}


def notify_mr_wayne(state: EmailState) -> Dict[str, Any]:
    email = state['email']
    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}")
    print(f"Subject: {email['subject']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state.get('draft_response', '(no draft available)'))
    print("=" * 50 + "\n")
    return {}


def route_email(state: EmailState) -> str:
    return 'spam' if state.get('is_spam') else 'legitimate'

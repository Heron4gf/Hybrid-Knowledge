"""Utility for injecting retrieval results as context messages."""
import json
from typing import Any, Dict, List

from app.models.retrieval import RetrievalQueryResponse


def build_context_message(retrieval_results: RetrievalQueryResponse) -> Dict[str, Any]:
    """Build a context message from retrieval results.
    
    Args:
        retrieval_results: The retrieval query response containing results
        
    Returns:
        A message dict with role='context' and JSON-serialized results in content
    """
    # Format results for the context message
    formatted_results = {
        "query": retrieval_results.query,
        "results": [
            {
                "id": r.id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata.model_dump(),
            }
            for r in retrieval_results.results
        ],
    }
    
    return {
        "role": "context",
        "content": json.dumps(formatted_results, indent=2),
    }


def inject_context_message(
    messages: List[Dict[str, Any]],
    context_message: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Inject a context message into the messages list.
    
    Injects after system/developer messages but before user messages.
    If no system/developer messages exist, prepends before the first user message.
    If no user message found, appends at the end.
    
    Args:
        messages: Original list of messages
        context_message: The context message to inject
        
    Returns:
        Modified list of messages with context injected
    """
    if not messages:
        return [context_message]
    
    # Find the insertion point (after system/developer, before first user)
    insert_index = 0
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role in ("system", "developer"):
            insert_index = i + 1
        elif role == "user":
            # Stop here - we'll insert before this user message
            break
        else:
            # For other roles (assistant, tool, function), continue searching
            insert_index = i + 1
    
    # Insert the context message
    new_messages = messages.copy()
    new_messages.insert(insert_index, context_message)
    return new_messages


def extract_query_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract the query text from the last user message.
    
    Args:
        messages: List of message dicts
        
    Returns:
        The content of the last user message, or empty string if not found
    """
    # Iterate in reverse to find the last user message
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list):
                # Handle content as list of parts
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "").strip()

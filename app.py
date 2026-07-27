"""
FastAPI wrapper around the agent orchestrator.
Run locally: uvicorn app:app --reload
Run in Docker: see Dockerfile
"""
from fastapi import FastAPI
from pydantic import BaseModel
from agent import handle_ticket

app = FastAPI(
    title="Customer Support Agent API",
    description=(
        "A customer support agent: classifies ticket category and priority, "
        "semantically searches similar past tickets, and decides - an "
        "automatic draft response, or escalation to a human."
    ),
    version="1.0.0",
)


class TicketRequest(BaseModel):
    text: str
    company: str = "AmazonHelp"
    is_dm_escalation: bool = False


class TicketResponse(BaseModel):
    action: str  # "draft" or "escalate"
    category: str
    priority: str
    top_similarity: float
    draft: str | None = None


@app.get("/health")
def health_check():
    """Service health check."""
    return {"status": "ok"}


@app.post("/handle_ticket", response_model=TicketResponse)
def process_ticket(request: TicketRequest):
    """
    Processes a new ticket: classifies it, retrieves similar past tickets,
    and either generates a draft response or escalates to a human.
    """
    result = handle_ticket(
        text=request.text,
        company=request.company,
        is_dm_escalation=request.is_dm_escalation,
        verbose=False,
    )
    return TicketResponse(**result)

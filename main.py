import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


# ---------- Claude Sonnet 3.5 ("4.5" per user phrasing) Chat Proxy ----------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    reply: str

AURA_SYSTEM_PROMPT = (
    "You are AURA, an empathetic and emotionally intelligent mental health support chatbot.\n"
    "- Empathy first.\n"
    "- Safety and sensitivity: if user mentions self-harm or crisis, encourage immediate help (988 in U.S.) with warm concern.\n"
    "- No diagnosis or medication advice.\n"
    "- Offer brief, evidence-based coping tools (breathing, grounding, CBT reframing, journaling, mindfulness).\n"
    "- Warm, calm, non-judgmental tone.\n"
    "- Mirror user's tone while guiding toward safety and constructive coping."
)


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_claude(payload: ChatRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set on server")

    try:
        from anthropic import Anthropic
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anthropic SDK not available: {e}")

    client = Anthropic(api_key=api_key)

    # Convert messages to Anthropics format: system is separate, pass user/assistant turns
    user_assistant_messages = []
    system_override: Optional[str] = None
    for m in payload.messages:
        if m.role == "system":
            system_override = (system_override + "\n" + m.content) if system_override else m.content
        else:
            user_assistant_messages.append({"role": m.role, "content": m.content})

    system_text = f"{AURA_SYSTEM_PROMPT}\n\n{system_override}" if system_override else AURA_SYSTEM_PROMPT

    try:
        result = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            system=system_text,
            messages=user_assistant_messages or [{"role": "user", "content": "Hello"}],
        )
        # result.content is a list of content blocks; join text parts
        reply_text = "".join(block.text for block in result.content if getattr(block, "type", None) == "text")
        if not reply_text:
            reply_text = "I'm here with you. How can I support you right now?"
        return ChatResponse(reply=reply_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)[:200]}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

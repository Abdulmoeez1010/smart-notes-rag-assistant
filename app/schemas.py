from pydantic import BaseModel, Field

# scema for ask request
class AskRequest(BaseModel):
    video_url : str = Field(..., description = "Full YouTube video URL")
    question: str = Field(..., description="User's question about the video")

# schema of ask response
class AskResponse(BaseModel):
    video_id: str
    question: str
    answer: str


# schemas for summarixation endpoint
class SummarizeRequest(BaseModel):
    video_url: str = Field(..., description="Full YouTube video URL")

class SummarizeResponse(BaseModel):
    video_id: str
    summary: str

# schemas.py additions

class IngestYouTubeRequest(BaseModel):
    video_url: str

class IngestResponse(BaseModel):
    doc_id: str
    title: str

class AskDocRequest(BaseModel):
    doc_id: str
    question: str

class SummarizeDocRequest(BaseModel):
    doc_id: str

class SummarizeDocResponse(BaseModel):
    doc_id: str
    summary: str

class QuizDocRequest(BaseModel):
    doc_id: str

class QuizDocResponse(BaseModel):
    doc_id: str
    questions: list

class MindmapDocRequest(BaseModel):
    doc_id: str

class MindmapDocResponse(BaseModel):
    doc_id: str
    mindmap: dict
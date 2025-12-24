from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from src.rag_engine import RAGEngine

app = FastAPI(title="个人知识库与智能问答系统", version="1.0.0")

# Initialize RAG engine
rag_engine = RAGEngine()

class QueryRequest(BaseModel):
    question: str

class DocumentRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/")
async def root():
    return {"message": "欢迎使用个人知识库与智能问答系统！"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    查询知识库并返回答案
    """
    try:
        result = rag_engine.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_document")
async def add_document_endpoint(request: DocumentRequest):
    """
    添加文档到知识库
    """
    try:
        success = rag_engine.add_document(request.content, request.metadata)
        if success:
            return {"message": "文档添加成功"}
        else:
            raise HTTPException(status_code=400, detail="文档添加失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    聊天接口
    """
    try:
        response = rag_engine.chat(request.messages)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_document")
async def upload_document_endpoint(file: UploadFile = File(...)):
    """
    上传文档文件
    """
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Add to knowledge base
        success = rag_engine.add_document(content_str, {
            "filename": file.filename,
            "size": len(content_str)
        })
        
        if success:
            return {"message": f"文件 {file.filename} 上传并处理成功"}
        else:
            raise HTTPException(status_code=400, detail="文件处理失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#!/bin/bash

echo "启动个人知识库与智能问答系统..."

# 检查是否安装了依赖
pip install -r requirements.txt

# 启动应用
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

echo "应用已启动，请访问 http://localhost:8000"
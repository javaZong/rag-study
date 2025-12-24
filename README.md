# 个人知识库与智能问答系统

基于Qwen3-Max LLM、Text-embedding-v4嵌入模型和Milvus向量数据库的RAG应用。

## 功能特性

- 文档上传和存储
- 智能问答
- 对话式交互
- 相似文档检索

## 安装步骤

1. 克隆项目
2. 安装依赖：`pip install -r requirements.txt`
3. 配置环境变量：复制 `.env.example` 为 `.env` 并填入API密钥
4. 启动Milvus数据库
5. 运行应用：`python src/main.py`

## API接口

- `POST /query` - 查询知识库
- `POST /add_document` - 添加文档
- `POST /chat` - 聊天对话
- `POST /upload_document` - 上传文档

## 配置说明

- LLM模型：qwen3-max
- 嵌入模型：text-embedding-v4
- 向量数据库：Milvus
- 默认端口：8000
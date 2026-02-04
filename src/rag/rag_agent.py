"""
简化版 RAG 知识问答 - 仅处理PDF文本（Graph节点+BGE中文模型+官方Milvus包）
最终版：使用 langchain-milvus 包，彻底解决兼容性问题
"""
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from datetime import datetime
import os

# 核心依赖（使用官方推荐的 langchain-milvus 包）
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import MilvusVectorStore  # 官方新版Milvus向量库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever

# 替换为你的LLM配置
from llm_db_config.chatmodel import llm_no_think

# ========== 环境配置（国内镜像+超时设置） ==========
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_CONNECT_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# ========== CUDA自动检测 ==========
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# ========== 配置类（适配新版Milvus） ==========
class SimpleRAGConfig:
    # Milvus连接配置
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: str = "19530"
    COLLECTION_NAME: str = "simple_pdf_rag_bge"  # 向量集合名
    # 嵌入模型配置（BGE中文最优模型）
    EMBEDDING_MODEL: str = "BAAI/bge-base-zh-v1.5"
    EMBEDDING_DEVICE: str = "cuda" if HAS_CUDA else "cpu"
    # 检索配置
    SEARCH_K: int = 6  # 召回文档数
    SEARCH_SCORE_THRESHOLD: float = 0.3  # 相似度阈值（0-1）
    # 文本切片配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

config = SimpleRAGConfig()

# 打印运行信息
print(f"🔧 当前运行设备：{config.EMBEDDING_DEVICE}")
if not HAS_CUDA:
    print("⚠️  未检测到CUDA，将使用CPU运行（BGE模型CPU运行速度较慢，建议安装GPU环境）")

# ========== 相关性评分函数（BGE模型专用） ==========
def cosine_similarity_score_fn(distance: float) -> float:
    """
    余弦相似度转换：Milvus L2距离 → 相似度分数（0-1）
    BGE模型向量归一化后，L2距离范围为0-2，对应相似度1.0-0.0
    """
    return 1.0 - (distance / 2.0)

# ========== RAG核心类 ==========
class SimplePDFRAGAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={
                "device": config.EMBEDDING_DEVICE,
                "trust_remote_code": True
            },
            encode_kwargs={
                "normalize_embeddings": True  # BGE必须归一化，确保相似度计算准确
            },
        )
        self.vector_store = MilvusVectorStore(
            embedding_function=self.embeddings,
            connection_args={
                "host": config.MILVUS_HOST,
                "port": config.MILVUS_PORT,
                "alias": "default"  # 连接别名（新版必填）
            },
            collection_name=config.COLLECTION_NAME,
            auto_id=True,  # 自动生成文档ID
            distance_metric="L2",  # 与BGE归一化向量兼容
            drop_old=False,  # 替代旧版overwrite：False=不删除旧集合（True=删除重建）
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": config.SEARCH_K,
                "score_threshold": config.SEARCH_SCORE_THRESHOLD,
                "relevance_score_fn": cosine_similarity_score_fn  # 显式评分函数
            },
            search_type="similarity_score_threshold",
        )
        self.document_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是设备运维助手，严格基于提供的PDF文档内容回答问题。
                - 仅使用上下文里的信息，不编造额外内容
                - 技术问题按「问题分析→解决方案→操作步骤」的结构回答
                - 若上下文无相关信息，直接回复"无法回答该问题"
                <context>
                {context}
                </context>"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.document_chain = create_stuff_documents_chain(
            self.llm,
            self.document_prompt,
            document_prompt=ChatPromptTemplate.from_messages([
                ("system", "[文档来源：{{doc.metadata.source}}] {{doc.page_content}}")
            ])
        )
        self.rag_chain = create_retrieval_chain(self.retriever, self.document_chain, rephrase_question=False) # 关闭问题重写功能

    # 加载PDF并入库
    def load_pdf_to_db(self, pdf_path: str) -> int:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists() or pdf_path.suffix != ".pdf":
            raise ValueError(f"❌ 无效PDF路径：{pdf_path}")

        print(f"📄 正在加载PDF：{pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f"✂️ PDF共{len(documents)}页，正在切片...")

        split_docs = self.text_splitter.split_documents(documents)
        print(f"✅ 切片完成：{len(split_docs)}个文档块")

        # 补充元数据
        for doc in split_docs:
            doc.metadata.update({
                "load_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content_type": "text",
                "embedding_model": config.EMBEDDING_MODEL
            })

        # 存入Milvus
        print(f"📥 正在写入Milvus集合：{config.COLLECTION_NAME}")
        self.vector_store.add_documents(split_docs)
        return len(split_docs)

    # 问答入口（适配Graph节点）
    def run(self, state: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
        messages = state.get("messages", [])
        chat_history = messages[:-1] if len(messages) > 1 else []
        user_input = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), "")

        if not user_input:
            return {"messages": [AIMessage(content="未获取到有效问题，请重新输入")]}

        print(f"🔍 检索查询：{user_input}")
        result = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        return {"messages": [AIMessage(content=result.get("answer", "无法回答该问题"))]}

    # 简化问答接口
    def ask(self, query: str, chat_history: List[Any] = None) -> str:
        chat_history = chat_history or []
        print(f"🔍 检索查询：{query}")
        result = self.rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        return result.get("answer", "无法回答该问题")

# ========== Graph节点创建函数（适配LangChain） ==========
def create_simple_rag_node(llm: Any) -> Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]]:
    rag_agent = SimplePDFRAGAgent(llm=llm)

    def rag_node(
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return rag_agent.run(state)

    return rag_node

# ========== 运行示例 ==========
if __name__ == "__main__":
    print("=== 启动RAG问答系统（BGE中文模型+Milvus向量库） ===")
    try:
        # 初始化Agent
        rag_agent = SimplePDFRAGAgent(llm=llm_no_think)

        # 加载PDF（确保learn.pdf在当前目录）
        pdf_path = "learn.pdf"
        chunk_count = rag_agent.load_pdf_to_db(pdf_path)
        print(f"✅ PDF加载完成！共{chunk_count}个文档块存入Milvus\n")

        # 测试问答
        query = "设备显示008通信故障怎么处理？"
        answer = rag_agent.ask(query)
        print(f"👤 用户：{query}")
        print(f"🤖 助手：{answer}\n")

        # Graph节点测试
        print("=== 测试Graph节点模式 ===")
        rag_node = create_simple_rag_node(llm=llm_no_think)
        state = {"messages": [HumanMessage(content=query)]}
        result_state = rag_node(state)
        print(f"🤖 Graph节点回复：{result_state['messages'][0].content}")

    except Exception as e:
        print(f"❌ 系统运行失败：{str(e)}")
        exit(1)
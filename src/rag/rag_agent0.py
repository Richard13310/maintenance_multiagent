"""
多模态 RAG 知识问答 Agent - 设备运维场景（极简版）
核心框架：文本+图片+PDF多模态检索增强生成
核心调整：所有配置内联，无中间变量，极简风格
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import base64
import io
import numpy as np

# 核心依赖
from PIL import Image
import fitz  # PyMuPDF：PDF图片提取
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from paddleocr import PaddleOCR  # 百度飞桨OCR
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_db_config.chatmodel import llm_no_think
from src.utils import get_last_user_input

# ========== 核心：多模态文档处理器（极简版） ==========
class MultiModalDocumentProcessor:
    """多模态文档处理核心逻辑：文本/图片/PDF加载→预处理→标准化"""
    def __init__(self, embeddings: HuggingFaceEmbeddings, vector_store: Milvus):
        self.embeddings = embeddings
        self.vector_store = vector_store

        # 文本切片器（配置直接内联）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
        )

        # PaddleOCR初始化（配置直接内联）
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=False,
            show_log=False
        )

    # 1. 文本处理
    def load_text_document(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)
        if file_path.suffix == ".pdf":
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        documents = [Document(
            page_content=text,
            metadata={
                "file_name": file_path.name,
                "file_path": str(file_path),
                "load_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content_type": "text",
            }
        )]
        return self.text_splitter.split_documents(documents)

    # 2. 图片处理
    def process_image(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)
        img = Image.open(file_path).convert("RGB")
        img_np = np.array(img)

        # PaddleOCR识别
        ocr_result = self.ocr.ocr(img_np, cls=True)
        ocr_text = ""
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                ocr_text += line[1][0] + "\n"
        ocr_text = ocr_text.strip() or "图片无可识别文字"

        # 图片转base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        doc = Document(
            page_content=ocr_text,
            metadata={
                "file_name": file_path.name,
                "file_path": str(file_path),
                "load_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content_type": "image",
                "image_base64": img_base64,
                "ocr_engine": "PaddleOCR",
            }
        )
        return [doc]

    # 3. PDF图片提取
    def extract_images_from_pdf(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)
        pdf_doc = fitz.open(file_path)
        image_docs = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                img_np = np.array(img)

                ocr_result = self.ocr.ocr(img_np, cls=True)
                ocr_text = ""
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        ocr_text += line[1][0] + "\n"
                ocr_text = ocr_text.strip() or "PDF图片无可识别文字"

                doc = Document(
                    page_content=ocr_text,
                    metadata={
                        "file_name": f"{file_path.name}_page{page_num+1}_img{img_idx+1}",
                        "content_type": "image",
                        "pdf_page": page_num + 1,
                        "ocr_engine": "PaddleOCR",
                    }
                )
                image_docs.append(doc)

        pdf_doc.close()
        return image_docs

    # 4. 统一入口：加载任意类型文档
    def load_document(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        # 支持的文件类型直接内联判断
        if file_ext in [".txt", ".pdf", ".docx", ".md"]:
            if file_ext == ".pdf":
                text_docs = self.load_text_document(file_path)
                image_docs = self.extract_images_from_pdf(file_path)
                return text_docs + image_docs
            else:
                return self.load_text_document(file_path)
        elif file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            return self.process_image(file_path)
        else:
            return []

    # 5. 批量入库
    def add_documents_to_db(self, file_paths: List[str]) -> int:
        total_chunks = 0
        for file_path in file_paths:
            docs = self.load_document(file_path)
            self.vector_store.add_documents(docs)
            total_chunks += len(docs)
        return total_chunks

# ========== 核心：多模态RAG Agent（极简版） ==========
class MultiModalRAGAgent:
    """多模态RAG核心类：整合Embedding→向量库→检索→生成"""
    def __init__(self, llm: Any, retriever: Optional[Any] = None):
        self.llm = llm

        # 初始化核心组件（极简风格：所有配置直接内联，无中间变量）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="openai/clip-vit-base-patch16",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"host": "localhost", "port": "19530"},
            collection_name="ev_charging_station_kb_multimodal",
            auto_id=True,
            overwrite=False,
        )

        # 初始化其他组件
        self.document_processor = MultiModalDocumentProcessor(self.embeddings, self.vector_store)
        self.retriever = retriever or self.vector_store.as_retriever(
            search_kwargs={
                "k": 8,
                "score_threshold": 0.25
            },
            search_type="similarity_score_threshold",
        )
        self.rag_chain = self._init_rag_chain()

    # 初始化RAG生成链
    def _init_rag_chain(self) -> Any:
        # 文档组合链
        document_chain = create_stuff_documents_chain(
            self.llm,
            ChatPromptTemplate.from_messages([ # 多模态Prompt
                ("system", """你是设备运维助手，基于以下上下文回答：
                <context>
                {context}
                </context>
                规则：
                1. 图片信息（标注[图片来源]）优先于文本
                2. 仅用上下文信息，不编造内容
                3. 技术问题按「问题分析→解决方案→操作步骤」回答
                """),MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
                ]),
            document_prompt=ChatPromptTemplate.from_messages([
                ("system", """
                {% if doc.metadata.content_type == 'text' %}
                [文本来源：{{doc.metadata.file_name}}] {{doc.page_content}}
                {% else %}
                [图片来源：{{doc.metadata.file_name}}] OCR内容：{{doc.page_content}}
                {% endif %}
                """)
            ])
        )

        # 完整RAG链
        return create_retrieval_chain(
            self.retriever,
            document_chain,
            rephrase_question=False
        )

    # 核心运行逻辑
    def run(self, state: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
        messages = state.get("messages", [])
        chat_history = messages[:-1] if len(messages) > 1 else []
        user_input = get_last_user_input(messages)

        result = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })

        return {"messages": [AIMessage(content=result.get("answer", "无法回答该问题"))]}

# ========== 兼容接口：创建Graph节点 ==========
def create_multimodal_rag_agent_node(llm: Any, retriever: Optional[Any] = None) -> callable:
    rag_agent = MultiModalRAGAgent(llm, retriever)
    def rag_agent_node(state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return rag_agent.run(state)
    return rag_agent_node

# ========== 学习用示例 ==========
if __name__ == "__main__":
    # 初始化Agent
    mm_rag_agent = MultiModalRAGAgent(llm_no_think)

    # 添加多模态知识库
    knowledge_files = [
        "docs/故障排查手册.pdf",
        "docs/通信故障截图.png",
        "docs/维护规范.txt"
    ]
    mm_rag_agent.document_processor.add_documents_to_db(knowledge_files)

    # 问答示例
    query = "设备显示008通信故障怎么处理？"
    state = {"messages": [HumanMessage(content=query)]}
    result = mm_rag_agent.run(state)
    print(f"用户：{query}")
    print(f"助手：{result['messages'][0].content}")
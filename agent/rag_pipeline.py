import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from  langchain_chroma import Chroma
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / 'data' / 'company_handbook.md'
VECTOR_DIR = PROJECT_ROOT / 'db' / 'chroma.db'

# 1全局单例初始化 Embedding model （bge）

print('正在加载BGE 嵌入模型 。。。。。')
# 英伟达显卡选cuda，苹果选mps
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv('LOCAL_EMBEDDING_MODEL'),
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)


def init_vector_store()->Chroma:
    """初始化向量库，如果存在则读取，不存在则切分文档生成"""
    if VECTOR_DIR.exists() and VECTOR_DIR.iterdir():#已经存在直接加载
        return Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings
        )
    print('未检测到本地向量库。并开始构建RAG检索引')

    if not DOC_PATH.exists():
        raise FileNotFoundError(f'找不到知识库{DOC_PATH}')
    with open(DOC_PATH,'r',encoding='utf-8') as f:
        markdown_text = f.read()


    # 基于markdown层级切分
    headers_to_split_on = [
        ("##",'Chapter'),
        ("###",'Section')
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)

    # 为了防止章节依然过长，再叠加一个字符集滑动窗口切分
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(md_header_splits)

    print(f'文档切分完毕，共生成{len(splits)}个语义块（chunks），正在存入数据库')

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    print(f'向量库构建完成，已经落盘再{VECTOR_DIR}')
    return vectorstore

vector_store = init_vector_store() #初始化向量数据库实例
retriever = vector_store.as_retriever(search_kwargs={'k':5})  #转换成检索其对象retriever设置召回top5的值
# retriever = retriever.as_retriever(search_kwargs={ #带阈值的做法，只返回相似度>=0.5
#
#     'score_threshold':0.5,
#     'k':20,     #候选池大小（先取20，再过滤相似度>=0.5）
# })

@tool
def search_hr_policy(query:str)->str:
    """
    搜索公司制度、差旅报销标准、假期政策、福利相关信息的必备工具。
    输出参数query必须是从员工问题中提炼出来的精准检索词
    """
    docs = retriever.invoke(query)
    if not docs:
        return '知识库中未检索到相关政策，请提示用户询问HR人工'
    # 组装召回的上下文，附带metedata 让大模型知道出自那个章节，有效降低幻觉
    context_parts = []
    for i,doc in enumerate(docs,1):
        chapter = doc.metadata.get('Chapter','未知章节')
        section = doc.metadata.get('Section','未知段落')
        context_parts.append(f'来源{i} {chapter}>{section} \n {doc.page_content}')
    merged_context = '\n'.join(context_parts)
    return f'知识库检索结果：{merged_context}'
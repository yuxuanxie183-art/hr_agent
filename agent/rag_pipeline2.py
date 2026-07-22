import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from  langchain_chroma import Chroma


from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import  BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import BaseModel,Field
from sentence_transformers import SentenceTransformer, CrossEncoder



load_dotenv()
# 0基础配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / 'data' / 'company_handbook.md'
VECTOR_DIR = PROJECT_ROOT / 'db' / 'chroma.db'


DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL_NAME')

EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL')

RERANKING_MODEL = os.getenv('RERANKER_MODEL')
# 1初始化核心模型组件

print('正在加载BBE嵌入模型...')
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print('正在加载BBE Reranker模型...')
reranker = CrossEncoder(RERANKING_MODEL,max_length=512,device='cpu')

llm = ChatOpenAI(
    model = DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7
)

#2 构建多回路的 retriever
def build_ensemble_retriever():
    """构建BM25+vector的混合检索器"""
    if not DOC_PATH.exists():
        raise FileNotFoundError(f'到不到知识库文件{DOC_PATH}')

    with open(DOC_PATH,'r',encoding='utf-8') as f:
        markdown_text = f.read()

    # 第一层基于markdown层级切分
    headers_to_split_on = [
        ("##", 'Chapter'),
        ("###", 'Section')
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)
    # 第二层按照 ['\n\n','\n']顺序递归切分
    # 第三层为了防止章节依然过长，再叠加一个字符集滑动窗口切分兜底      递归降级一般包含2层
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n','\n']
    )
    splits = text_splitter.split_documents(md_header_splits)
    print(f'文档切分完毕，共生成{len(splits)}个语义块（chunks），正在存入数据库')

    # 路线A：全文关键字检索（BM25）
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    #路线B: 向量语义检索
    if VECTOR_DIR.exists() and VECTOR_DIR.iterdir():  # 已经存在直接加载
        print('知识库构建：检测到持久化向量库，直接加载')
        vectorstore =  Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings
        )
    else:
        print('知识库构建：未检测到持久化向量库，正在生成向量数据库并落盘')
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(VECTOR_DIR),
        )
    vector_retriever = vectorstore.as_retriever(search_kwargs={'k':5})

    #混合 使用 ensembleRetriever （采用 事 RRF 倒序融合算法）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4,0.6]  #权重可调，偏向关键词还是偏向语义   40依赖关键词  60依赖语义泛化
    )
    return ensemble_retriever
print('正在构建混合检索器')
retriever = build_ensemble_retriever() #启动初始化ensemble_retriever

#3 智能扩写与 HyDE
class QueryExpansion(BaseModel):
    expanded_queries:list[str] =Field(description='从不同维度扩写3个相关检索词或短语')
    hypothetical_document:str=Field(description='针对该问题的一段假设，看似专业的官方制度回答片段（允许伪造数字）')

expansion_parser = JsonOutputParser(pydantic_object=QueryExpansion)

def expand_and_hyde(original_query:str)->list[str]:
    """利用LLM生成多维度扩写与HyDE 假设"""
    prompt = ChatPromptTemplate.from_template(
        "你是一名专业的企业 HR 专家。为了提高知识库检索命中率，请协助处理用户的原始提问。\n"
        "任务 1（多维扩展）：站在不同视角（如政策名词、审批流程、系统操作）扩展 3 个相关检索词或短语。\n"
        "任务 2（HyDE假设）：用官方、严谨的 HR 规章制度口吻，伪造一段回答该问题的文本。不管事实是否正确，重点是极度模仿'员工手册'的很专业行文风格和词汇分布。\n\n"
        "用户原始问题：{query}\n\n"
        "{format_instruction}"
    )

    chain = prompt | llm |expansion_parser

    try:
        result = chain.invoke({
            'query': original_query,
            'format_instruction': expansion_parser.get_format_instructions(),
        })
        print(f'原始问题：“{original_query}”')
        print(f'     ->衍生查询：{result['expanded_queries']}')
        print(f'     ->HyDE伪文：{result['hypothetical_document'][:30]}')

        # 汇总：原始问题 + 3个衍生问题 +1个假设文档
        return [original_query] + result['expanded_queries'] + [result['hypothetical_document']]
    except Exception as e:
        print(f'LLM 调用失败，降级使用基础检索，原因：{e}')
        return [original_query]

# 4封装工具
@tool
def search_hr_policy(query:str)->str:
    """
高级知识搜索引擎（具备自动改写、混合检索、重拍功能）。

当用户询问任何关于公司规章制度、差旅报销标准、假期政策、福利等相关信息，必须调用此工具。

输入参数 query 必须是用户原始问题

    """
    # 步骤一：获取5个查询变体组成的查询矩阵
    search_queries = expand_and_hyde(query)

    # 步骤二：多路并发检索（mb25+语义vector）
    all_condition_docs = []
    for q in search_queries:
        docs = retriever.invoke(q)
        all_condition_docs.extend(docs)
    # 步骤三：文档去重（文档内容作为唯一标识） 字典推导式实现
    unique_docs = {doc.page_content:doc for doc in all_condition_docs}.values()
    unique_docs = list(unique_docs)

    if not unique_docs:
        return '知识库中未检索到相关政策，请提示用户询问HR人工'
    # 步骤四 Cross-Encoder(交叉编码器)精准重排
    # 必须使用用户原始真实问题，去召回的文档计算相关性得分
    sentence_pairs = [[query,doc.page_content]  for doc in unique_docs]
    score = reranker.predict(sentence_pairs)

    scored_doc = list(zip(unique_docs,score))
    # 按照模型打分从高到低排序
    scored_doc.sort(key=lambda x:x[1],reverse=True)

    # 步骤五：截取真正的Top-3 并组装返回文本  _占位符不输出    切片排序过后的scored_doc
    top3_docs = [doc for doc, _ in scored_doc[:3]]

    context_parts = []
    for i, doc in enumerate(top3_docs, 1):
        chapter = doc.metadata.get('Chapter', '未知章节')
        section = doc.metadata.get('Section', '未知段落')
        context_parts.append(f'来源{i} {chapter}>{section} \n {doc.page_content}')
    merged_context = '\n'.join(context_parts)

    return f'知识库检索结果：{merged_context}'


# Actor-Critic 执行者-审计者:反思架构
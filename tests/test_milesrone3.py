import sys
from pathlib import Path

# 为了防止ModuleNotFoundError 我们加入下面两行代码 可以自动将项目根目录挂在到系统环境变量中，调高鲁棒性
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import io
from PIL import Image as PILImage
from langchain_core.messages import HumanMessage
from agent.graph_builderr import hr_agent_app

def display_graph(graph, xray=False):
    try:
        png_data  = graph.get_graph(xray=xray).draw_mermaid_png()    # 获取 LangGraph 生成的 mermaid_png 二进制数据
        image_stream = io.BytesIO(png_data)                 # 使用 io.BytesIO 将二进制数据转化为文件流
        img = PILImage.open(image_stream)                   # 使用 pillow 打开这个图片流
        print('架构图已弹窗显示，请查看屏幕')
        img.show()                                          # 调用 系统默认的图片查看器 查看弹出窗口
        img.save('my_graph.png')
    except Exception as e:
        print(f'出错：{e}')

def chat_with_agent(uid:str,question:str):
    print('==================================')
    print(f'员工uid：{uid} 提问：{question}')
    print('==================================')

    initial_state ={
        'messages':[HumanMessage(content=question)],
        'current_uid':uid,
        'loop_tep':0
    }

    # 启用流式输出方便观察内部执行路径
    for event in hr_agent_app.stream(initial_state,stream_mode='values'):
        last_msg = event['messages'][-1]
        # 过滤调系统初始输入和反思审计打回的提示，让展示日志干净一些
        if isinstance(last_msg, HumanMessage) and '[SYSTEM AUDIT FAILED]' not in last_msg.content :
            continue

        if last_msg.type =='ai' and not last_msg.tool_calls:
            print(f'\n AI最终回复：\n{last_msg.content}\n')

        if last_msg.type =='ai' and last_msg.tool_calls:
            for tool in last_msg.tool_calls:
                print(f'     调度工具   ->{tool['name']} ({tool['args']})')


if __name__ == '__main__':
    # display_graph(hr_agent_app)
    # 简单数据库操作
    print('==============简单数据库操作==============')
    chat_with_agent(uid='1002',question='帮我查一下还有几天年假？如果可以的话顺便帮我开一个证明')
    print('==============简单数据库操作 结束==============')

    # 触发RAG
    print('==============测试2==============')
    chat_with_agent(uid='1002',question='我下周要去北京出差，住宿费最高报销多少？')
    print('==============测试2 结束==============')

    # 触发RAG
    print('==============测试3==============')
    chat_with_agent(uid='1001', question='我刚刚入职两年，如果休3天事假，需要谁来审批？')
    print('==============测试3 结束==============')
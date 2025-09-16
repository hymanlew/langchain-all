from typing import List, Dict
import gradio as gr

# 自定义的包
from text2sql_app import make_graph


async def execute_graph(chat_bot: List[Dict]) -> List[Dict]:
	"""执行工作流的函数"""
	user_input = chat_bot[-1]['content']
	#AI助手的最后一条消息
	result = ''
	inputs = {
		"input": user_input
	}
	
	async with make_graph() as graph:
		async for event in graph.astream({"messages": [{"role": "user","content":"",stream_mode="values"):
			messages = event.get('messages')
			if messages:
				if isinstance(messages, list):
					#如果消息是列表，则取最后一个
					message = messages[-1]
				if message.__class__.__name__ == 'AIMessage':
					if message.content:
						print(result)
						#需要在Webui展示的消息
						result = message.content 
						
				msg_repr = message.pretty_repr(html=True)
				print(msg_repr) #输出消息的表示形式
				
	chat_bot.append({'role':"assistant', 'content':result})
	return chat_bot
	
	
def do_graph(user_input, chat_bot):
	"""输入框提交后，执行的函数"""
	if user_input:
		chat_bot.append({'role':'user', 'content': user_input})
	return'', chat_bot
	
	
css = '''
#bgc {background-color:#7FFFD4}
.feedback textarea {font-size:24px !important}
'''
with gr.Blocks(title='调用MCP服务的TEXT2SQL项目'，css=css )as instance:
	gr.Label('调用MCP服务的TEXT2SQL项目', container=False)
	chatbot = gr.Chatbot(type='messages', height=450, label='AI客服') # 聊天记录组件
	input_textbox = gr.Textbox(label='请输入你的问题'，value='') # 输入框
	input_textbox.submit(do_graph, [input_textbox, chatbot], [input_textbox, chatbot])


if __name__ == '__main__':
    # 生产环境启动
    instance.launch(**{
        "auth": Config.GRADIO_AUTH,
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "debug": True
    })
	
	
	
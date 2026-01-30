from llmlingua import PromptCompressor
llm_lingua = PromptCompressor()
compressed_prompt = llm_lingua.compress_prompt(
    context="云冈石窟位于中国北部山西省大同市西郊17公里处的武周山南麓……",
    instruction="压缩并保持主要内容",
    question="",
    target_token=100  # 设定目标token数
)
print(compressed_prompt['compressed_prompt'])


json_data = {
    "id": 987654,
    "name": "悟空",
    "biography": "鸿蒙之初，天地未分……"
}
json_config = {
    "id": {"rate": 1, "compress": False, "pair_remove": False, "value_type": "number"},
    "name": {"rate": 0.7, "compress": False, "pair_remove": False, "value_type": "string"},
    "biography": {"rate": 0.3, "compress": True, "pair_remove": False, "value_type": "string"}
}
compressed_json = llm_lingua.compress_json(json_data, json_config)
print(compressed_json['compressed_prompt'])


import pandas as pd
import requests
import time
import os
import random
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

CHAT_API_URL = ''
AUTH_TOKEN = ""
API_KEY    = ""
APP_ID     = ""
INPUT_EXCEL = r'd:/Project_my/yitushibie/intranet/新建 Microsoft Excel 工作表.xlsx'
OUTPUT_EXCEL = 'auto_eval_results.xlsx'
MAX_WORKERS = 5
CATEGORY_MAPPING = {
    "运营业务": "运营知识助手",
    "运营": "运营知识助手",
    "对公业务": "对公问答助手",
    "对公": "对公问答助手",
    "授信业务": "授信问答助手",
    "授信": "授信问答助手",
    "人力知识": "人力助手",
    "人力": "人力助手",
    "人力业务": "人力助手",
    "小微业务": "小微业务AI助手",
    "小微": "小微业务AI助手",
    "零贷业务": "零贷业务AI助手",
    "零贷": "零贷业务AI助手",
    "合规业务": "合规知识助手",
    "合规": "合规知识助手",
    "合规知识": "合规知识助手",
    "贷后业务": "信贷管理部AI助手",
    "贷后": "信贷管理部AI助手",
    "信贷管理部": "信贷管理部AI助手",
    "信贷系统": "信贷管理系统问答助手",
    "系统": "信贷管理系统问答助手",
    "信贷管理系统": "信贷管理系统问答助手",
    "信息安全": "合规知识助手"
}

def extract_agent_id(actual_answer):
    try:
        if not actual_answer:
            return "empty"
        if isinstance(actual_answer, str):
            clean_json = actual_answer
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0]
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0]
            start = clean_json.find('{')
            end = clean_json.rfind('}')
            if start != -1 and end != -1:
                clean_json = clean_json[start:end+1]
            data = json.loads(clean_json.strip())
            return data.get("selected_agent_id", "unknown")
        elif isinstance(actual_answer, dict):
            return actual_answer.get("selected_agent_id", "unknown")
    except Exception:
        return "parse_error"
    return "unknown"

def extract_top_agent_ids(actual_answer):
    try:
        res = []
        if not actual_answer:
            return res
        if isinstance(actual_answer, str):
            s = actual_answer
            if "```json" in s:
                s = s.split("```json")[1].split("```")[0]
            elif "```" in s:
                s = s.split("```")[1].split("```")[0]
            start = s.find('{')
            end = s.rfind('}')
            data = None
            if start != -1 and end != -1:
                inner = s[start:end+1]
                try:
                    data = json.loads(inner.strip())
                except Exception:
                    data = None
            if isinstance(data, dict):
                if 'selected_agent_id' in data:
                    res.append(str(data.get('selected_agent_id')))
                if 'choices' in data and isinstance(data['choices'], list):
                    for ch in data['choices'][:3]:
                        c = ch.get('message', {}).get('content', ch)
                        ids = extract_top_agent_ids(c)
                        for v in ids:
                            res.append(v)
                for key in ['top_candidates','candidates','topn_candidates','results','topn','top']:
                    if key in data and isinstance(data[key], list):
                        for item in data[key][:3]:
                            if isinstance(item, dict):
                                if 'selected_agent_id' in item:
                                    res.append(str(item['selected_agent_id']))
                                elif 'agent_id' in item:
                                    res.append(str(item['agent_id']))
                                elif '预测类别' in item:
                                    res.append(str(item['预测类别']))
                            elif isinstance(item, str):
                                res.append(item)
            elif isinstance(data, list):
                for item in data[:3]:
                    if isinstance(item, dict):
                        if 'selected_agent_id' in item:
                            res.append(str(item['selected_agent_id']))
                        elif '预测类别' in item:
                            res.append(str(item['预测类别']))
                    elif isinstance(item, str):
                        res.append(item)
            else:
                m = re.findall(r'"selected_agent_id"\s*:\s*"(.*?)"', s)
                for v in m[:3]:
                    res.append(v)
        elif isinstance(actual_answer, dict):
            if 'selected_agent_id' in actual_answer:
                res.append(str(actual_answer.get('selected_agent_id')))
            if 'choices' in actual_answer and isinstance(actual_answer['choices'], list):
                for ch in actual_answer['choices'][:3]:
                    c = ch.get('message', {}).get('content', ch)
                    ids = extract_top_agent_ids(c)
                    for v in ids:
                        res.append(v)
        seen = set()
        uniq = []
        for x in res:
            x = x.replace('"','').replace("'","").strip()
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq[:3]
    except Exception:
        return []

def calculate_metrics(results_df):
    y_true = []
    y_pred_top1 = []
    y_pred_top3 = []
    y_pred_relaxed = []
    print(f"Debug: DataFrame Columns: {results_df.columns.tolist()}")

    # 处理每一行
    for _, row in results_df.iterrows():
        raw_expected = ""
        possible_keys = ['Expected_Answer', 'Expected', 'Expection', 'Expectation', 'Answer', '问题所属知识']
        for key in possible_keys:
            # 获取每行中对应列的值
            if key in row and pd.notna(row[key]):
                val = str(row[key]).strip()
                if val:
                    raw_expected = val
                    break

        # 获取实际的助手名称
        expected = CATEGORY_MAPPING.get(raw_expected, raw_expected)
        # 回复的答案
        actual = row['Actual_Answer']
        # 提取回复答案 对应的 助手名称
        preds = extract_top_agent_ids(actual)
        predicted_top1 = preds[0] if preds else extract_agent_id(actual)

        # 拼装实际的助手
        y_true.append(expected)
        # 拼装回复的助手
        y_pred_top1.append(predicted_top1)
        y_pred_top3.append(preds)

        if expected in preds:
            y_pred_relaxed.append(expected)
        else:
            y_pred_relaxed.append(predicted_top1)

    # 真正的助手，新增/修改列
    results_df['Mapped_Expected'] = y_true
    # 预测的助手
    results_df['Extracted_Predicted'] = y_pred_top1
    results_df['Extracted_Top3'] = ['|'.join(items) if items else '' for items in y_pred_top3]
    if 'Mapped_Expected' in results_df.columns:

        # from collections import Counter
        # list1 == list2 顺序+内容
        # Counter(list1) == Counter(list2) 内容+个数
        # set(list1) == set(list2) 仅内容
        results_df['Is_Correct_Top1'] = results_df['Mapped_Expected'] == results_df['Extracted_Predicted']
        def _any_ok(row):
            tops = str(row['Extracted_Top3']).split('|') if pd.notna(row['Extracted_Top3']) else []
            # row['Mapped_Expected']
            return row['Mapped_Expected'] in tops

        # Invoke function on values of Series
        results_df['Is_Correct_Any'] = results_df.apply(_any_ok, axis=1)
    else:
        results_df['Is_Correct_Top1'] = False
        results_df['Is_Correct_Any'] = False

    total = len(results_df)
    correct_any = results_df['Is_Correct_Any'].sum()
    accuracy_any = correct_any / total if total > 0 else 0.0
    print(f"总体准确率 (Top3-Any): {accuracy_any:.2%} ({correct_any}/{total})")

    # 并集去重
    labels = sorted(list(set(y_true) | set(y_pred_relaxed)))
    metrics_data = []
    for label in labels:
        if label in {"unknown", "parse_error", "empty"}:
            continue
        tp = sum(1 for t, p in zip(y_true, y_pred_relaxed) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred_relaxed) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred_relaxed) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        metrics_data.append({
            "Category": label,
            "Precision (Top3)": round(precision, 4),
            "Recall (Top3)": round(recall, 4),
            "F1_Score (Top3)": round(f1, 4),
            "Support": support,
            "TP": tp,
            "FP": fp,
            "FN": fn
        })
    metrics_df = pd.DataFrame(metrics_data)
    if not metrics_df.empty:
        print(metrics_df.sort_values(by="F1_Score (Top3)", ascending=False).head(20).to_string(index=False))
    return results_df, metrics_df

def generate_random_id():
    return str(random.randint(10000, 99999))

def call_agent_chat(question, chat_id):
    headers = {
        'Authorization': AUTH_TOKEN,
        'api-key': API_KEY,
        'app-id': APP_ID,
        'Content-Type': 'application/json'
    }
    payload = {
        "chatId": chat_id,
        "requestId": generate_random_id(),
        "stream": False,
        "detail": False,
        "variables": {"topn": 3},
        "messages": [
            {
                "content": question,
                "role": "user"
            }
        ]
    }
    try:
        response = requests.post(CHAT_API_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            res_json = response.json()
            if 'data' in res_json:
                data = res_json['data']
                if 'message' in data and 'content' in data['message']:
                    return data['message']['content']
                elif 'choices' in data and len(data['choices']) > 0:
                     return data['choices'][0]['message']['content']
                else:
                    return str(data)
            else:
                return str(res_json)
        else:
            return f"HTTP 错误: {response.status_code} - {response.text}"
    except Exception as e:
        return f"请求异常: {e}"

def process_single_row(index, row, session_chat_id):
    question = str(row['Question']).strip()
    expected = str(row['Expected_Answer']).strip()
    if not question or "问题" in question:
        return None
    print(f"[{index+1}] 正在请求: {question}")
    actual_answer = call_agent_chat(question, session_chat_id)
    display_answer = str(actual_answer).replace('\n', ' ')
    print(f"[{index+1}] 完成: {display_answer[:30]}...")
    return {
        'Row': index + 1,
        'Question': question,
        'Expected_Answer': expected,
        'Actual_Answer': actual_answer,
        'Chat_ID': session_chat_id
    }

def main():
    print("=== 全自动评测脚本启动 (并行版) ===")
    print(f"目标接口: {CHAT_API_URL}")
    print(f"读取文件: {INPUT_EXCEL}")
    print(f"并发线程数: {MAX_WORKERS}")
    if not os.path.exists(INPUT_EXCEL):
        print("错误: 文件不存在")
        return
    try:
        try:
            df = pd.read_excel(INPUT_EXCEL)
            print(f"成功读取 Excel，列名: {df.columns.tolist()}")
            col_map = {
                '问题内容': 'Question',
                '问题所属知识': 'Expected_Answer'
            }
            if '问题内容' in df.columns and '问题所属知识' in df.columns:
                df.rename(columns=col_map, axis=1, inplace=True)
            else:
                print("未找到标准中文列名，尝试按列位置读取 (第1列=问题, 第2列=答案)...")
                df = pd.read_excel(INPUT_EXCEL, header=None)
                df.rename(columns={0: 'Question', 1: 'Expected_Answer'}, inplace=True)
                if str(df.iloc[0]['Question']).strip() in ['问题内容', 'Question']:
                    df = df.iloc[1:].reset_index(drop=True)
        except Exception as e:
            print(f"读取 Excel 失败: {e}")
            return
        results = []
        session_chat_id = generate_random_id()
        print(f"本次评测使用的会话 ID (chatId): {session_chat_id}")
        print(f"开始评测 {len(df)} 条数据...\n")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(process_single_row, index, row, session_chat_id): index
                for index, row in df.iterrows()
            }
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if len(results) % 100 == 0:
                            print(f"\n    [自动保存] 已收集 {len(results)} 条结果，正在写入 Excel...")
                            sorted_results = sorted(results, key=lambda x: x['Row'])
                            pd.DataFrame(sorted_results).to_excel(OUTPUT_EXCEL, index=False)
                            print("    [自动保存] 写入完成。\n")
                except Exception as exc:
                    print(f"任务执行异常: {exc}")
        print("\n所有任务已完成，正在保存最终结果...")
        sorted_results = sorted(results, key=lambda x: x['Row'])

        # 行列、二维的、大小可变的、可以存储多种类型数据的表格结构
        result_df = pd.DataFrame(sorted_results)
        final_df, metrics_df = calculate_metrics(result_df)
        with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
            final_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            metrics_df.to_excel(writer, sheet_name='Metrics_Summary', index=False)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

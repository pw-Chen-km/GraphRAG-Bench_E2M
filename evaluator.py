import json
from openai import OpenAI
import os
from tqdm import tqdm
import re

def load_model_outputs(base_path):
    model_predictions = {}

    for model_folder in os.listdir(base_path):
        model_folder_path = os.path.join(base_path, model_folder)

        if os.path.isdir(model_folder_path):
            predictions = {}
            for file_name in ["GraphRAG-Bench_FB", "GraphRAG-Bench_MC", "GraphRAG-Bench_MS", "GraphRAG-Bench_OE", "GraphRAG-Bench_TF"]:
                file_path = os.path.join(model_folder_path, file_name + ".json")

                if os.path.exists(file_path):
                    with open(file_path, 'r',encoding="utf-8",errors="ignore") as f:
                        # data = json.load(f)
                        # 修改了
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as e:
                            print(f"JSON 解析错误: {e}")
                            print(f"错误位置: {e.lineno} 行, {e.colno} 列")
                        prediction_list = []
                        for key, value in data.items():
                            prediction_dict = {
                                "prediction": value["prediction"],
                                "idx": int(key)
                            }
                            prediction_list.append(prediction_dict)

                        predictions[file_name.split('_')[-1]] = prediction_list

            model_predictions[model_folder] = predictions

    return model_predictions

def load_original_data(original_path):
    original_data = {}

    for file_name in ["FB", "MC", "MS", "OE", "TF"]:
        file_path = os.path.join(original_path, file_name + ".jsonl")

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                original_data[file_name] = [json.loads(line.strip()) for line in f]

    return original_data

def extract_feature(text):
    if "Please answer the following" in text:
        text = text.split("Please answer the following")[0].strip()

    parts = text.split("ANSWER:")
    if len(parts) < 2:
        return "", ""

    answer = parts[1].strip().split('\n')[0]

    rationale_text = parts[0].strip()
    if 'RATIONALE:' in rationale_text:
        rationale = rationale_text.split('RATIONALE:')[1].strip()
    else:
        rationale = rationale_text.strip()

    return [rationale, answer]

def post_processor(original_data, prediction, question_type):
    merged_data = {type_: [] for type_ in question_type}

    for type_ in question_type:

            merged_data[type_] = [
                {
                    "predict_rationale": extract_feature(pred["prediction"])[0],
                    "predict_answer": extract_feature(pred["prediction"])[1],
                    "rationale": orig["Rationale"],
                    "answer": orig["Answer"],
                    "topic": orig["Level-1 Topic"]
                }
                for orig, pred in zip(original_data[type_], prediction[type_])
            ]

    return merged_data

def rationale_calculator(pred_rationale, gold_rationale):
    prompt = f"""
        You are a strict evaluator. Compare the following two rationales for correctness and completeness:

        Predicted Rationale: {pred_rationale}

        Gold Rationale: {gold_rationale}

        Please evaluate the predicted rationale in comparison to the gold rationale. Respond with a score between 0 and 1:
        - 1: The predicted rationale fully aligns with the gold rationale.
        - 0.5: The predicted rationale is partially correct but lacks completeness or includes incorrect information.
        - 0: The predicted rationale is incorrect or completely misaligned with the gold rationale.

        only provide the score without any explanation.
        """

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=5,
        temperature=0
    )

    score_text = response.choices[0].message.content.strip()

    try:
        return float(score_text)
    except ValueError:
        return 0.0

def open_ended_calculator(pred_answer, gold_answer):
    prompt = f"""
            You are a strict evaluator. Compare the following two answers for correctness and completeness:

            Predicted Answer: {pred_answer}

            Gold Answer: {gold_answer}

            Please evaluate the predicted answer in comparison to the gold answer. Respond with a score between 0 and 1:
            - 1: The predicted answer fully aligns with the gold answer.
            - 0.5: The predicted answer is partially correct but lacks completeness or includes incorrect information.
            - 0: The predicted answer is incorrect or completely misaligned with the gold answer.

            only provide the score without any explanation.
            """

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=5,
        temperature=0
    )

    score_text = response.choices[0].message.content.strip()

    try:
        return float(score_text)
    except ValueError:
        return 0.0


def em_calculator(data):
    eval_data = []

    for item in data:
        rationale_score = rationale_calculator(pred_rationale=item["predict_rationale"],
                                               gold_rationale=item["rationale"])
        if item["predict_answer"] == item["answer"]:
            answer_score = 1.0
        else:
            answer_score = 0.0

        item["rationale_score"] = rationale_score
        item["answer_score"] = answer_score

        eval_data.append(item)

    return eval_data


def ms_calculator(data):
    eval_data = []

    for item in data:
        rationale_score = rationale_calculator(pred_rationale=item["predict_rationale"],
                                               gold_rationale=item["rationale"])

        correct_answers = set(item["answer"])
        predicted_answers = set(item["predict_answer"])

        if predicted_answers == correct_answers:
            answer_score = 1.0
        elif predicted_answers.issubset(correct_answers):
            answer_score = 0.5
        else:
            answer_score = 0.0

        item["rationale_score"] = rationale_score
        item["answer_score"] = answer_score

        eval_data.append(item)

    return eval_data

def oe_calculator(data):
    eval_data = []

    for item in data:
        rationale_score = rationale_calculator(pred_rationale=item["predict_rationale"],
                                               gold_rationale=item["rationale"])
        answer_score = open_ended_calculator(pred_answer=item["predict_answer"],
                                             gold_answer=item["answer"])

        item["rationale_score"] = rationale_score
        item["answer_score"] = answer_score

        eval_data.append(item)

    return eval_data

def evaluator(type_, data):

    if type_ == "MC" or type_ == "TF":
        eval_data = em_calculator(data)
    elif type_ == "MS":
        eval_data = ms_calculator(data)
    elif type_ == "OE" or type_ == "FB":
        eval_data = oe_calculator(data)
    else:
        print("Type Error")
        eval_data = []

    return eval_data

def calculate_ar_score(answer_score, rationale_score):
    if answer_score == 1 and rationale_score == 1:
        return 1.0
    elif answer_score == 1 and rationale_score == 0.5:
        return 0.5
    elif answer_score == 1 and rationale_score == 0:
        return 0.0
    elif answer_score == 0 and rationale_score == 1:
        return 0.5
    elif answer_score == 0 and rationale_score == 0.5:
        return 0.25
    else:
        return 0.0

def run_eval(origin_data, prediction, question_type):
    merged_data = post_processor(origin_data, prediction, question_type)

    results = {
        "total": {"rationale_score": 0.0, "answer_score": 0.0, "ar_score": 0.0, "count": 0},
        "by_type": {type_: {"rationale_score": 0.0, "answer_score": 0.0, "ar_score": 0.0, "count": 0} for type_ in question_type},
        "by_topic": {}
    }

    for type_ in tqdm(question_type, desc="Evaluating different question types"):
        eval_data = evaluator(type_, merged_data[type_])

        for item in eval_data:
            rationale_score = item["rationale_score"]
            answer_score = item["answer_score"]

            results["total"]["rationale_score"] += rationale_score
            results["total"]["answer_score"] += answer_score
            ar_score = calculate_ar_score(answer_score, rationale_score)
            results["total"]["ar_score"] += ar_score
            results["total"]["count"] += 1

            results["by_type"][type_]["rationale_score"] += rationale_score
            results["by_type"][type_]["answer_score"] += answer_score
            results["by_type"][type_]["ar_score"] += ar_score
            results["by_type"][type_]["count"] += 1

            topic = item["topic"]
            if topic not in results["by_topic"]:
                results["by_topic"][topic] = {"rationale_score": 0.0, "answer_score": 0.0, "ar_score": 0.0, "count": 0}
            results["by_topic"][topic]["rationale_score"] += rationale_score
            results["by_topic"][topic]["answer_score"] += answer_score
            results["by_topic"][topic]["ar_score"] += ar_score
            results["by_topic"][topic]["count"] += 1

    if results["total"]["count"] > 0:
        results["total"]["rationale_score"] /= results["total"]["count"]
        results["total"]["answer_score"] /= results["total"]["count"]
        results["total"]["ar_score"] /= results["total"]["count"]

    for type_ in results["by_type"]:
        if results["by_type"][type_]["count"] > 0:
            results["by_type"][type_]["rationale_score"] /= results["by_type"][type_]["count"]
            results["by_type"][type_]["answer_score"] /= results["by_type"][type_]["count"]
            results["by_type"][type_]["ar_score"] /= results["by_type"][type_]["count"]

    for topic in results["by_topic"]:
        if results["by_topic"][topic]["count"] > 0:
            results["by_topic"][topic]["rationale_score"] /= results["by_topic"][topic]["count"]
            results["by_topic"][topic]["answer_score"] /= results["by_topic"][topic]["count"]
            results["by_topic"][topic]["ar_score"] /= results["by_topic"][topic]["count"]

    return results

if __name__ == "__main__":

    client = OpenAI(api_key="")

    question_type = ["FB", "MC", "MS", "OE", "TF"]

    original_path = "/"
    original_data = load_original_data(original_path)

    base_path = "/"
    predictions = load_model_outputs(base_path)

    output_file_path = "/"

    with open(output_file_path, 'r', encoding='utf-8',errors="ignore") as outfile:
        data = json.load(outfile)

    for model_name, prediction in tqdm(predictions.items(), desc="Evaluating Models", unit="model"):

        if model_name in data:
            print(f"Skipping {model_name}, already evaluated.")
            continue

        results = run_eval(original_data, prediction, question_type)

        data[model_name] = results

        with open(output_file_path, 'w') as outfile:
            json.dump(data, outfile)
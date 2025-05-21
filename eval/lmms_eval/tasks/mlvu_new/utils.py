import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger
import hmac
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
import hashlib
import requests
hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
# hf_home="/share/junjie/shuyan/lmms-eval/~/.cache/huggingface"
base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mlvu_dev.yaml", "r") as f:
    raw_data_dev = f.readlines()
    safe_data_dev = []
    for i, line in enumerate(raw_data_dev):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_dev.append(line)
cache_name_dev = yaml.safe_load("".join(safe_data_dev))["dataset_kwargs"]["cache_dir"]
cache_dir_dev = os.path.join(base_cache_dir, cache_name_dev)


with open(Path(__file__).parent / "mlvu_test.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_test.append(line)
cache_name_test = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
cache_dir_test = os.path.join(base_cache_dir, cache_name_test)



NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = "gpt-4-turbo"#"gpt-4-turbo"#config["metadata"]["gpt_eval_model_name"]

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_ID = os.getenv("OPENAI_API_ID", "YOUR api id")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    SOURCE = os.getenv("SOURCE", 'Your source')
    configs = {
        'appid': API_ID,
        'appkey': API_KEY,
        'source': "webpage_image_gpt4v",
        "apiurl": API_URL
    }

def mlvu_doc_to_visual_dev(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir_dev, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_visual_test(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir_test, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # option_prompt="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question." without parentheses
    option_prompt = ""
    question = doc["question"] + "\nAnswer with the best option's letter from the given choices directly."
    full_prompt = option_prompt + "\n" + question + "\n" + "Best option: ("
    return full_prompt

def mlvu_sum_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def extract_characters_regex(s):
    s = s.strip()
    if ")" in s:
        index = s.index(")")
        pred = s[index - 1 : index]
        return pred
    else:
        return s


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]

    pred_ans = extract_characters_regex(pred)

    task_type = doc["task_type"]
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"mlvu_percetion_score": data_dict}


def mlvu_sum_process_results_generic(doc, result):
    pred = result[0]
    doc["pred"] = pred
    eval_results = gpt_eval(doc)

    return {
        "gpt_eval_score": {"video_name": doc["video_name"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "score": eval_results["score"], "review": eval_results["review"]},
    }

def mlvu_sub_scene_process_results_generic(doc, result):
    
    pred = result[0]
    doc["pred"] = pred
    eval_results = gpt_eval_sub_scene(doc)

    return {
        "gpt_eval_score": {"video_name": doc["video_name"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "score": eval_results["score"], "review": eval_results["review"]},
    }

def mlvu_aggregate_results_dev(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needle", "order", "plotQA", "topic_reasoning"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Calculate and log average accuracy across all task categories
    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy


def mlvu_aggregate_results_test(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needleQA", "order", "plotQA", "sportsQA", "topic_reasoning", "tutorialQA"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Calculate and log average accuracy across all task categories
    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy

def calcAuthorization(config):
    source = config['source']
    appkey = config['appkey']
    timestamp = int(time.time())
    signStr = "x-timestamp: %s\nx-source: %s" % (timestamp, source)
    sign = hmac.new(appkey.encode('utf-8'), signStr.encode('utf-8'), hashlib.sha256).digest()
    return sign.hex(), timestamp

def get_eval_generic(question, answer, pred, max_tokens: int, retries: int = 5):
    global configs
    auth, timestamp = calcAuthorization(configs)
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {api_key}"
        "x-appid": configs["appid"],
        "x-source": configs["source"],
        "x-timestamp": str(timestamp),
        "x-authorization": auth,
    }

    messages = [
        {
            "role": "system",
            "content": 
            """
                ##TASK DESCRIPTION: 
                You are required to evaluate the performance of the respondent in the video summarization task based on the standard answer and the respondent's answer. You should provide two scores. The first is the COMPLETENESS score, which should range from 1 to 5. The second is the RELIABILITY score, which should also range from 1 to 5. Below are the criteria for each scoring category:
                ##COMPLETENESS Scoring Criteria:
                The completeness score focuses on whether the summary covers all key points and main information from the video. 
                Score 1: The summary hardly covers any of the main content or key points of the video.
                Score 2: The summary covers some of the main content and key points but misses many.
                Score 3: The summary covers most of the main content and key points.
                Score 4: The summary is very comprehensive, covering most to nearly all of the main content and key points.
                Score 5: The summary completely covers all the main content and key points of the video.
                ##RELIABILITY Scoring Criteria:
                The reliability score evaluates the correctness and clarity of the video summary. It checks for factual errors, misleading statements, and contradictions with the video content. If the respondent's answer includes details that are not present in the standard answer, as long as these details do not conflict with the correct answer and are reasonable, points should not be deducted.
                Score 1: Contains multiple factual errors and contradictions; presentation is confusing.
                Score 2: Includes several errors and some contradictions; needs clearer presentation.
                Score 3: Generally accurate with minor errors; minimal contradictions; reasonably clear presentation.
                Score 4: Very accurate with negligible inaccuracies; no contradictions; clear and fluent presentation.
                Score 5: Completely accurate with no errors or contradictions; presentation is clear and easy to understand.
                ----
                ##INSTRUCTION:
                1. Evaluate COMPLETENESS: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
                2. Evaluate RELIABILITY: First, analyze the respondent's answer according to the scoring criteria, then provide an integer score between 1 and 5 based on sufficient evidence. 
                3. Output Scores in JSON Format: Present the scores in JSON format as follows:
                {'score_completeness': score_comp, 'score_reliability': score_reli, 'total_score': score_comp + score_reli}
            """
        },
        {
            "role": "user",
            "content": f"""
                Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                Standard Answer: {answer}
                Respondent's Answer: {pred}
            """
        }
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "cid": "guangtwang"
        # "response_format": {"type": "json_object"},
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(configs["apiurl"], headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response.json()['response']
            if content != "":
                
                return content,  response_data['detail']['model']
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        if "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt." in json.loads(response.content)["error"]["message"]:
            eval_logger.error(f"Repetitive patterns in prompt. Drop this data.")
            return "", ""

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed.")
            return "", ""

    return "", ""

def get_eval_generic_sub_scene(question, scoring_points, pred, max_tokens: int, retries: int = 5):
    global configs
    auth, timestamp = calcAuthorization(configs)
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {api_key}"
        "x-appid": configs["appid"],
        "x-source": configs["source"],
        "x-timestamp": str(timestamp),
        "x-authorization": auth,
    }

    messages = [
        {
            "role": "system",
            "content": 
            """
                ##TASK DESCRIPTION: 
                You are required to evaluate a respondent's answer based on a provided question, some scoring points, and the respondent's answer. You should provide two scores. The first is the accuracy score, which should range from 1 to 5. The second is the relevance score, which should also range from 1 to 5. Below are the criteria for each scoring category.
                ##ACCURACY Scoring Criteria: 
                Evaluate the respondent's answer against specific scoring points as follows:
                Score 1: The response completely misses the scoring point.
                Score 3: The response mentions content related to the scoring point but is not entirely correct.
                Score 5: The response accurately addresses the scoring point.
                Calculate the average score across all scoring points to determine the final accuracy score.
                ##RELEVANCE Scoring Criteria:
                Assess how the respondent's answer relates to the original question:
                Score 1: The response is completely off-topic from the question.
                Score 2: The response is partially related to the question but contains a significant amount of irrelevant content.
                Score 3: The response primarily addresses the question, but the respondent seems uncertain about their own answer.
                Score 4: The response mostly addresses the question and the respondent appears confident in their answer.
                Score 5: The response is fully focused on addressing the question with no irrelevant content and demonstrates complete certainty.
                ----
                ##INSTRUCTION:
                1. Evaluate Accuracy: First, assess and score each scoring point based on the respondent's answer. Calculate the average of these scores to establish the final accuracy score. Provide a detailed rationale before assigning your score.
                2. Evaluate RELEVANCE: Assess the relevance of the respondent’s answer to the question. Note that when evaluating relevance, the correctness of the answer is not considered; focus solely on how relevant the answer is to the question. Provide a comprehensive rationale before assigning your score.
                3. Output Scores in JSON Format: Present the scores in JSON format as follows:
                {'score_accuracy': score_acc, 'score_relevance': score_rele, 'total_score': score_acc + score_rele}
            """
        },
        {
            "role": "user",
            "content": f"""
                Please score the respondent's answer according to the steps in the Instructions. You must end with a JSON dict to store the scores.
                Question: {question}
                Scoring Points: {scoring_points}
                Respondent's Answer: {pred}
            """
        }
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "cid": "guangtwang"
        # "response_format": {"type": "json_object"},
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(configs["apiurl"], headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response.json()['response']
            if content != "":
                
                return content,  response_data['detail']['model']
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        if "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt." in json.loads(response.content)["error"]["message"]:
            eval_logger.error(f"Repetitive patterns in prompt. Drop this data.")
            return "", ""

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed.")
            return "", ""

    return "", ""

def parse_score(text):
    # Define the keys to locate in the text
    keys = ["score_completeness", "score_reliability"]
    scores = []

    for key in keys:
        # Find the index where each key starts
        start_index = text.find(key)
        if start_index == -1:
            continue  # Skip if key is not found

        # Find the start of the number which is after the colon and space
        start_number_index = text.find(":", start_index) + 2
        end_number_index = text.find(",", start_number_index)  # Assuming the number ends before a comma

        # Extract and convert the number to float
        score = float(text[start_number_index:end_number_index])
        scores.append(score)

    return scores

def parse_score_sub_scene(text):
    # Define the keys to locate in the text
    keys = ["score_accuracy", "score_relevance"]
    scores = []

    for key in keys:
        # Find the index where each key starts
        start_index = text.find(key)
        if start_index == -1:
            continue  # Skip if key is not found

        # Find the start of the number which is after the colon and space
        start_number_index = text.find(":", start_index) + 2
        end_number_index = text.find(",", start_number_index)  # Assuming the number ends before a comma

        # Extract and convert the number to float
        score = float(text[start_number_index:end_number_index])
        scores.append(score)

    return scores


def gpt_eval(data_dict):
    evaluated_results = []

    try:
        question = data_dict["question"]
        answer = data_dict["answer"]
        pred = data_dict["pred"]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval_generic(question, answer, pred, 1024)
        score = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = ""
        score = 0

    # Update the dictionary with the new entries
    updated_dict = {
        "video_name": data_dict["video_name"],
        "review": review,
        "score": score,
    }

    return updated_dict
def gpt_eval_sub_scene(data_dict):
    evaluated_results = []
    
    try:
        question = data_dict["question"]
        scoring_points = data_dict["scoring_points"]
        pred = data_dict["pred"]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval_generic_sub_scene(question, scoring_points, pred, 1024)
       
        score = parse_score_sub_scene(review)
        
    except Exception as e:
        eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = ""
        score = 0

    # Update the dictionary with the new entries
    
    updated_dict = {
        "video_name": data_dict["video_name"],
        "review": review,
        "score": score,
    }

    return updated_dict

def mlvu_sum_aggregate_score(results, args):
    score = 0
    accu = 0
    rele = 0
    # import pdb; pdb.set_trace()
    for result in results:
        eval_score = result["score"]
        accu += eval_score[0]
        rele += eval_score[1]
    accu = accu/ len(results)
    rele = rele/ len(results)
    total= (accu + rele ) 
    return total


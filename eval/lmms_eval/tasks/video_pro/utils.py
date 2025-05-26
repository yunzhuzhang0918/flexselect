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
base_cache_dir = os.path.expanduser(hf_home)

SUB_CATEGORIES = ["Segment QA", "NIAH QA", "Counting Problem", "Action Recognition", "Attribute Perception", "Object Reasoning", "Temporal Reasoning", "Plot Reasoning", "Entity Recognition", "Key Info Retrieval", "Event Understanding", "Others", "Egocentric Reasoning"]

QA_TYPES = ["Local Perception", "Local Reasoning", "Holistic Perception", "Holistic Reasoning"]

with open(Path(__file__).parent / "videopro_mcq.yaml", "r") as f:
    raw_data_mcq = f.readlines()
    safe_data_mcq = []
    for i, line in enumerate(raw_data_mcq):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_mcq.append(line)
cache_name_mcq = yaml.safe_load("".join(safe_data_mcq))["dataset_path"]
cache_dir_mcq = os.path.join(cache_name_mcq, "videos")


with open(Path(__file__).parent / "videopro_oe.yaml", "r") as f:
    raw_data_oe = f.readlines()
    safe_data_oe = []
    for i, line in enumerate(raw_data_oe):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_oe.append(line)
cache_name_oe = yaml.safe_load("".join(safe_data_oe))["dataset_path"]
cache_dir_oe = os.path.join(cache_name_oe, "videos")



NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = "gpt-4o-2024-08-06"#"gpt-4-turbo"#config["metadata"]["gpt_eval_model_name"]

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

def videopro_doc_to_visual_mcq(doc):
    video_path = doc["video"]
    video_path = os.path.join(cache_dir_mcq, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def videopro_doc_to_visual_oe(doc):
    video_path = doc["video"]
    video_path = os.path.join(cache_dir_oe, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def videopro_doc_to_text_mcq(doc, lmms_eval_specific_kwargs=None):
    # option_prompt="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question." without parentheses
    options = ' '.join(doc["options"])
    full_prompt ='\n'.join([
        'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option, with no text around it.',
        doc["question"], options
    ])
    return full_prompt

def videopro_doc_to_text_oe(doc, lmms_eval_specific_kwargs=None):
    # option_prompt="Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question." without parentheses
    full_prompt = doc["question"] + ' Keep the answer short and concise.'
    return full_prompt


def option_judge(response):
    # response = response.lower()
    if "the answer is" in response:
        response = response.split("the answer is")[-1].strip()
    elif "answer:" in response:
        response = response.split("answer:")[-1].strip()
    elif "the option is" in response:
        response = response.split("the option is")[-1].strip()
    for char in response:
        if char.isalpha():
            response = char
            break
    return response



def videopro_mcq_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]

    pred_ans = option_judge(pred)

    qa_type = doc["qa_type"]
    qa_subtype = doc["qa_subtype"]
    data_dict = {"question_id": doc["question"], "qa_type": qa_type, "qa_subtype": qa_subtype, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"videopro_percetion_score": data_dict}


def videopro_oe_process_results(doc, result):
    pred = result[0]
    doc["pred"] = pred
    eval_results = gpt_eval(doc)
    qa_type = doc["qa_type"]
    qa_subtype = doc["qa_subtype"]
    return {
        "gpt_eval_score": {"video": doc["video"], "question": doc["question"], "answer_text": doc["answer_text"], "pred": pred, "grade": eval_results["grade"], "qa_type": qa_type, "qa_subtype": qa_subtype,}
    }

def vidoepro_mcq_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    
    category2score = {}
    for qa_type in QA_TYPES:
        for sub_type in SUB_CATEGORIES:
            key = f"{qa_type}_{sub_type}"
            category2score[key] = {"correct": 0, "answered": 0}
    for result in results:
        qa_type = result["qa_type"]
        sub_type = result["qa_subtype"]
        key = f"{qa_type}_{sub_type}"
        category2score[key]["answered"] += 1
        response = result["answer"]
        gt = result["pred_answer"]
        category2score[key]["correct"] += (response[0] == gt or response[0] == gt.upper() if len(response) > 0 else False)
    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in QA_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    for task_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Sub Categories: {task_cate}: {accuracy:.1f}%")
    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
    

def calcAuthorization(config):
    source = config['source']
    appkey = config['appkey']
    timestamp = int(time.time())
    signStr = "x-timestamp: %s\nx-source: %s" % (timestamp, source)
    sign = hmac.new(appkey.encode('utf-8'), signStr.encode('utf-8'), hashlib.sha256).digest()
    return sign.hex(), timestamp

def get_eval_generic(question, target, predicted_answer, max_tokens: int, retries: int = 5):
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
            "role": "user",
            "content": 
            f"""
                Your job is to look at a question generated from the video, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]. First, I will give examples of each grade, and then you will grade a new example. The following are examples of CORRECT predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia Obama and Sasha Obama Predicted answer 1: sashaand maliaobama Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001. ``` These predicted answers are all CORRECT because:-They fully contain the important information in the gold target.-They do not contain any information that contradicts the gold target.-Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.-Hedging and guessing are permissible, provided that the gold target is fully includedand the response contains no incorrect information or contradictions. The following are examples of INCORRECT predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia and Sasha Predicted answer 1: Malia. Predicted answer 2: Malia, Sasha, and Susan. Predicted answer 3: Barack Obama does not have any children. Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia. Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children. Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer? Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information. ``` These predicted answers are all INCORRECT because:-A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'mnot sure, i think") are also considered incorrect. The following are examples of NOT_ATTEMPTED predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia and Sasha Predicted answer 1: I don't know. Predicted answer 2: I need more context about which Obama you are talking about. Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children. Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one. ``` These predicted answers are all NOT_ATTEMPTED because:-The important information in the gold target is not included in the answer.-No statements in the answer contradict the gold target.
                Also note the following things:-For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". -Predicted answers "120k", "124k", and 115k" are all CORRECT. -Predicted answers "100k" and "113k" are INCORRECT. -Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.-The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.-For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.-Do not punish predicted answers if they omit information that would be clearly inferred from the question.-For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".-Consider the question "What award did A pretrainer'sguide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.-For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.-For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.-Do not punish for typos in people's name if it's clearly the same name. -For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "HyoongWon Choong", "HyungwonChung", or "Hyun Won Chung". 
                Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer. 
                ```
                Question:{question} 
                Goldtarget:{target} 
                Predictedanswer:{predicted_answer} 
                ``` 
                Grade the predicted answer ofthe question as one of: A: CORRECT B: INCORRECT C: NOT_ATTEMPTED Just return the letter "A", "B", or "C", with no text around it.
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
                return content, response_data['detail']['model']
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




def gpt_eval(data_dict):
    evaluated_results = []

    try:
        question = data_dict["question"]
        answer = data_dict["answer_text"]
        pred = data_dict["pred"]
        # Assume get_eval returns a review and the model name, and parse_score parses this review
        grade, model_name = get_eval_generic(question, answer, pred, 5)
        
    except Exception as e:
        eval_logger.error(f"Error for Video: {data_dict.get('video', 'Unknown')}: {e}")
        model_name = ""
        grade = "C Failed to Get a Proper Grade"

    # Update the dictionary with the new entries
    updated_dict = {
        "video": data_dict["video"],
        "grade": grade,
    }

    return updated_dict


def vidoepro_oe_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    
    category2score = {}
    for qa_type in QA_TYPES:
        for sub_type in SUB_CATEGORIES:
            key = f"{qa_type}_{sub_type}"
            category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        qa_type = result["qa_type"]
        sub_type = result["qa_subtype"]
        key = f"{qa_type}_{sub_type}"
        category2score[key]["answered"] += 1
        response = result["grade"]
        category2score[key]["correct"] += (response[0].upper() == "A")
    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in QA_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    for task_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Sub Categories: {task_cate}: {accuracy:.1f}%")
    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0


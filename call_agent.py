import argparse
import openai

import pandas as pd

from tqdm import tqdm
from IPython import embed
import time
from random import random, uniform
import warnings
import os
import requests
import ast

warnings.filterwarnings("ignore")

# User inputs:
# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
# OR 
# Load your API key manually:
# openai.api_key = API_KEY

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

def ask_agent(model,history,vicuna_server=None):
    count = 0
    # embed()
    try:
        count+=1
        # time.sleep(10)
        if model in ['gpt-3.5-turbo','gpt-4','gpt-4-0314','gpt-4-visual','gpt-4-vision-preview']:
            # params = {"model":model, "messages":state.history, "api_key":api_key, "organization":api_organization, "headers": {"Openai-Version": "2020-11-07"}}
            params = {
            "model": model,
            "messages": history,
            # "headers": {"Openai-Version": "2020-11-07"},
            "max_tokens": 4096,
            # "temperature": 0.0000000000001
            }
            r = openai.ChatCompletion.create(**params)
            resp = r['choices'][0]['message']['content']
            # costFactor = [0.03, 0.06] if model == 'gpt-4' else [0.002, 0.002]
            # history.append({'role': 'assistant', 'content': resp})
        elif model in ['gemini']:
            print("!!")
            embed()
            url = 'http://localhost:8080/v1/chat/completions'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer AIzaSyC9OMp51SjV92mfU9Xcx2KpalHls64GvXI'
            }
            image_count = 0
            for iter in history:
                for item in iter['content']:
                    if isinstance(item, dict):
                        if 'image_url' in item.values():
                            image_count+=1
                            break
                    else:
                        if 'image_url' in item:
                            image_count+=1
                            break
            if image_count==0:                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": history,
                    "temperature": 0
                }   
            else:
                embed()
                data = {
                    "model": "gpt-4-vision-preview",
                    "messages": history,
                    "temperature": 0
                } 
            print("##")  
            response = requests.post(url, headers=headers, json=data)
            r = ast.literal_eval(response.text)
            if 'code' in r:
                resp = ask_model(model,history,vicuna_server)
            else: 
                resp = r['choices'][0]['message']['content']
        else:
            print(f"Unrecognize model name: {model}")
            return 0
    except Exception as e:
        print(e)
        # embed()
        if ('gpt-4-vis' in model) and ((e.http_status==429) or (e.http_status==502) or (e.http_status==500)) :
            # time.sleep(10)
            time.sleep(60+10*random())
            if count < 25:
                resp = ask_model(model,history,vicuna_server)
            else: return e
        elif ('gpt-4-vis' in model) and (e.http_status==400):
            if (len(history) == 4) or (len(history) == 2):
                return e
            else:
                resp = ask_model(model,history[:-2],vicuna_server)
    return resp
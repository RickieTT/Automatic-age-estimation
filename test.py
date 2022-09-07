import json
import time
import re
import os
import datetime
from time import sleep

from Crypto.Cipher import AES
from Crypto.Util import Counter
from bs4 import BeautifulSoup
import hashlib
import requests

def send_message(title, message):
    url = 'http://www.pushplus.plus/send'
    payload = {
        # 'token': '057072b6249846f3b5c5174acdd2da05',
        'token': '057072b6249846f3b5c5174acdd2da05',
        'title': title,
        'content': message,
        # 'topic' : 'book_demo',
    }
    resp = requests.post(url, data=json.dumps(payload))
    if resp.status_code == 200:
        print('发送成功', resp.text)

def book_hotel_panda():
    token_json = get_panda_token()
    server_timestamp = int(time.time()) - token_json['offset']
    token = '{timestamp}{token}'.format(token=token_json['token'], timestamp=server_timestamp)
    token = hashlib.sha512(token.encode('utf-8')).hexdigest()
    url = 'https://www.book-secure.com/api.php?_quotation'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Referer': 'https://www.book-secure.com/',
        'X-Requested-With': 'XMLHttpRequest'
    }
    payload = {
        'token': token,
        'json_body': json.dumps(
            {"c": "quotation", "property": "hknew00001", "arrival": "2022-08-03", "departure": "2022-08-10",
             "adults": 1, "children": 0, "currency": "EUR", "bestRate": False, "mcmEnable": True, "isMobile": False,
             "loyaltyTeasing": True, "context": {"totalNbRoomsWanted": 1, "relevantDataFromSelectedPackages": []}}
        )
    }

    try:
        resp = requests.post(url, data=payload, headers=headers)
        print(resp.text)
        if resp.status_code == 200:
            result = resp.json()["data"]["data"]["packages"]
            # print(result)
            if len(result) != 0:
                print('悦来酒店可以入住')
                # send_message('悦来酒店预定提醒 2022-08-03 悦来酒店可以预定，请尽快预定\n执行时间：{}'.format(time.time()))
                return True
    except Exception as e:
        print(e)
    print('2022-08-03 悦来酒店不可预定', datetime.datetime.now())
    return False


def get_panda_token():
    token = ''
    with open('token.json', 'r') as f:
        token = f.read()
    if len(token) > 0:
        return json.loads(token)

    url = 'https://www.book-secure.com/api.php?_undefined'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Referer': 'https://www.book-secure.com/',
        'X-Requested-With': 'XMLHttpRequest'
    }
    payload = {
        '_sync': True
    }
    resp = requests.post(url, data=payload, headers=headers)
    if resp.status_code == 200:
        resp = resp.json()['data']
        time_offset = int(time.time()) - int(resp['time'])
        resp['offset'] = time_offset
        with open('token.json', 'w') as f:
            json.dump(resp, f, ensure_ascii=False, indent=4)
        return resp



if __name__ == '__main__':
    # book_hotel_panda()
    max_try = 6
    step = 0
    while True:
        is_success = book_hotel_panda()
        if is_success:
            print('成功推送', step + 1)
            step += 1
            if step >= max_try:
                break
        sleep(10)
    # print(time.time())
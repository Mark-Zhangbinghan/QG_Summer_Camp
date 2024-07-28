import requests

API_KEY = 'gsw6fkz9r4ax88bodu9ffjaa'

if __name__ == '__main__':
    params = {
        'model': 'taichu_vqa',
        'messages': [{"role": "user", "question": "你好",
                      "picture": r"C:\Users\24468\Desktop\python练习\cassava-leaf-disease-classification\train_images\6103.jpg",
                      "api_key": "gsw6fkz9r4ax88bodu9ffjaa"
                      }],
        'stream': False
    }
    api = 'https://ai-maas.wair.ac.cn/maas/v1/chat/completions'
    headers = {'Authorization': 'Bearer gsw6fkz9r4ax88bodu9ffjaa'}
    response = requests.post(api, json=params, headers=headers)
    if response.status_code == 200:
        print(response.json())
    else:
        body = response.content.decode('utf-8')
        print(f'request failed,status_code:{response.status_code},body:{body}')

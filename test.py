import requests
import time

'''
TODO: Include this test in base 

your localhost url. If running on port 5000
'''

BASEURL = "http://127.0.0.1:5000/"
URL_PREDICT = f"{BASEURL}predict"
IMAGE_PATH = "test_image.jpg"

def test_ok():
    print("runing ok")
    starttime = time.time()
    results = requests.get(BASEURL)
    print("time taken:", time.time() - starttime)
    print(results.text)

def test_predict_default(url=URL_PREDICT, image_path=IMAGE_PATH, model='flowers'):
    # Path to image file
    files = {"image": open(image_path, "rb"), "model":model}
    starttime = time.time()
    # TODO: include model in request
    results = requests.post(url, files=files)
    print("time taken:", time.time() - starttime)
    print(results.text)

def test_predict_custom():
    test_predict_default(url=URL_PREDICT, image_path=IMAGE_PATH, model='flowers')

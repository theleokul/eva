import argparse as ap


import requests
import json




addr = 'http://localhost:5000'
test_url = addr + '/api?path=/home/nicholas/Videos/video.mp4'

# prepare headers for http request
# content_type = 'image/jpeg'
# headers = {'content-type': content_type}

# img = cv2.imread('lena.png')
# encode image as jpeg
# _, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url)

# decode response
response = json.loads(response.text)

with open('C:\xampp\htdocs\EVA\dist\results.txt', 'w') as f:
    f.write(response)

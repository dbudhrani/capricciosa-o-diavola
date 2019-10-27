from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from fastai import *
from fastai.vision import *
from pathlib import Path
from io import BytesIO
import torch
import sys, os
import uvicorn
import aiohttp

async def get_bytes(url):
  async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        return await response.read()

app = Starlette()

path = Path('models/')
classes = ['capricciosa', 'diavola']
data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = load_learner(path, file='capricciosa-o-diavola.pkl')

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        Is your pizza a capricciosa or a diavola?
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)

@app.route('/classify-url', methods=['GET'])
async def classify_url(request):
  bytes = await get_bytes(request.query_params['url'])
  return predict_image_from_bytes(bytes)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = learn.predict(img)
    confidence = '{}%'.format(torch.round(((outputs[pred_idx] * 100) * 10**2) / 10**2))

    return HTMLResponse(
        """
        <html>
           <body>
             <p>Prediction: <b>%s</b></p>
             <p>Confidence: %s</p>
           </body>
        </html>
    """ % (classes[pred_idx], confidence))

if __name__ == '__main__':
  if 'serve' in sys.argv:
    port = int(os.environ.get('PORT', 8008))
    uvicorn.run(app, host='0.0.0.0', port=port)


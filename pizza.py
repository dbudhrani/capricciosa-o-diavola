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

path = Path('/models')
classes = ['capricciosa', 'diavola']
data = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34)

@app.route('/classify-url', methods=['GET'])
async def classify_url(request):
  bytes = await get_bytes(request.query_params['url'])
  return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = learn.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(learn.data.classes, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    
    return HTMLResponse(
        """
        <html>
           <body>
             <p>Prediction: <b>%s</b></p>
             <p>Confidence: %s</p>
           </body>
        </html>
    """ % (classes[pred_idx], pred_probs))

if __name__ == '__main__':
  if 'serve' in sys.argv:
    port = int(os.environ.get('PORT', 8008))
    uvicorn.run(app, host='0.0.0.0', port=port)


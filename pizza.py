from starlette.applications import Starlette
from starlette.responses import JSONResponse
from fastai.vision import ImageDataBunch, create_cnn, open_image, get_transforms, models, imagenet_stats
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp

async def get_bytes(url):
  async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        return await response.read()

app = Starlette()

path = Path('/models')
classes = ['capricciosa', 'diavola']
data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34)

@app.route('/classify-url', methods=['GET']):
async def classify_url(request):
  bytes = await get_bytes(request.query_params['url'])
  return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    losses = img.predict(learner)
    return JSONResponse({
        "predictions": sorted(
            zip(cat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })

if __name__ == '__main__':
  if 'serve' in sys.argv:
    uvicorn.run(app, host='0.0.0.0', port=8008)


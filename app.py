import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = RealESRGAN(device, scale=2)
model2.load_weights('weights/RealESRGAN_x2.pth', download=True)
model4 = RealESRGAN(device, scale=4)
model4.load_weights('weights/RealESRGAN_x4.pth', download=True)
model8 = RealESRGAN(device, scale=8)
model8.load_weights('weights/RealESRGAN_x8.pth', download=True)


def inference(image, size):
    global model2
    global model4
    global model8
    if image is None:
        raise gr.Error("Image not uploaded")
        
    width, height = image.size
    if width >= 5000 or height >= 5000:
        raise gr.Error("The image is too large.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if size == '2x':
        try:
            result = model2.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model2 = RealESRGAN(device, scale=2)
            model2.load_weights('weights/RealESRGAN_x2.pth', download=False)
            result = model2.predict(image.convert('RGB'))
    elif size == '4x':
        try:
            result = model4.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model4 = RealESRGAN(device, scale=4)
            model4.load_weights('weights/RealESRGAN_x4.pth', download=False)
            result = model2.predict(image.convert('RGB'))
    else:
        try:
            result = model8.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model8 = RealESRGAN(device, scale=8)
            model8.load_weights('weights/RealESRGAN_x8.pth', download=False)
            result = model2.predict(image.convert('RGB'))
            
    print(f"Image size ({device}): {size} ... OK")
    return result


title = "Real ESRGAN UpScale Webui: 2x 4x 8x"
description = "这是 Real-ESRGAN 的非官方演示。 缩放照片的分辨率。 与原始版本相比，该模型在面部上显示出更好的结果 | <a href='https://huggingface.co/sberbank-ai/Real-ESRGAN' target='_blank'>Model card</a><div>"


gr.Interface(inference,
    [gr.Image(type="pil"), 
    gr.Radio(['2x', '4x', '8x'], 
    type="value",
    value='2x',
    label='Resolution model')], 
    gr.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[['groot.jpeg', "2x"]],
    allow_flagging='never',
    cache_examples=False,
    ).queue(api_open=False).launch(show_error=True, show_api=False, share=True)
    

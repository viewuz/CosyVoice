import os
import sys
import argparse
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

cosyvoice: CosyVoice2


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.post("/add_zero_shot_spk")
async def add_zero_shot_spk(zero_shot_spk_id: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    return cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)


@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), zero_shot_spk_id: str = ""):
    model_output = cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=zero_shot_spk_id)
    return StreamingResponse(generate_data(model_output))


@app.post("/inference_instruct")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), zero_shot_spk_id: str = ""):
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, "", zero_shot_spk_id=zero_shot_spk_id)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    args = parser.parse_args()

    try:
        cosyvoice = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    except Exception:
        raise TypeError('no valid model_type!')

    uvicorn.run(app, host="0.0.0.0", port=args.port)

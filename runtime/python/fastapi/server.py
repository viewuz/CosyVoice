import os
import sys
import argparse
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)

from fastapi import FastAPI, UploadFile, Depends, Form, File, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import uvicorn
import numpy as np
import jwt

from modelscope import snapshot_download

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

import torchaudio

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM = "HS256"

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

cosyvoice: CosyVoice2
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """JWT 토큰 검증"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"Authenticate": "Bearer"},
        )


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.post("/add_zero_shot_spk")
async def add_zero_shot_spk(zero_shot_spk_id: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File(),
                            token_payload: dict = Depends(verify_token)
                            ):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    return cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)


@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), zero_shot_spk_id: str = "",
                              token_payload: dict = Depends(verify_token)
                              ):
    model_output = cosyvoice.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=zero_shot_spk_id)
    return StreamingResponse(generate_data(model_output))


@app.post("/inference_instruct")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), zero_shot_spk_id: str = "",
                              token_payload: dict = Depends(verify_token)
                              ):
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, "", zero_shot_spk_id=zero_shot_spk_id)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=80)

    parser.add_argument('--jwt-secret-key',
                        type=str,
                        default='')

    args = parser.parse_args()

    if not args.jwt_secret_key:
        raise ValueError("jwt_secret_key must be set")

    try:
        JWT_SECRET_KEY = str(args.jwt_secret_key)

        cosyvoice = CosyVoice2(
            'pretrained_models/CosyVoice2-0.5B',
            load_jit=False,
            load_trt=True,
            load_vllm=False,
            fp16=True,
            trt_concurrent=3
        )

    except Exception:
        raise TypeError('no valid model_type!')

    uvicorn.run(app, host="0.0.0.0", port=args.port)

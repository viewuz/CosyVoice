import asyncio
import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Depends, Form, File, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import uvicorn
import numpy as np
import jwt
# 세상에서 가장 즐거운 목소리로 노래를 불러줘요오~ 부를르 부를르
#  curl -X POST "http://150.230.35.143:8000/add_zero_shot_spk" -H "Authorization: Bearer " -F "zero_shot_spk_id=pororo" -F "prompt_text=세상에서 가장 즐거운 목소리로 노래를 불러줘요오~ 부를르 부를르" -F "prompt_wav=@reference.wav"
from modelscope import snapshot_download

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

import torchaudio
import threading

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
DATA_DIR = os.getenv("DATA_DIR", "")
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
    logging.info(f"Add spk: {zero_shot_spk_id}, prompt_text: {prompt_text}")
    save_audio_path = Path(DATA_DIR) / f"{zero_shot_spk_id}.wav"
    save_text_path = Path(DATA_DIR) / f"{zero_shot_spk_id}.txt"

    # 2) UploadFile → 로컬 파일로 저장
    with save_audio_path.open("wb") as buffer:
        shutil.copyfileobj(prompt_wav.file, buffer)

    save_text_path.write_text(prompt_text, encoding="utf-8")

    prompt_speech_16k = load_wav(str(save_audio_path), 16000)
    assert cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id) is True

    logging.info(f"Saved spk: {zero_shot_spk_id}")

    # 3) 목소리 저장
    cosyvoice.save_spkinfo()

    return {
        "sample_rate": cosyvoice.sample_rate
    }


@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(),
                              zero_shot_spk_id: str = Form(),
                              speed: float = Form(default=1.0),
                              token_payload: dict = Depends(verify_token),
                              ):
    model_output = cosyvoice.inference_zero_shot(
        tts_text,
        '',
        '',
        zero_shot_spk_id=zero_shot_spk_id,
        stream=True,
        speed=speed
    )

    for i, j in enumerate(cosyvoice.inference_zero_shot(
        tts_text,
        '',
        '',
        zero_shot_spk_id=zero_shot_spk_id,
        stream=False,
        speed=speed
    )):
        torchaudio.save('zero_shot_{}_{}.wav'.format(i, zero_shot_spk_id), j['tts_speech'], cosyvoice.sample_rate)


    return StreamingResponse(generate_data(model_output))


@app.post("/inference_instruct")
async def inference_instruct2(tts_text: str = Form(),
                              instruct_text: str = Form(),
                              zero_shot_spk_id: str = Form(),
                              speed: float = Form(default=1.0),
                              token_payload: dict = Depends(verify_token)
                              ):
    model_output = cosyvoice.inference_instruct2(
        tts_text,
        instruct_text,
        "",
        zero_shot_spk_id=zero_shot_spk_id,
        stream=True,
        speed=speed
    )


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

    parser.add_argument('--data-dir',
                        type=str,
                        default='')

    args = parser.parse_args()

    if not args.jwt_secret_key:
        raise ValueError("jwt_secret_key must be set")
    elif not args.data_dir:
        raise ValueError("data_dir must be set")

    JWT_SECRET_KEY = str(args.jwt_secret_key)
    DATA_DIR = str(args.data_dir)

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    try:
        payload = {
            "sub": "user",
        }

        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)

    except Exception:
        raise TypeError('cannot create access token')

    try:

        cosyvoice = CosyVoice2(
            'pretrained_models/CosyVoice2-0.5B',
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False,
        )

    except Exception:
        raise TypeError('no valid model_type!')

    logging.info(f"Starting server on port {args.port}...")
    logging.info("Server is ready to accept requests!")
    logging.info("=== Server Startup Complete ===")
    logging.info(f"Access Token: {token}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)

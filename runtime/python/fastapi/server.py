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

JWT_SECRET_KEY: str

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


def generate_data(model_output, target_sample_rate=None):
    """오디오 데이터 생성 (샘플링 레이트 변환 지원)"""
    for i, j in enumerate(model_output):
        audio_tensor = j['tts_speech']

        # 샘플링 레이트 변환이 필요한 경우
        if target_sample_rate and target_sample_rate != cosyvoice.sample_rate:
            audio_tensor = torchaudio.transforms.Resample(cosyvoice.sample_rate, target_sample_rate)(audio_tensor)

        # int16로 변환
        audio = (audio_tensor.numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield audio


@app.post("/add_zero_shot_spk")
async def add_zero_shot_spk(zero_shot_spk_id: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File(),
                            token_payload: dict = Depends(verify_token)
                            ):
    logging.info(f"Add spk: {zero_shot_spk_id}, text: {prompt_text}")

    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    assert cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id) is True

    logging.info(f"Added spk: {zero_shot_spk_id}")

    cosyvoice.save_spkinfo()

    return {
        "sample_rate": cosyvoice.sample_rate
    }


@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(),
                              zero_shot_spk_id: str = Form(),
                              speed: float = Form(default=1.0),
                              token_payload: dict = Depends(verify_token),
                              stream: bool = Form(default=True),
                              sample_rate: int = Form(default=None),  # 새로 추가된 파라미터
                              ):
    model_output = cosyvoice.inference_zero_shot(
        tts_text,
        '',
        '',
        zero_shot_spk_id=zero_shot_spk_id,
        speed=speed,
        stream=stream,
    )

    return StreamingResponse(generate_data(model_output, sample_rate))


@app.post("/inference_instruct")
async def inference_instruct2(tts_text: str = Form(),
                              instruct_text: str = Form(),
                              zero_shot_spk_id: str = Form(),
                              speed: float = Form(default=1.0),
                              token_payload: dict = Depends(verify_token),
                              stream: bool = Form(default=True),
                              sample_rate: int = Form(default=None),  # 새로 추가된 파라미터

                              ):
    model_output = cosyvoice.inference_instruct2(
        tts_text,
        instruct_text,
        "",
        zero_shot_spk_id=zero_shot_spk_id,
        stream=stream,
        speed=speed
    )

    return StreamingResponse(generate_data(model_output, sample_rate))


if __name__ == '__main__':
    snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=80)

    parser.add_argument('--jwt-secret-key',
                        type=str,
                        default='')

    parser.add_argument('--trt-concurrent',
                        type=int,
                        default=1)

    args = parser.parse_args()

    if not args.jwt_secret_key:
        raise ValueError("jwt_secret_key must be set")

    JWT_SECRET_KEY = str(args.jwt_secret_key)

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
            load_jit=True,
            load_trt=True,
            load_vllm=False,
            trt_concurrent=int(args.trt_concurrent),
            fp16=True,
        )
    except Exception:
        raise TypeError('no valid model_type!')

    logging.info(f"Starting server on port {args.port}...")
    logging.info("Server is ready to accept requests!")
    logging.info("=== Server Startup Complete ===")
    logging.info(f"Access Token: {token}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)

# SurgZoom challenge container (SEGMENT track, ORena SAVE FOCUS).
# Build on an amd64 host with network access:
#   python challenge/fetch_resources.py           # base model + adapter -> challenge/resources/
#   docker build --platform linux/amd64 -t surgzoom-segment challenge/
# cu12.8 is required for the RTX PRO 6000 (sm_120) evaluation GPU.
FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    MODEL_SEQ_LEN=22756 \
    VIDEO_MAX_TOKEN_NUM=1024 \
    VIDEO_MIN_TOKEN_NUM=64 \
    FORCE_QWENVL_VIDEO_READER=torchvision \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HOME=/tmp/hf MODELSCOPE_CACHE=/tmp/ms MPLCONFIGDIR=/tmp/mpl HOME=/tmp

RUN groupadd -r user && useradd -m --no-log-init -r -g user -u 1000 user
WORKDIR /opt/app
COPY requirements.txt /opt/app/
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user resources/ /opt/app/resources/
COPY --chown=user:user inference.py /opt/app/
USER user
ENTRYPOINT ["python", "inference.py"]

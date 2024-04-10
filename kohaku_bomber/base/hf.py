import os

from huggingface_hub import HfApi, HfFileSystem

hf_client = HfApi(token=os.environ['HF_TOKEN'])
hf_fs = HfFileSystem(token=os.environ['HF_TOKEN'])

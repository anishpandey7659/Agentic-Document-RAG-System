# services/mistral_client.py

from mistralai.client import Mistral
from config import MISTRAL_API_KEY

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
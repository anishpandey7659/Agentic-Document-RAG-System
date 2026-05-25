from groq import Groq
from core.config import GROQ_API_KEY


groq_client = Groq(api_key=GROQ_API_KEY)
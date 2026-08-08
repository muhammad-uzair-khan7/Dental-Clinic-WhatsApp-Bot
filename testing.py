from pywa import WhatsApp, types
import os
from dotenv import load_dotenv

#Credentials import from .env file
load_dotenv()
PHONENUMBER_ID= os.getenv("PHONENUMBER_ID") #PHONENUMBER_ID is the WhatsApp Phone ID from Meta App Dashboard.
ACCESS_TOKEN= os.getenv("ACCESS_TOKEN") #ACCES_TOKEN is the WhatsApp App Access Token from Meta App Dasboard
APP_ID= os.getenv("APP_ID") #APP_ID is the whatsapp app id from Facebook apps
APP_SECRET=os.getenv("APP_SECRET") #APP_SECRET is the wahtsapp app secret from Facebook apps
VERIFY_TOKEN= os.getenv("VERIFY_TOKEN") #This is the verify token you set in your webhook configuration in the Meta App Dashboard. It is used to verify that incoming requests to your webhook are from WhatsApp.
GOOGLE_GENERATIVE_AI= os.getenv("GEMINI_API_KEY") #This is Gemini API key
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
POS_BASE_URL= os.getenv("POS_BASE_URL")

wa = WhatsApp(phone_id=PHONENUMBER_ID, token=ACCESS_TOKEN, app_id=APP_ID, app_secret=APP_SECRET, verify_token=VERIFY_TOKEN)

@wa.on_message()
def reply_text(_: WhatsApp, message: types.Message):
    message.reply_text("Hello")
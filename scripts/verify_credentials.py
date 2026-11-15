import os
import requests
import smtplib
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("VERIFYING CONFIGURED CREDENTIALS")
print("=" * 60)

# 1. Telegram
print("\n[1/4] Testing Telegram Bot API...")
tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
if not tg_token:
    print("❌ TELEGRAM_BOT_TOKEN is missing!")
else:
    try:
        r = requests.get(f"https://api.telegram.org/bot{tg_token}/getMe", timeout=10)
        res = r.json()
        if res.get("ok"):
            bot_info = res["result"]
            print(f"✅ Telegram Bot authenticated successfully: @{bot_info.get('username')} ({bot_info.get('first_name')})")
            print(f"   Target Chat ID: {tg_chat_id}")
        else:
            print(f"❌ Telegram Bot error: {res}")
    except Exception as e:
        print(f"❌ Telegram connection failed: {e}")

# 2. Groq
print("\n[2/4] Testing Groq API...")
groq_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_key:
    print("❌ GROQ_API_KEY is missing!")
else:
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Respond with 'Groq connected successfully'."}],
            max_tokens=20,
        )
        msg = resp.choices[0].message.content.strip()
        print(f"✅ Groq API connected: {msg}")
    except Exception as e:
        print(f"❌ Groq API error: {e}")

# 3. Gemini
print("\n[3/4] Testing Google Gemini API...")
gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
if not gemini_key:
    print("❌ GEMINI_API_KEY is missing!")
else:
    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        test_models = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]
        model_found = False
        for m in test_models:
            try:
                r = client.models.generate_content(
                    model=m,
                    contents="Reply with 'Gemini connected successfully'."
                )
                print(f"✅ Gemini API connected using '{m}': {r.text.strip()}")
                model_found = True
                break
            except Exception as em:
                print(f"   Model '{m}' check: {em}")
        if not model_found:
            print("❌ Could not connect to Gemini models.")
    except Exception as e:
        print(f"❌ Gemini API client error: {e}")

# 4. Gmail SMTP
print("\n[4/4] Testing Gmail SMTP...")
gmail_user = os.getenv("GMAIL_USER", "").strip()
gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "").strip()
if not gmail_user or not gmail_pass:
    print("ℹ️ Gmail credentials not configured (optional).")
else:
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
            s.login(gmail_user, gmail_pass)
            print(f"✅ Gmail SMTP Auth succeeded for: {gmail_user}")
    except Exception as e:
        print(f"❌ Gmail SMTP Auth failed: {e}")

print("\n" + "=" * 60)

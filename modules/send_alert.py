# telegram_module/telebot_alerts.py

import requests
from datetime import datetime
import logging

TELEGRAM_TOKEN = "8137723121:AAHZ2zbLni6ldsg8cggfOZ2E_nyKwyNK3Fs"
CHAT_ID = "@trades09bot"  # use @username format for channels or group IDs

def send_telegram_alert(prediction: int, prob_up: float, prob_down: float, threshold: float):
    """
    Sends a formatted alert to the Telegram bot with prediction details.
    """
    direction = "📈 CALL (BUY)" if prediction == 1 else "📉 PUT (SELL)"
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    message = (
        f"⚡ <b>Quotex AI Signal</b>\n"
        f"🕒 <b>Time:</b> {now}\n"
        f"📊 <b>Prediction:</b> {direction}\n"
        f"🧠 <b>Prob ↑:</b> {round(prob_up*100, 2)}%  |  <b>Prob ↓:</b> {round(prob_down*100, 2)}%\n"
        f"🎯 <b>Threshold:</b> {round(threshold*100, 1)}%\n"
        f"🔔 <b>Status:</b> {'✅ CONFIRMED' if prob_up > threshold else '⚠️ LOW CONFIDENCE'}"
    )

    url = f"https://api.telegram.org/bot{8137723121:AAHZ2zbLni6ldsg8cggfOZ2E_nyKwyNK3Fs}/sendMessage"
    payload = {
        "chat_id": "@trades09bot" ,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.warning(f"Telegram alert failed: {response.text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

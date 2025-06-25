import requests
from datetime import datetime
import logging

TELEGRAM_TOKEN = "8137723121:AAHZ2zbLni6ldsg8cggfOZ2E_nyKwyNK3Fs"
CHAT_ID = "7985809435"  # your chat ID

def send_telegram_alert(prediction=None, prob_up=None, prob_down=None, threshold=None, error_message=None):
    """
    Sends an alert to Telegram.
    If error_message is provided, sends a safe-formatted error alert.
    Otherwise, sends a prediction alert.
    """
    if error_message:
        # Wrap error message in code block to prevent Telegram parsing errors
        message = f"```\n{error_message}\n```"
    else:
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

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML" if not error_message else None  # Disable HTML if sending error_message
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.warning(f"Telegram message failed: {response.text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def send_telegram_message(text: str):
    """
    Send a simple text message alert (no prediction data needed).
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"⚠️ <b>Quotex AI Alert</b>\n🕒 <b>Time:</b> {now}\n\n{text}"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.warning(f"Telegram message failed: {response.text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

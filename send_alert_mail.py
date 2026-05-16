import smtplib
from email.message import EmailMessage
import os
import requests
from datetime import datetime


def get_location():
    """
    Gets approximate GPS location using public IP
    """
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        data = res.json()

        city = data.get("city", "Unknown")
        region = data.get("region", "Unknown")
        country = data.get("country", "Unknown")
        loc = data.get("loc", "0,0")  # latitude,longitude

        latitude, longitude = loc.split(",")

        return {
            "city": city,
            "region": region,
            "country": country,
            "latitude": latitude,
            "longitude": longitude
        }
    except:
        return {
            "city": "Unknown",
            "region": "Unknown",
            "country": "Unknown",
            "latitude": "0",
            "longitude": "0"
        }


def send_alert_email(
    image_path,
    poacher_detected=False,
    weapon_detected=False,
    animal_detected=False
):
    # ---------- CHECK CONDITION ----------
    if not (poacher_detected or weapon_detected or animal_detected):
        return "No detection → Mail not sent"

    # ---------- EMAIL CONFIG ----------
    SENDER_EMAIL = "samikshaskamanna@gmail.com"
    APP_PASSWORD = "ozsmxdeuoplxwjyg"
    RECEIVER_EMAIL = "samikshaskamanna@gmail.com"

    # ---------- THREAT LEVEL ----------
    if poacher_detected and weapon_detected:
        threat_level = "🔴 HIGH"
    elif poacher_detected:
        threat_level = "🟡 MEDIUM"
    elif animal_detected:
        threat_level = "🟢 LOW"
    else:
        threat_level = "NONE"

    # ---------- LOCATION + TIME ----------
    location = get_location()
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # ---------- CREATE MESSAGE ----------
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = "🚨 WILDEYE AI ALERT – Detection Detected"

    detected_items = []
    if poacher_detected:
        detected_items.append("Poacher")
    if weapon_detected:
        detected_items.append("Weapon")
    if animal_detected:
        detected_items.append("Animal")

    msg.set_content(
        f"""
🚨 ALERT FROM WILDEYE AI

Detected Objects:
- {', '.join(detected_items)}

Threat Level:
{threat_level}

📍 Location:
City: {location['city']}
Region: {location['region']}
Country: {location['country']}
Latitude: {location['latitude']}
Longitude: {location['longitude']}

🕒 Time:
{timestamp}

Immediate action may be required.

Regards,
WILDEYE AI System
"""
    )

    # ---------- ATTACH IMAGE ----------
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_data = f.read()
        msg.add_attachment(
            img_data,
            maintype="image",
            subtype="jpeg",
            filename="detected_output.jpg"
        )

    # ---------- SEND MAIL ----------
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

    return "Mail Sent Successfully"

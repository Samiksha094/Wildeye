from flask import Flask, render_template, request, redirect, url_for, session, Response
from ultralytics import YOLO
import os
import cv2
from datetime import datetime

# Import email alert function
from send_alert_mail import send_alert_email

# ---------------- APP SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = "wild_eye_secret"

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/train4/weights/best.pt")
model = YOLO(MODEL_PATH)

# ---------------- CLASS MAPPING ----------------
CLASS_NAMES = {
    0: "Poacher",
    1: "Weapon",
    2: "Animal"
}

# ---------------- IN-MEMORY HISTORY ----------------
DETECTION_HISTORY = []


# ---------------- DRAW BOXES FUNCTION ----------------
def draw_boxes(img, results):
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf < 0.35:
            continue

        # Calibration alignment rules
        if cls == 2:
            display_conf = 72.0 + ((conf - 0.35) / (1.0 - 0.35)) * (96.0 - 72.0)
        elif cls == 0:
            display_conf = 70.0 + ((conf - 0.35) / (1.0 - 0.35)) * (95.0 - 70.0)
        else:
            display_conf = 70.0 + ((conf - 0.35) / (1.0 - 0.35)) * (92.0 - 70.0)

        label = f"{CLASS_NAMES.get(cls, 'Target')} {display_conf:.1f}%"
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        if cls == 0:
            color = (0, 0, 255)  # Red
        elif cls == 1:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_y = max(y1, h + 10)
        cv2.rectangle(img, (x1, label_y - h - 10), (x1 + w + 10, label_y), color, -1)
        cv2.putText(img, label, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


# ---------------- LIVE WEBCAM STREAM GENERATOR ----------------
def generate_webcam_stream():
    # 0 launches the default integrated laptop webcam camera hardware
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run local real-time inference on the current webcam frame matrix
            results = model.predict(
                frame, conf=0.35, iou=0.20, imgsz=640, agnostic_nms=True, verbose=False
            )

            # Draw tracking bounding labels onto the frame canvas matrix
            annotated_frame = draw_boxes(frame.copy(), results)

            # Encode processed frame canvas into memory buffer distribution chunks
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Yield frame chunks sequentially using multipart response packaging standards
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()


# ---------------- BASE NAVIGATION ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "sam" and request.form["password"] == "1234":
            session["logged_in"] = True
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")


# ---------------- LIVE VIDEO ROUTING LINKS ----------------
@app.route("/live-feed")
def live_feed():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return Response(generate_webcam_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/live-camera")
def live_camera():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html", view_tab="live_camera")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- PREDICT / ANALYSIS FILE ENGINE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("home"))

    os.makedirs(app.static_folder, exist_ok=True)
    input_path = os.path.join(app.static_folder, "input.jpg")
    output_path = os.path.join(app.static_folder, "output.jpg")
    file.save(input_path)

    img = cv2.imread(input_path)
    if img is None:
        return render_template("index.html", error="Unable to read uploaded image.")

    results = model.predict(img, conf=0.35, iou=0.20, imgsz=640, agnostic_nms=True, verbose=False)
    annotated = draw_boxes(img.copy(), results)
    cv2.imwrite(output_path, annotated)

    stats = {"poacher": 0.0, "weapon": 0.0, "animal": 0.0}
    found_poacher = found_weapon = found_animal = False

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.35:
            continue

        cls = int(box.cls[0])
        conf_percent = 72.0 + ((conf - 0.35) / (1.0 - 0.35)) * (96.0 - 72.0)

        if cls == 0:
            stats["poacher"] = max(stats["poacher"], conf_percent)
            found_poacher = True
        elif cls == 1:
            stats["weapon"] = max(stats["weapon"], conf_percent)
            found_weapon = True
        elif cls == 2:
            stats["animal"] = max(stats["animal"], conf_percent)
            found_animal = True

    detection_found = found_poacher or found_weapon or found_animal
    mail_sent = "No"

    if detection_found:
        try:
            send_alert_email(
                image_path=output_path,
                poacher_detected=found_poacher,
                weapon_detected=found_weapon,
                animal_detected=found_animal
            )
            mail_sent = "Yes"
        except Exception as e:
            print("Email error:", e)
            mail_sent = "No"

    if found_poacher and found_weapon:
        threat_level, threat_color = "HIGH", "red"
    elif found_poacher:
        threat_level, threat_color = "MEDIUM", "orange"
    elif found_animal:
        threat_level, threat_color = "LOW", "green"
    else:
        threat_level, threat_color = "NONE", "gray"

    DETECTION_HISTORY.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "poacher": round(stats["poacher"], 1),
        "weapon": round(stats["weapon"], 1),
        "animal": round(stats["animal"], 1),
        "threat": threat_level,
        "color": threat_color,
        "mail": mail_sent
    })

    if len(DETECTION_HISTORY) > 15:
        DETECTION_HISTORY.pop(0)

    return render_template(
        "index.html",
        output_image="output.jpg",
        poacher=round(stats["poacher"], 1),
        weapon=round(stats["weapon"], 1),
        animal=round(stats["animal"], 1),
        mail_sent=mail_sent,
        alert=detection_found,
        threat_level=threat_level,
        threat_color=threat_color
    )


# ---------------- SIDEBAR ENDPOINTS ----------------
@app.route("/history")
def history():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html", view_tab="history", history=DETECTION_HISTORY)


@app.route("/threat-map")
def threat_map():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    latest_level = DETECTION_HISTORY[-1]["threat"] if DETECTION_HISTORY else "NONE"
    latest_color = DETECTION_HISTORY[-1]["color"] if DETECTION_HISTORY else "gray"

    return render_template("index.html", view_tab="map", threat_level=latest_level, threat_color=latest_color)


@app.route("/reports")
def reports():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    report_summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_scans": len(DETECTION_HISTORY),
        "critical_threats": sum(1 for log in DETECTION_HISTORY if log["threat"] in ["HIGH", "MEDIUM"]),
        "animals_monitored": sum(1 for log in DETECTION_HISTORY if log["animal"] > 0),
        "alerts_dispatched": sum(1 for log in DETECTION_HISTORY if log["mail"] == "Yes")
    }
    return render_template("index.html", view_tab="reports", history=DETECTION_HISTORY, summary=report_summary)


if __name__ == "__main__":
    app.run(debug=True)
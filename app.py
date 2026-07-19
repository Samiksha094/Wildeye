from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from ultralytics import YOLO
import os
import cv2
import numpy as np
from datetime import datetime
import geocoder

from send_alert_mail import send_alert_email

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = "wild_eye_secret"

MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/train4/weights/best.pt")
model = YOLO(MODEL_PATH)

# ── BIRD DETECTION: pretrained COCO model (no retraining needed) ──────────────
# yolov8m.pt (~50 MB, downloads once) — detects ALL bird species (sparrow,
# eagle, parrot, pigeon, crow, etc.) since COCO class 14 = "bird" (generic).
# Much better than yolov8n for small/distant/partially visible birds.
bird_model = YOLO("yolov8m.pt")
BIRD_COCO_CLASS = 14
BIRD_CONF = 0.15   # Lower threshold — birds are often partially visible
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {0: "Poacher", 1: "Weapon", 2: "Animal", 3: "Bird"}


# ---------------- FETCH REGIONAL FALLBACK GPS ----------------
def get_current_gps():
    """
    IP-based geolocation. This is only an approximation (tied to your ISP's
    registered location, not your device's actual position), and if the
    lookup fails entirely it falls back to the hardcoded coordinates below.
    This should ONLY be used when we have no real browser GPS yet — see the
    session-first logic in /predict and /get-location.
    """
    try:
        g = geocoder.ip('me')
        if g.latlng:
            return g.latlng[0], g.latlng[1]
    except Exception as e:
        print("Network location lookup dropped:", e)
    return 12.3336, 76.6432


# ---------------- CALIBRATED CONFIDENCE ----------------
def get_calibrated_conf(cls, raw_conf):
    if cls == 2:
        return 72.0 + ((raw_conf - 0.35) / (1.0 - 0.35)) * (96.0 - 72.0)
    elif cls == 0:
        return 70.0 + ((raw_conf - 0.35) / (1.0 - 0.35)) * (95.0 - 70.0)
    elif cls == 3:  # Bird — COCO model gives reliable raw conf, light calibration
        return 65.0 + ((raw_conf - 0.35) / (1.0 - 0.35)) * (94.0 - 65.0)
    else:
        return 70.0 + ((raw_conf - 0.35) / (1.0 - 0.35)) * (92.0 - 70.0)


# ---------------- DRAW BOXES ----------------
def draw_boxes(img, results, bird_results=None):
    # ── Original 3-class detections (Poacher / Weapon / Animal) ──
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if conf < 0.35:
            continue
        display_conf = get_calibrated_conf(cls, conf)
        label = f"{CLASS_NAMES.get(cls, 'Target')} {display_conf:.1f}%"
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        color = (0, 0, 255) if cls == 0 else (0, 165, 255) if cls == 1 else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_y = max(y1, h + 10)
        cv2.rectangle(img, (x1, label_y - h - 10), (x1 + w + 10, label_y), color, -1)
        cv2.putText(img, label, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── Bird detections from COCO model ──────────────────────────
    if bird_results is not None:
        for box in bird_results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id != BIRD_COCO_CLASS or conf < BIRD_CONF:
                continue
            display_conf = get_calibrated_conf(3, conf)
            label = f"Bird {display_conf:.1f}%"
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            color = (255, 200, 0)  # cyan-yellow — distinct from other classes
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_y = max(y1, h + 10)
            cv2.rectangle(img, (x1, label_y - h - 10), (x1 + w + 10, label_y), color, -1)
            cv2.putText(img, label, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img


# ---------------- NIGHT MODE WEBCAM STREAM ----------------
def generate_webcam_stream(is_night_mode=False):
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        if is_night_mode:
            # ── CRITICAL FIX ──────────────────────────────────────────────
            # YOLO MUST run on the original color frame so it has full color
            # cues (skin tone, clothing) to correctly distinguish Poacher vs
            # Animal vs Weapon.  The thermal effect is applied ONLY to the
            # display frame shown to the user — detection accuracy is unaffected.
            # ─────────────────────────────────────────────────────────────

            # Step 1: Run detection on the ORIGINAL color frame
            results = model.predict(frame, conf=0.35, iou=0.20, imgsz=640, agnostic_nms=True, verbose=False)
            bird_results = bird_model.predict(frame, conf=BIRD_CONF, iou=0.20, imgsz=640, classes=[BIRD_COCO_CLASS], verbose=False)

            # Step 2: Build the thermal display frame from the original
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Step 3: CLAHE — strong local contrast (thermal silhouette effect)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Step 4: Gamma LUT — crushes shadows, blows out bright regions (white-hot look)
            lut = np.array([
                np.clip(int(255 * ((i / 255.0) ** 0.45)), 0, 255)
                for i in range(256)
            ], dtype=np.uint8)
            thermal = cv2.LUT(enhanced, lut)

            # Step 5: Smooth noise then sharpen edges for clean silhouettes
            blurred = cv2.GaussianBlur(thermal, (3, 3), 0)
            sharpened = cv2.addWeighted(thermal, 1.8, blurred, -0.8, 0)

            # Step 6: Convert thermal display frame back to BGR for box drawing
            display_frame = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

            # Step 7: Draw detection boxes (from color-frame results) onto thermal display
            annotated_frame = draw_boxes(display_frame, results, bird_results)

        else:
            results = model.predict(frame, conf=0.35, iou=0.20, imgsz=640, agnostic_nms=True, verbose=False)
            bird_results = bird_model.predict(frame, conf=BIRD_CONF, iou=0.20, imgsz=640, classes=[BIRD_COCO_CLASS], verbose=False)
            annotated_frame = draw_boxes(frame.copy(), results, bird_results)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()


# ---------------- AUTH ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "sam" and request.form["password"] == "1234":
            session["logged_in"] = True
            if "history_log" not in session:
                session["history_log"] = []
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- MAIN ROUTES ----------------
@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html",
                           output_image=session.get("last_image"),
                           poacher=session.get("last_poacher", 0.0),
                           weapon=session.get("last_weapon", 0.0),
                           animal=session.get("last_animal", 0.0),
                           bird=session.get("last_bird", 0.0),
                           mail_sent=session.get("last_mail", "No"),
                           threat_level=session.get("last_threat"),
                           threat_color=session.get("last_color"))


@app.route("/live-feed")
def live_feed():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    lens_mode = request.args.get('mode', 'normal')
    is_night = (lens_mode == 'night')
    return Response(generate_webcam_stream(is_night_mode=is_night),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/live-camera")
def live_camera():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html", view_tab="live_camera")


@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(url_for("home"))

    input_path = os.path.join(app.static_folder, "input.jpg")
    output_path = os.path.join(app.static_folder, "output.jpg")
    file.save(input_path)
    img = cv2.imread(input_path)

    results = model.predict(img, conf=0.35, iou=0.20, imgsz=640, agnostic_nms=True, verbose=False)
    bird_results = bird_model.predict(img, conf=BIRD_CONF, iou=0.20, imgsz=640, classes=[BIRD_COCO_CLASS], verbose=False)
    cv2.imwrite(output_path, draw_boxes(img.copy(), results, bird_results))

    stats = {"p": 0.0, "w": 0.0, "a": 0.0, "b": 0.0}
    fp = fw = fa = fb = False
    for box in results[0].boxes:
        c, cl = float(box.conf[0]), int(box.cls[0])
        if c < 0.35:
            continue
        val = get_calibrated_conf(cl, c)
        if cl == 0: stats["p"], fp = max(stats["p"], val), True
        elif cl == 1: stats["w"], fw = max(stats["w"], val), True
        elif cl == 2: stats["a"], fa = max(stats["a"], val), True

    # Check birds from COCO model
    # Birds get their own flag/stat (fb) instead of being folded into the
    # Animal flag, since no bird photos exist in the custom training set —
    # this is the only signal we have that a bird was seen.
    for box in bird_results[0].boxes:
        c, cl = float(box.conf[0]), int(box.cls[0])
        if cl == BIRD_COCO_CLASS and c >= BIRD_CONF:
            fb = True
            val = get_calibrated_conf(3, c)
            stats["b"] = max(stats["b"], val)

    # ── REAL-TIME GPS FIX ──────────────────────────────────────────────────
    # Use the live browser GPS already pushed into the session by
    # /update-location (sent from the user's device every 2s). Only fall
    # back to the rough IP-based lookup if we genuinely have no real fix
    # yet. Previously this unconditionally called get_current_gps(), which
    # overwrote accurate live coordinates with an inaccurate IP guess on
    # every single detection.
    lat = session.get("last_lat")
    lng = session.get("last_lng")
    if lat is None or lng is None:
        lat, lng = get_current_gps()
    # ─────────────────────────────────────────────────────────────────────

    m_sent = "No"
    if fp or fw or fa or fb:
        try:
            # send_alert_email receives lat, lng so the email reports the
            # real-time location instead of doing its own (inaccurate)
            # lookup internally, plus the bird flag so bird-only sightings
            # also trigger a mail.
            send_alert_email(output_path, fp, fw, fa, fb, lat, lng)
            m_sent = "Yes"
        except:
            m_sent = "No"

    t_lvl, t_clr = (
        ("HIGH", "red") if fp and fw else
        ("MEDIUM", "orange") if fp else
        ("LOW", "green") if fa else
        ("BIRD", "blue") if fb else
        ("NONE", "gray")
    )

    session.update({
        "last_image": "output.jpg",
        "last_poacher": round(stats["p"], 1),
        "last_weapon": round(stats["w"], 1),
        "last_animal": round(stats["a"], 1),
        "last_bird": round(stats["b"], 1),
        "last_mail": m_sent,
        "last_threat": t_lvl,
        "last_color": t_clr,
        "last_lat": lat,
        "last_lng": lng
    })

    logs = session.get("history_log", [])
    logs.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "poacher": round(stats["p"], 1),
        "weapon": round(stats["w"], 1),
        "animal": round(stats["a"], 1),
        "bird": round(stats["b"], 1),
        "threat": t_lvl,
        "color": t_clr,
        "mail": m_sent,
        "lat": lat,
        "lng": lng
    })
    if len(logs) > 15:
        logs.pop(0)
    session["history_log"] = logs

    return redirect(url_for("home"))


@app.route("/history")
def history():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html", view_tab="history", history=session.get("history_log", []))


# ---------------- LIVE GPS ENDPOINTS (FAST BROWSER-BASED) ----------------
@app.route("/update-location", methods=["POST"])
def update_location():
    """Receives high-accuracy GPS coordinates pushed from the browser every 2s."""
    if not session.get("logged_in"):
        return jsonify({"status": "unauthorized"}), 401
    data = request.get_json()
    if data and "lat" in data and "lng" in data:
        session["last_lat"] = data["lat"]
        session["last_lng"] = data["lng"]
        session["location_accuracy"] = data.get("accuracy", None)
        session["location_source"] = "browser_gps"
        return jsonify({"status": "ok", "lat": data["lat"], "lng": data["lng"]})
    return jsonify({"status": "error"}), 400


@app.route("/get-location")
def get_location():
    """Polled by map every 2 seconds to get latest GPS + threat state."""
    if not session.get("logged_in"):
        return jsonify({"status": "unauthorized"}), 401
    lat = session.get("last_lat")
    lng = session.get("last_lng")
    if lat is None:
        lat, lng = get_current_gps()
    return jsonify({
        "lat": lat,
        "lng": lng,
        "source": session.get("location_source", "ip"),
        "accuracy": session.get("location_accuracy", None),
        "threat": session.get("last_threat", "NONE"),
        "color": session.get("last_color", "gray")
    })


# ---------------- MAP & REPORTS ----------------
@app.route("/threat-map")
def threat_map():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    lat, lng = session.get("last_lat"), session.get("last_lng")
    if lat is None:
        lat, lng = get_current_gps()
    return render_template("index.html", view_tab="map",
                           threat_level=session.get("last_threat", "NONE"),
                           threat_color=session.get("last_color", "gray"),
                           lat=lat, lng=lng)


@app.route("/reports")
def reports():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    logs = session.get("history_log", [])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_scans": len(logs),
        "critical_threats": sum(1 for log in logs if log["threat"] in ["HIGH", "MEDIUM"]),
        "animals_monitored": sum(1 for log in logs if log["animal"] > 0),
        "alerts_dispatched": sum(1 for log in logs if log["mail"] == "Yes")
    }
    return render_template("index.html", view_tab="reports", history=logs, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)

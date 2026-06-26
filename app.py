from flask import Flask
from flask import render_template
from flask import Response
from flask import jsonify

import cv2
import mediapipe as mp
import numpy as np
import joblib

app = Flask(__name__)

# ==================================
# LOAD MODEL
# ==================================

model = joblib.load("model.pkl")

# ==================================
# GLOBAL VARIABLE
# ==================================

hasil_deteksi = "-"
akurasi_deteksi = 0

# ==================================
# MEDIAPIPE
# ==================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

# ==================================
# CAMERA
# ==================================

camera = cv2.VideoCapture(0)

# ==================================
# VIDEO STREAM
# ==================================

def generate_frames():

    global hasil_deteksi
    global akurasi_deteksi

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(rgb)

        data = []

        hand_list = []

        tangan_terdeteksi = False

        # ==========================
        # DETEKSI TANGAN
        # ==========================

        if result.multi_hand_landmarks:

            tangan_terdeteksi = True

            for hand in result.multi_hand_landmarks:

                xs = [lm.x for lm in hand.landmark]

                avg_x = sum(xs) / len(xs)

                hand_list.append(
                    (avg_x, hand)
                )

            hand_list.sort(
                key=lambda x: x[0]
            )

            for _, hand in hand_list[:2]:

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                temp = []

                for lm in hand.landmark:

                    temp.extend([
                        lm.x,
                        lm.y
                    ])

                base_x = temp[0]
                base_y = temp[1]

                for i in range(
                    0,
                    len(temp),
                    2
                ):

                    temp[i] -= base_x
                    temp[i + 1] -= base_y

                data.extend(temp)

        # ==========================
        # JIKA TIDAK ADA TANGAN
        # ==========================

        if not tangan_terdeteksi:

            hasil_deteksi = "-"
            akurasi_deteksi = 0

        else:

            while len(data) < 84:
                data.append(0)

            data = data[:84]

            data = np.array(
                data
            ).reshape(1, -1)

            try:

                pred = model.predict(
                    data
                )[0]

                conf = np.max(
                    model.predict_proba(
                        data
                    )
                ) * 100

                hasil_deteksi = pred

                akurasi_deteksi = round(
                    conf,
                    2
                )

            except:

                hasil_deteksi = "-"
                akurasi_deteksi = 0

        # ==========================
        # ENCODE FRAME
        # ==========================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

# ==================================
# ROUTE HOME
# ==================================

@app.route('/')
def index():

    return render_template(
        "index.html"
    )

# ==================================
# ROUTE VIDEO
# ==================================

@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype=
        'multipart/x-mixed-replace; boundary=frame'
    )

# ==================================
# ROUTE DATA AJAX
# ==================================

@app.route('/data')
def data():

    return jsonify({

        "hasil":
        hasil_deteksi,

        "akurasi":
        akurasi_deteksi

    })

# ==================================
# MAIN
# ==================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
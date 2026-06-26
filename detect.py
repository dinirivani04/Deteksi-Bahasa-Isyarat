import cv2
import mediapipe as mp
import numpy as np
import joblib

# ==========================
# Load model
# ==========================
model = joblib.load("model.pkl")

# ==========================
# MediaPipe setup
# ==========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

# ==========================
# Kamera
# ==========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera:", cap.isOpened())

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    data = []
    hand_list = []

    # ==========================
    # DETEKSI TANGAN
    # ==========================
    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            # rata-rata posisi tangan (biar urutan stabil)
            xs = [lm.x for lm in hand.landmark]
            avg_x = sum(xs) / len(xs)

            hand_list.append((avg_x, hand))

        # urutkan (boleh kiri/kanan, tapi konsisten)
        hand_list.sort(key=lambda x: x[0])

        # ambil maksimal 2 tangan
        for _, hand in hand_list[:2]:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            temp = []

            for lm in hand.landmark:
                temp.extend([lm.x, lm.y])

            # ==========================
            # normalisasi tangan
            # ==========================
            base_x = temp[0]
            base_y = temp[1]

            for i in range(0, len(temp), 2):
                temp[i] -= base_x
                temp[i + 1] -= base_y

            data.extend(temp)

    # ==========================
    # PAKSA 84 FITUR
    # ==========================
    while len(data) < 84:
        data.append(0)

    data = data[:84]
    data = np.array(data).reshape(1, -1)

    # ==========================
    # PREDIKSI
    # ==========================
    pred = model.predict(data)[0]
    conf = np.max(model.predict_proba(data)) * 100

    # tampilkan jika cukup yakin
    if conf > 70:

        cv2.putText(
            frame,
            f"{pred} ({conf:.1f}%)",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("BISINDO 1-2 HAND STABLE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
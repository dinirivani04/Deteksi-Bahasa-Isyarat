import os
import cv2
import numpy as np
import mediapipe as mp
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ==========================
# MEDIAPIPE
# ==========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(

    static_image_mode=True,

    max_num_hands=2,

    min_detection_confidence=0.6

)

# ==========================
# DATASET
# ==========================

DATASET_PATH = "dataset"

X=[]
y=[]

print("Membaca dataset...")

for category in os.listdir(DATASET_PATH):

    category_path=os.path.join(
        DATASET_PATH,
        category
    )

    if not os.path.isdir(category_path):
        continue

    print("\nKategori :",category)

    for label in os.listdir(category_path):

        label_path=os.path.join(
            category_path,
            label
        )

        if not os.path.isdir(label_path):
            continue

        print("  Kelas :",label)

        success=0

        for file in os.listdir(label_path):

            if not file.lower().endswith(
                (".jpg",".png",".jpeg")
            ):
                continue

            image_path=os.path.join(
                label_path,
                file
            )

            image=cv2.imread(image_path)

            if image is None:
                continue

            rgb=cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            result=hands.process(rgb)

            data=[]

            hand_list=[]

            # ==========================
            # DETEKSI TANGAN
            # ==========================

            if result.multi_hand_landmarks:

                for hand in result.multi_hand_landmarks:

                    xs=[lm.x for lm in hand.landmark]

                    avg_x=sum(xs)/len(xs)

                    hand_list.append(
                        (avg_x,hand)
                    )

                hand_list.sort(
                    key=lambda x:x[0]
                )

                for _,hand in hand_list[:2]:

                    temp=[]

                    for lm in hand.landmark:

                        temp.extend([

                            lm.x,

                            lm.y

                        ])

                    base_x=temp[0]

                    base_y=temp[1]

                    for i in range(
                        0,
                        len(temp),
                        2
                    ):

                        temp[i]-=base_x

                        temp[i+1]-=base_y

                    data.extend(temp)

            # ==========================
            # PAKSA 84 FITUR
            # ==========================

            while len(data)<84:

                data.append(0)

            data=data[:84]

            X.append(data)

            y.append(label)

            success+=1

        print(
            "     Gambar:",
            success
        )

hands.close()

# ==========================
# DATA TRAIN
# ==========================

X=np.array(X)

y=np.array(y)

print("\nTotal Data :",len(X))

print("Jumlah kelas :",len(np.unique(y)))

X_train,X_test,y_train,y_test=\
train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ==========================
# MODEL
# ==========================

model=make_pipeline(

    StandardScaler(),

    MLPClassifier(

        hidden_layer_sizes=(128,64),

        max_iter=1500,

        random_state=42

    )

)

print("\nTraining...")

model.fit(

    X_train,

    y_train

)

pred=model.predict(X_test)

acc=accuracy_score(

    y_test,

    pred

)

print(

    f"\nAkurasi : {acc*100:.2f}%"

)

joblib.dump(

    model,

    "model.pkl"

)

print("\nModel berhasil disimpan")

print("File : model.pkl")
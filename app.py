import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, model_complexity=0)

sample_rate = 44100
current_freq = 440
current_volume = 0.2
phase = 0

# ================= AUDIO =================
def audio_callback(outdata, frames, time, status):
    global current_freq, current_volume, phase

    t = (np.arange(frames) + phase) / sample_rate
    wave = current_volume * np.sin(2 * np.pi * current_freq * t)

    phase += frames
    phase %= sample_rate

    outdata[:] = wave.reshape(-1, 1)

stream = sd.OutputStream(
    callback=audio_callback,
    channels=1,
    samplerate=sample_rate,
    blocksize=1024
)

stream.start()

# ================= FUNÇÕES =================
def distancia(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def freq_to_note(freq):
    A4 = 440
    if freq <= 0:
        return "-"
    n = round(12 * math.log2(freq / A4))
    notes = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    note_index = (n + 9) % 12
    octave = 4 + ((n + 9) // 12)
    return f"{notes[note_index]}{octave}"

def draw_bar(img, value, max_value, x, y, w, h, color):
    ratio = value / max_value
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y),
                  (x + int(w * ratio), y + h),
                  color, -1)

def draw_glow_circle(img, center, radius, color):
    for i in range(3, 0, -1):
        alpha = 0.2
        overlay = img.copy()
        cv2.circle(overlay, center, radius + i*5, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, center, radius, color, -1)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            label = handedness.classification[0].label

            polegar = hand_landmarks.landmark[4]
            indicador = hand_landmarks.landmark[8]

            x1, y1 = int(polegar.x * w), int(polegar.y * h)
            x2, y2 = int(indicador.x * w), int(indicador.y * h)

            dist = distancia(polegar, indicador)

            if label == "Left":
                cor = (255, 100, 0)  # Laranja (Volume)
            else:
                cor = (0, 150, 255)  # Azul (Frequência)

            if dist < 0.05:
                y_norm = indicador.y

                if label == "Left":
                    current_volume = max(0.0, min(1.0, 1 - y_norm))
                else:
                    current_freq = int(200 + (1 - y_norm) * 800)

                cor = (0, 255, 0)

            # Glow + linha
            draw_glow_circle(img, (x1, y1), 10, cor)
            draw_glow_circle(img, (x2, y2), 10, cor)
            cv2.line(img, (x1, y1), (x2, y2), cor, 2)

    # ====== HUD ======
    note = freq_to_note(current_freq)

    cv2.putText(img,
                f"Freq: {int(current_freq)} Hz ({note})",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.putText(img,
                f"Volume: {current_volume:.2f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    # Barras
    draw_bar(img, current_volume, 1.0,
             30, 100, 200, 20, (0, 255, 0))

    draw_bar(img, current_freq - 200, 800,
             30, 140, 200, 20, (255, 0, 0))

    cv2.imshow("MUSGA.PY", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
stream.stop()
stream.close()
cv2.destroyAllWindows()

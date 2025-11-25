# app.py
import io, base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


# ---------- CONFIGURA AQUÍ SI QUIERES ----------
MODEL_PATH = "model/vgg16_fisuras.h5" # coloca tu modelo aquí
CLASS_NAMES = ['sin_fisura','leve','media','grave']
IMG_SIZE = (224,224)
# ------------------------------------------------


app = FastAPI(title="API Fisuras")
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


# Cargar el modelo al iniciar
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_pil(pil_img):
img = pil_img.convert('RGB').resize(IMG_SIZE)
arr = np.array(img).astype('float32')
arr = preprocess_input(arr)
arr = np.expand_dims(arr, 0)
return arr, np.array(img)




def predict_and_probs(pil_img):
x, img_resized = preprocess_pil(pil_img)
preds = model.predict(x)
# TTA simple: flip horizontal
pil_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT).resize(IMG_SIZE)
x_flip = preprocess_input(np.array(pil_flip).astype('float32'))
x_flip = np.expand_dims(x_flip, 0)
preds = (preds + model.predict(x_flip)) / 2.0
probs = preds[0].tolist()
idx = int(np.argmax(probs))
label = CLASS_NAMES[idx]
return label, probs, idx, img_resized




def gradcam_base64(pil_img, pred_index):
x, img_resized = preprocess_pil(pil_img)
# encontrar última capa conv
last_conv = None
for layer in reversed(model.layers):
if 'conv' in layer.name:
last_conv = layer.name
break
if last_conv is None:
return None
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
with tf.GradientTape() as tape:
conv_outputs, predictions = grad_model(tf.convert_to_tensor(x))
loss = predictions[:, pred_index]
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
conv_outputs = conv_outputs[0].numpy()
for i in range(pooled_grads.shape[-1]):
conv_outputs[:, :, i] *= pooled_grads[i]
heatmap = np.mean(conv_outputs, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (heatmap.max() + 1e-8)
heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
return {"status":"ok"}

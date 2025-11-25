# adaptado para backend (no usa Colab ni display)
import io
import base64
from PIL import Image
import cv2
import numpy as np

# --- pega aquí tu función analizar_fisura_completo adaptada para recibir numpy array ---
def analizar_fisura_completo_from_array(img_bgr):
    # img_bgr: array BGR (como lee cv2)
    # Copia exactamente tu lógica pero adaptando las rutas de salida y sin usar files.upload ni plt.show
    # (aquí te muestro un extracto; pega todo tu código dentro y sustituye cv2.imread(...) por usar img_bgr)
    img = img_bgr.copy()
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ... resto de tu procesamiento idéntico ...
    # Al final debes devolver lo mismo:
    # return True/False, clasificaciones, datos_estadisticos, resultado_img (numpy BGR)
    # NOTA: en lugar de guardar en /content/... devolvemos la imagen en memoria
    # --- (implementa tu lógica aquí) ---
    raise NotImplementedError("Pega tu implementación aquí y devuelve resultado_img en memoria")
# -------------------------------------------------------------------------------

# Función envoltorio que recibe bytes (por ejemplo desde FastAPI) y devuelve JSON-friendly outputs
def analizar_fisura_de_bytes(image_bytes, max_size=1280):
    # Convertir bytes -> PIL -> RGB -> BGR numpy
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Reducir tamaño si es muy grande (mantener aspect ratio)
    w,h = pil.size
    scale = 1.0
    if max(w,h) > max_size:
        scale = max_size / float(max(w,h))
        new_w = int(w*scale)
        new_h = int(h*scale)
        pil = pil.resize((new_w, new_h), Image.ANTIALIAS)
    img_rgb = np.array(pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Llama a tu función principal (modificada para recibir numpy)
    hay_fisuras, clasificaciones, datos_estadisticos, resultado_img = analizar_fisura_completo_from_array(img_bgr)

    # Codificar resultado_img (BGR) a base64 JPEG para enviar en JSON
    _, buffer = cv2.imencode('.jpg', resultado_img)
    jpg_b64 = base64.b64encode(buffer).decode('utf-8')
    gradcam_dataurl = "data:image/jpeg;base64," + jpg_b64

    return {
        "detected": bool(hay_fisuras),
        "clasificaciones": clasificaciones,
        "estadisticas": datos_estadisticos,
        "image_result_base64": gradcam_dataurl
    }


# fisuras-deploy


1. Coloca tu modelo dentro de `model/vgg16_fisuras.h5`.
2. Construye la imagen Docker localmente para test:
- `docker build -t fisuras-app .`
- `docker run -p 8000:8000 fisuras-app`
- Prueba: `curl http://localhost:8000/health`
3. Subir a GitHub y desplegar en Render (o Railway/Cloud Run):
- En Render: New -> Web Service -> conectar repo -> selecciona Docker
- Espera a que Render construya la imagen y devuelva la URL.
4. Sube `index.html` a Netlify / GitHub Pages y actualiza `API_URL` con la URL de Render (`/predict`).


Notas:
- Si el modelo pesa mucho, considera descargar el modelo desde un bucket en el `startup` en lugar de almacenarlo en el repo.
- En producción restringe CORS al dominio del frontend.

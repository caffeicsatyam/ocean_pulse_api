from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse,  HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io
import os
import shutil
import uuid

app = FastAPI(title="Marine Debris Detection API")

# --- Load YOLO model ---
MODEL_PATH = "model/best.pt"   # Using forward slashes for cross-platform compatibility
model = YOLO(MODEL_PATH)

# --- Create folders with proper permissions ---
for folder in ["uploads", "outputs"]:
    os.makedirs(folder, exist_ok=True)
    # Ensure proper permissions on Linux
    if os.name == 'posix':
        os.chmod(folder, 0o755)


@app.get("/")
def home():
    return {"message": "🌊 Marine Debris YOLO Detection API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image or video file, performs YOLO detection,
    and returns JSON + link to result with bounding boxes.
    """
    # Generate unique filename to avoid overwrites
    file_ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{file_ext}"
    upload_path = os.path.join("uploads", unique_name)
    output_path = os.path.join("outputs", f"detected_{unique_name}")

    # Save uploaded file
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Detect whether it's an image or video
    is_image = file.content_type.startswith("image/")
    is_video = file.content_type.startswith("video/")

    # --- Run YOLO prediction ---
    results = model.predict(source=upload_path, save=True, project="outputs", name=f"det_{uuid.uuid4().hex}", conf=0.25)

    # YOLO automatically saves result under: outputs/det_xx/
    result_dir = results[0].save_dir
    output_files = [os.path.join(result_dir, f) for f in os.listdir(result_dir)]

    # Pick the first file as final output
    output_file = output_files[0] if output_files else None

    # --- Build detection info ---
    detections = []
    if hasattr(results[0], "boxes") and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            detections.append({
                "class": label,
                "confidence": round(conf, 3)
            })

    if not output_file or not os.path.exists(output_file):
        return JSONResponse({"error": "No output file generated"}, status_code=500)

    # --- Return JSON + download link ---
    return {
        "file_type": "image" if is_image else "video",
        "detections": detections,
        "output_url": f"/output/{os.path.basename(output_file)}"  # Using relative URL
    }


@app.get("/output/{filename}")
def get_output(filename: str):
    """Serve the processed output file (image/video)"""
    for root, _, files in os.walk("outputs"):
        if filename in files:
            file_path = os.path.join(root, filename)
            # Detect content type automatically
            ext = os.path.splitext(filename)[1].lower()
            media_type = "image/jpeg" if ext in [".jpg", ".jpeg", ".png", ".webp"] else "video/mp4"
            return FileResponse(file_path, media_type=media_type)
    return JSONResponse({"error": "Output file not found"}, status_code=404)

@app.get("/ui", response_class=HTMLResponse)
def upload_ui():
    with open("templates/index.html", "r") as f:
        return f.read()
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

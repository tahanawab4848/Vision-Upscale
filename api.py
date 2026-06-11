import os
import shutil
import sys
import asyncio
from typing import List
from PIL import Image
import numpy as np

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs/infer", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("outputs/microcosm_frames", exist_ok=True)

# ---------------------------------------------------------------------------
# Models & Metrics
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Real-ESRGAN x4+ (Photos)": "RealESRGAN_x4plus.pth",
    "Real-ESRGAN x4+ Anime (Cartoons)": "RealESRGAN_x4plus_anime_6B.pth"
}

def get_sharpness_score(image_path: str) -> float:
    try:
        img = Image.open(image_path).convert("L")
        arr = np.array(img).astype(float)
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            return 0.0
        laplacian = (
            arr[1:-1, 0:-2] + arr[1:-1, 2:] + 
            arr[0:-2, 1:-1] + arr[2:, 1:-1] - 
            4 * arr[1:-1, 1:-1]
        )
        return float(laplacian.var())
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# WebSocket Logs Manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/api/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class InferRequest(BaseModel):
    filename: str
    preset: str
    fp16: bool = True
    tile_size: int = 512
    tile_pad: int = 32

class MicrocosmRequest(BaseModel):
    filename: str
    preset: str
    levels: int = 3
    frames: int = 60

@app.get("/api/models")
async def get_models():
    return {"models": list(MODEL_CONFIGS.keys())}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    sharpness = get_sharpness_score(file_location)
    return {"info": "saved", "filename": file.filename, "sharpness": sharpness}

@app.post("/api/run_inference")
async def run_inference(req: InferRequest):
    input_path = f"uploads/{req.filename}"
    output_path = "outputs/infer"
    
    ckpt_file = MODEL_CONFIGS.get(req.preset, "RealESRGAN_x4plus.pth")
    checkpoint = f"checkpoints/{ckpt_file}"
    
    if not os.path.exists(checkpoint):
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Checkpoint {ckpt_file} not found locally."})
        
    cmd = [
        sys.executable, "-u", "infer.py",
        "--input", input_path,
        "--checkpoint", checkpoint,
        "--output", output_path,
        "--tile", str(req.tile_size),
        "--tile_pad", str(req.tile_pad)
    ]
    if req.fp16:
        cmd.append("--fp16")
    
    await manager.broadcast(f"Starting inference with {ckpt_file}...")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await manager.broadcast(line.decode().strip())
        
    await process.wait()
    
    if process.returncode == 0:
        base_name = os.path.splitext(req.filename)[0]
        out_file = base_name + "_SR.png"
        sharpness = get_sharpness_score(os.path.join(output_path, out_file))
        await manager.broadcast("Inference completed successfully.")
        return {"status": "success", "output_file": out_file, "sharpness": sharpness}
    else:
        await manager.broadcast("Inference failed.")
        return JSONResponse(status_code=500, content={"status": "error"})

@app.post("/api/run_microcosm")
async def run_microcosm(req: MicrocosmRequest):
    input_path = f"uploads/{req.filename}"
    output_path = "outputs/microcosm_frames"
    
    ckpt_file = MODEL_CONFIGS.get(req.preset, "RealESRGAN_x4plus.pth")
    checkpoint = f"checkpoints/{ckpt_file}"
    
    cmd = [
        sys.executable, "-u", "microcosm_explorer.py",
        "--input", input_path,
        "--checkpoint", checkpoint,
        "--output_dir", output_path,
        "--levels", str(req.levels),
        "--frames_per_level", str(req.frames)
    ]
    
    await manager.broadcast("Starting Infinite Microcosm generation...")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await manager.broadcast(line.decode().strip())
        
    await process.wait()
    
    if process.returncode == 0:
        await manager.broadcast(f"Microcosm frames saved to {output_path}")
        return {"status": "success"}
    else:
        await manager.broadcast("Microcosm generation failed.")
        return JSONResponse(status_code=500, content={"status": "error"})

@app.post("/api/run_fingerprint")
async def run_fingerprint():
    await manager.broadcast("Starting Fingerprint Forensics Demo...")
    cmd = [sys.executable, "-u", "fingerprint_forensics.py"]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await manager.broadcast(line.decode().strip())
    await process.wait()
    if process.returncode == 0:
        await manager.broadcast("Fingerprint Demo Complete.")
        return {"status": "success"}
    else:
        await manager.broadcast("Fingerprint Demo Failed.")
        return JSONResponse(status_code=500, content={"status": "error"})

@app.post("/api/run_video")
async def run_video(req: InferRequest):
    input_path = f"uploads/{req.filename}"
    output_path = "outputs/videos"
    
    ckpt_file = MODEL_CONFIGS.get(req.preset, "RealESRGAN_x4plus.pth")
    checkpoint = f"checkpoints/{ckpt_file}"
    
    cmd = [
        sys.executable, "-u", "video_infer.py",
        "--input", input_path,
        "--checkpoint", checkpoint,
        "--output_dir", output_path,
        "--tile", str(req.tile_size),
        "--tile_pad", str(req.tile_pad)
    ]
    if req.fp16:
        cmd.append("--fp16")
    
    await manager.broadcast(f"Starting Video Pipeline with {ckpt_file}...")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await manager.broadcast(line.decode().strip())
        
    await process.wait()
    
    if process.returncode == 0:
        base_name = os.path.splitext(req.filename)[0]
        out_file = base_name + "_4K.mp4"
        await manager.broadcast("Video inference completed successfully.")
        return {"status": "success", "output_file": out_file}
    else:
        await manager.broadcast("Video inference failed.")
        return JSONResponse(status_code=500, content={"status": "error"})

@app.get("/api/output/videos/{filename}")
async def get_output_video(filename: str):
    file_path = f"outputs/videos/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/api/output/infer/{filename}")
async def get_output_image(filename: str):
    file_path = f"outputs/infer/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/api/uploads/{filename}")
async def get_upload_image(filename: str):
    file_path = f"uploads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

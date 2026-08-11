import argparse
import csv
import json
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "captcha"
REJECTED_DIR = IMAGE_DIR / "rejected"
LABELS_FILE = ROOT / "labels.csv"
MODEL_ID = "anuashok/ocr-captcha-v3"
model_lock = threading.Lock()
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

PAGE = r'''<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAPTCHA Labeler</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui,sans-serif}main{width:min(760px,92vw);margin:5vh auto}.top{display:flex;justify-content:space-between;color:#9ca3af}.card{margin-top:16px;padding:28px;background:#1f2937;border-radius:14px}.image{display:flex;min-height:260px;align-items:center;justify-content:center;background:white;border-radius:9px;padding:24px}.image img{width:min(100%,600px);image-rendering:auto}.status{height:24px;margin:18px 0 6px;color:#93c5fd}input{width:100%;padding:14px 16px;border:2px solid #4b5563;border-radius:8px;background:#111827;color:white;font:700 24px monospace;text-transform:none}input:focus{outline:none;border-color:#60a5fa}.actions{display:flex;gap:10px;margin-top:14px}button{padding:11px 18px;border:0;border-radius:7px;font-weight:700;cursor:pointer}.save{background:#2563eb;color:white}.reject{margin-left:auto;background:#991b1b;color:white}.skip{background:#374151;color:white}.hint{margin-top:18px;color:#9ca3af;font-size:13px}progress{width:100%;margin-top:12px}
</style>
<main><div class="top"><strong>CAPTCHA Labeler</strong><span id="count"></span></div><progress id="progress" max="1"></progress><section class="card"><div class="image"><img id="image"></div><div class="status" id="status"></div><input id="label" autocomplete="off" spellcheck="false" placeholder="Label CAPTCHA"><div class="actions"><button class="save" id="save">Simpan & lanjut</button><button class="skip" id="skip">Skip</button><button class="reject" id="reject">Reject</button></div><div class="hint">Enter: simpan · →: skip · Ctrl+Backspace: reject</div></section></main>
<script>
let current=null;
const image=document.querySelector('#image'), label=document.querySelector('#label'), status=document.querySelector('#status');
async function request(path, options){const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw Error(data.error||'Request gagal');return data}
async function load(){label.value='';label.disabled=true;status.textContent='Memuat gambar...';try{const data=await request('/api/next');current=data.filename;if(!current){image.removeAttribute('src');status.textContent='Semua gambar sudah dilabel atau direject.';document.querySelector('.card').querySelectorAll('button').forEach(x=>x.disabled=true);return}image.src='/image/'+encodeURIComponent(current);document.querySelector('#count').textContent=`${data.done} / ${data.total}`;document.querySelector('#progress').max=data.total;document.querySelector('#progress').value=data.done;status.textContent='Model sedang memprediksi...';const prediction=await request('/api/predict/'+encodeURIComponent(current));label.value=prediction.text;status.textContent='Periksa prediksi, lalu simpan.';label.disabled=false;label.focus();label.select()}catch(error){status.textContent=error.message}}
async function save(){const text=label.value.trim();if(!text)return label.focus();label.disabled=true;await request('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:current,text})});load()}
async function reject(){if(!current)return;label.disabled=true;await request('/api/reject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:current})});load()}
document.querySelector('#save').onclick=save;document.querySelector('#skip').onclick=load;document.querySelector('#reject').onclick=reject;
label.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();save()}else if(event.key==='ArrowRight'){event.preventDefault();load()}else if(event.key==='Backspace'&&event.ctrlKey){event.preventDefault();reject()}};
load();
</script>'''


def read_labels():
    if not LABELS_FILE.exists():
        return {}
    with LABELS_FILE.open(newline="", encoding="utf-8") as file:
        return {row["filename"]: row["text"] for row in csv.DictReader(file)}


def write_labels(labels):
    temporary = LABELS_FILE.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["filename", "text"])
        writer.writerows(sorted(labels.items()))
    temporary.replace(LABELS_FILE)


def images():
    return sorted(path.name for path in IMAGE_DIR.glob("*.png"))


def predict(path):
    global processor, model
    with model_lock:
        if processor is None:
            processor = TrOCRProcessor.from_pretrained(MODEL_ID, use_fast=False)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, early_stopping=False).to(device).eval()
        with Image.open(path) as source:
            pixels = processor(source.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated = model.generate(pixels)
        return processor.batch_decode(generated, skip_special_tokens=True)[0]


class Handler(BaseHTTPRequestHandler):
    def send_data(self, data, status=200, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self.send_data(PAGE.encode(), content_type="text/html; charset=utf-8")
        if path == "/api/next":
            labels = read_labels()
            files = images()
            rejected = list(REJECTED_DIR.glob("*.png")) if REJECTED_DIR.exists() else []
            pending = [name for name in files if name not in labels]
            return self.send_data({"filename": pending[0] if pending else None, "done": len(labels) + len(rejected), "total": len(files) + len(rejected)})
        if path.startswith("/api/predict/"):
            name = unquote(path.removeprefix("/api/predict/"))
            source = self.safe_image(name)
            return self.send_data({"text": predict(source)}) if source else self.send_data({"error": "Gambar tidak ditemukan"}, 404)
        if path.startswith("/image/"):
            name = unquote(path.removeprefix("/image/"))
            source = self.safe_image(name)
            return self.send_data(source.read_bytes(), content_type="image/png") if source else self.send_data({"error": "Gambar tidak ditemukan"}, 404)
        self.send_data({"error": "Not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            source = self.safe_image(data.get("filename", ""))
            if not source:
                return self.send_data({"error": "Gambar tidak ditemukan"}, 404)
            if self.path == "/api/label":
                text = data.get("text", "").strip()
                if not text:
                    return self.send_data({"error": "Label kosong"}, 400)
                labels = read_labels()
                labels[source.name] = text
                write_labels(labels)
                return self.send_data({"ok": True})
            if self.path == "/api/reject":
                REJECTED_DIR.mkdir(exist_ok=True)
                shutil.move(source, REJECTED_DIR / source.name)
                return self.send_data({"ok": True})
            self.send_data({"error": "Not found"}, 404)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_data({"error": str(error)}, 400)

    def safe_image(self, name):
        if Path(name).name != name or not name.lower().endswith(".png"):
            return None
        path = IMAGE_DIR / name
        return path if path.is_file() else None

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"CAPTCHA Labeler: {url}")
    if args.host in {"127.0.0.1", "localhost"}:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

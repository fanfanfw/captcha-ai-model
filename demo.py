import argparse
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "captcha-model" / "best"
MAX_UPLOAD = 5 * 1024 * 1024
model_lock = threading.Lock()
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

PAGE = r'''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAPTCHA Reader</title>
<style>
:root{color-scheme:light;--ink:#172033;--muted:#667085;--paper:#f4f2eb;--panel:#fff;--blue:#3157d5;--line:#d7d9df;--good:#19704b;--bad:#b42318}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}main{width:min(940px,92vw);margin:0 auto;padding:56px 0}header{display:flex;align-items:end;justify-content:space-between;gap:24px;margin-bottom:28px}h1{max-width:600px;margin:0;font-size:clamp(2.2rem,6vw,4.8rem);line-height:.96;letter-spacing:-.04em}header p{max-width:270px;margin:0;color:var(--muted)}.workspace{display:grid;grid-template-columns:1.25fr .75fr;min-height:420px;background:var(--panel);border-radius:16px;box-shadow:0 18px 50px rgba(23,32,51,.12);overflow:hidden}.drop{display:grid;place-items:center;padding:38px;border:3px dashed var(--line);margin:24px;border-radius:12px;cursor:pointer;text-align:center;transition:border-color .2s,background .2s}.drop:hover,.drop.active,.drop:focus-visible{border-color:var(--blue);background:#f5f7ff;outline:none}.drop img{display:none;max-width:100%;max-height:260px}.drop.has-image img{display:block}.drop.has-image .prompt{display:none}.prompt strong{display:block;font-size:1.25rem}.prompt span{display:block;margin-top:7px;color:var(--muted)}aside{display:flex;flex-direction:column;justify-content:center;padding:38px;background:#172033;color:white}.label{color:#adb7cc;font-size:.82rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}.result{min-height:88px;margin:8px 0 20px;font:800 clamp(2.8rem,7vw,5.3rem)/1 ui-monospace,monospace;letter-spacing:.08em}.status{min-height:48px;color:#c8d0df}.status.error{color:#ffb4ab}button{width:100%;padding:14px 18px;border:0;border-radius:10px;background:var(--blue);color:white;font:700 1rem system-ui;cursor:pointer}button:disabled{cursor:not-allowed;opacity:.45}button:focus-visible{outline:3px solid #aabaff;outline-offset:3px}input{position:absolute;opacity:0;pointer-events:none}.meta{margin-top:20px;color:#8f9bb1;font-size:.82rem}@media(max-width:720px){main{padding:30px 0}header{display:block}header p{margin-top:16px}.workspace{grid-template-columns:1fr}.drop{min-height:280px}aside{min-height:280px}}
</style></head><body><main><header><h1>Baca CAPTCHA dalam satu langkah.</h1><p>Unggah PNG atau JPEG. Model lokal memproses gambar tanpa menyimpannya.</p></header><section class="workspace"><label class="drop" id="drop" tabindex="0"><input id="file" type="file" accept="image/png,image/jpeg"><img id="preview" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" alt="Pratinjau CAPTCHA"><span class="prompt"><strong>Taruh gambar di sini</strong><span>atau klik untuk memilih · maksimal 5 MB</span></span></label><aside><span class="label">Hasil prediksi</span><output class="result" id="result">—</output><div class="status" id="status">Pilih gambar untuk mulai.</div><button id="predict" disabled>Prediksi CAPTCHA</button><div class="meta">Model: fine-tuned TrOCR · Output: 5 huruf kapital</div></aside></section></main>
<script>
const file=document.querySelector('#file'),drop=document.querySelector('#drop'),preview=document.querySelector('#preview'),button=document.querySelector('#predict'),result=document.querySelector('#result'),status=document.querySelector('#status');let selected;
function choose(value){if(!value)return;if(!['image/png','image/jpeg'].includes(value.type)){fail('Format harus PNG atau JPEG.');return}if(value.size>5*1024*1024){fail('Ukuran gambar melebihi 5 MB.');return}selected=value;preview.src=URL.createObjectURL(value);drop.classList.add('has-image');button.disabled=false;result.textContent='—';status.className='status';status.textContent=value.name}
function fail(message){status.className='status error';status.textContent=message}
file.onchange=()=>choose(file.files[0]);drop.ondragover=event=>{event.preventDefault();drop.classList.add('active')};drop.ondragleave=()=>drop.classList.remove('active');drop.ondrop=event=>{event.preventDefault();drop.classList.remove('active');choose(event.dataTransfer.files[0])};drop.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();file.click()}};
button.onclick=async()=>{button.disabled=true;button.textContent='Memproses...';status.className='status';status.textContent='Model sedang membaca gambar.';try{const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':selected.type},body:selected});const data=await response.json();if(!response.ok)throw Error(data.error||'Prediksi gagal.');result.textContent=data.text;status.textContent='Prediksi selesai.'}catch(error){fail(error.message)}finally{button.disabled=false;button.textContent='Prediksi CAPTCHA'}};
</script></body></html>'''


def predict(image):
    global processor, model
    with model_lock:
        if processor is None:
            processor = TrOCRProcessor.from_pretrained(MODEL_PATH, use_fast=False)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH, early_stopping=False).to(device).eval()
        pixels = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated = model.generate(pixels, max_length=7, num_beams=1)
        return processor.batch_decode(generated, skip_special_tokens=True)[0]


class Handler(BaseHTTPRequestHandler):
    def send_data(self, data, status=200, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/":
            return self.send_data(PAGE.encode(), content_type="text/html; charset=utf-8")
        self.send_data({"error": "Not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.send_data({"error": "Not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length < 1 or length > MAX_UPLOAD:
                return self.send_data({"error": "Ukuran gambar harus 1 byte sampai 5 MB."}, 400)
            if self.headers.get("Content-Type", "").split(";", 1)[0] not in {"image/png", "image/jpeg"}:
                return self.send_data({"error": "Format harus PNG atau JPEG."}, 415)
            body = self.rfile.read(length)
            with Image.open(io.BytesIO(body)) as image:
                image.verify()
            with Image.open(io.BytesIO(body)) as image:
                text = predict(image)
            self.send_data({"text": text})
        except UnidentifiedImageError:
            return self.send_data({"error": "File bukan gambar valid."}, 400)
        except (ValueError, OSError) as error:
            return self.send_data({"error": str(error)}, 400)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        parser.error(f"checkpoint tidak ditemukan: {MODEL_PATH}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"CAPTCHA Reader: {url}")
    if args.host in {"127.0.0.1", "localhost"}:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

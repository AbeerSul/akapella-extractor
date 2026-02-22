import os
import tempfile
import base64
import logging
from flask import Flask, request, jsonify, send_from_directory, Response
import runpod
from demucs.pretrained import get_model
from demucs.separate import load_track
from demucs.apply import apply_model
import torchaudio
import torch
import uuid
import time
import threading
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Directory to store output stems (served as URLs)
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Output directory: {OUTPUT_DIR}")

 # Optional S3 configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PRESIGN_EXP = int(os.environ.get("S3_PRESIGN_EXP", 86400))  # default 1 day
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_REGION = os.environ.get("S3_REGION")

def upload_file_to_s3(file_path: str, bucket: str, key: str, expire: int = S3_PRESIGN_EXP) -> str:
    """Upload a file to S3 and return a presigned GET URL on success.
    Raises exception on failure.
    """
    s3_kwargs = {}
    if S3_ENDPOINT:
        s3_kwargs["endpoint_url"] = S3_ENDPOINT
    if S3_REGION:
        s3_kwargs["region_name"] = S3_REGION
    s3 = boto3.client("s3", **s3_kwargs)
    last_exception = None
    for attempt in range(1, 4):
        try:
            s3.upload_file(file_path, bucket, key)
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
            logging.info(f"Uploaded {file_path} to s3://{bucket}/{key} (endpoint={S3_ENDPOINT}, region={S3_REGION}) on attempt {attempt}")
            return url
        except (BotoCoreError, ClientError) as e:
            logging.error(f"S3 upload failed for {file_path} to {bucket}/{key} (endpoint={S3_ENDPOINT}, region={S3_REGION}) on attempt {attempt}: {e}")
            last_exception = e
            time.sleep(1)
    logging.exception(f"S3 upload failed after 3 attempts for {file_path} to {bucket}/{key} (endpoint={S3_ENDPOINT}, region={S3_REGION})")
    raise last_exception

def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Removed expired file: {path}")
    except Exception:
        logging.exception(f"Failed to remove file: {path}")

def _schedule_deletion(path: str, delay_seconds: int = 15 * 60):
    try:
        t = threading.Timer(delay_seconds, _safe_remove, args=(path,))
        t.daemon = True
        t.start()
        logging.info(f"Scheduled deletion for {path} in {delay_seconds} seconds")
    except Exception:
        logging.exception(f"Failed to schedule deletion for: {path}")

@app.route('/output/<path:filename>', methods=['GET'])
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

# Health check for load balancer
@app.route("/api/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/ping", methods=["GET"])
def ping_root():
    return "pong", 200

# Main endpoint to separate audio
@app.route("/api/separate", methods=["POST"])
def separate_audio():
    logging.info("Received request to /api/separate")
    logging.info(f"Request files: {list(request.files.keys())}")
    logging.info(f"Request form: {request.form}")
    logging.info(f"Request args: {request.args}")
    global model

    # Lazy-load the model
    if model is None:
        try:
            model = get_model("htdemucs")
            model.to(device)
            model.eval()
            logging.info("htdemucs model loaded successfully")
        except Exception as e:
            logging.exception("Failed to load model")
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    # Check uploaded file
    if 'file' not in request.files:
        logging.warning("No file provided in request.files")
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files['file']
    filename = audio_file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.wav', '.mp3', '.flac']:
        logging.warning(f"Unsupported file format: {ext}")
        return jsonify({"error": "Unsupported file format. Use WAV, MP3, or FLAC."}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, filename)
            audio_file.save(input_path)
            logging.info(f"Saved uploaded file to {input_path}")
            
            # Load the audio
            try:
                wav = load_track(input_path, model.audio_channels, model.samplerate)
                # Add batch dimension if missing
                if wav.ndim == 2:
                    wav = wav.unsqueeze(0)  # -> (1, channels, samples)
                logging.info(f"Loaded audio file, shape: {wav.shape}")
            except Exception as e:
                logging.exception("Failed to load audio file")
                return jsonify({"error": f"Failed to load audio: {str(e)}"}), 500

            # Separate stems
            try:
                sources = apply_model(model, wav, device=device)

                logging.info(f"apply_model returned type: {type(sources)}")

                # Normalize `sources` into a list of tensors shaped (channels, samples)
                normalized = None

                if isinstance(sources, torch.Tensor):
                    logging.info(f"sources is Tensor with shape {sources.shape}, dtype: {sources.dtype}, device: {sources.device}")
                    # Common shapes:
                    # - [num_sources, channels, samples] -> 3D
                    # - [batch, num_sources, channels, samples] -> 4D (batch usually 1)
                    # - [channels, samples] -> 2D (single source)
                    if sources.ndim == 4:
                        # handle batch dimension if present
                        batch = sources.shape[0]
                        if batch == 1:
                            t = sources.squeeze(0)  # -> [num_sources, channels, samples]
                            normalized = [t[i] for i in range(t.shape[0])]
                            logging.info("Squeezed batch dim from 4D tensor")
                        else:
                            raise ValueError(f"apply_model returned batched output with batch={batch}; only batch=1 is supported")
                    elif sources.ndim == 3:
                        normalized = [sources[i] for i in range(sources.shape[0])]
                    elif sources.ndim == 2:
                        # Single source returned as 2D tensor
                        normalized = [sources]
                    else:
                        raise ValueError(f"Unexpected tensor ndim from apply_model: {sources.ndim}")

                elif isinstance(sources, (list, tuple)):
                    logging.info(f"sources is {type(sources)} with length {len(sources)}")
                    # Case: single-element list containing a stacked 3D tensor
                    if len(sources) == 1 and isinstance(sources[0], torch.Tensor) and sources[0].ndim == 3:
                        t = sources[0]
                        logging.info(f"Unwrapping single-item list containing tensor of shape {t.shape}")
                        normalized = [t[i] for i in range(t.shape[0])]
                    else:
                        # If elements are tensors, make a shallow copy
                        normalized = list(sources)

                else:
                    raise ValueError(f"Unexpected return type from apply_model: {type(sources)}")

                # Ensure normalized list contains tensors and log their shapes
                for idx, s in enumerate(normalized):
                    if not isinstance(s, torch.Tensor):
                        raise ValueError(f"Normalized source at index {idx} is not a tensor: {type(s)}")
                    logging.info(f"Normalized source {idx} shape: {s.shape}, dtype: {s.dtype}, device: {s.device}")

                sources = normalized

            except Exception as e:
                logging.exception("Audio separation failed")
                return jsonify({"error": f"Audio separation failed: {str(e)}"}), 500

            # Save stems and encode as Base64
            stems_data = {}
            logging.info(f"About to iterate over {len(sources)} sources and {len(model.sources)} stem names")
            for i, stem in enumerate(model.sources):
                if i >= len(sources):
                    logging.warning(f"Skipping stem '{stem}' – not enough separated tensors")
                    continue
                
                logging.info(f"Processing stem {i}: '{stem}', sources type: {type(sources)}, sources len: {len(sources)}")
                data = sources[i]
                
                logging.info(f"Retrieved sources[{i}], type: {type(data)}, shape: {data.shape if isinstance(data, torch.Tensor) else 'N/A'}")
                logging.info(f"Stem '{stem}' raw shape: {data.shape}, dtype: {data.dtype}, device: {data.device}, ndim: {data.ndim}")

                # Aggressive shape normalization: squeeze all singleton dimensions
                data = data.squeeze()
                
                # If now 1D, add channel dimension
                if data.ndim == 1:
                    data = data.unsqueeze(0)
                
                logging.info(f"Stem '{stem}' after normalization: {data.shape}, ndim: {data.ndim}")
                
                # Validate exactly 2D
                if data.ndim != 2:
                    logging.error(f"CRITICAL: Stem '{stem}' is {data.ndim}D, expected 2D. Shape: {data.shape}")
                    raise ValueError(f"Stem '{stem}' tensor must be 2D (channels, samples), got shape {data.shape}")

                # Ensure tensor is detached, moved to CPU, float32 and contiguous for torchaudio
                data = data.detach().cpu().float().contiguous()

                # Save to persistent output directory with unique name
                out_name = f"{int(time.time())}_{uuid.uuid4().hex}_{stem}.wav"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                logging.info(f"Saving stem '{stem}' to output path: {out_path}")
                torchaudio.save(out_path, data, model.samplerate)

                # Schedule deletion after 15 minutes
                _schedule_deletion(out_path, delay_seconds=15 * 60)

                # Always try S3 first if configured, only fall back to local URL if upload fails
                file_url = None
                if S3_BUCKET:
                    s3_key = out_name
                    try:
                        file_url = upload_file_to_s3(out_path, S3_BUCKET, s3_key)
                        logging.info(f"S3 upload succeeded for {out_name}, url: {file_url}")
                    except Exception as e:
                        logging.error(f"S3 upload failed for {out_name}, falling back to local URL: {e}")
                if not file_url:
                    logging.warning("Abeer, unfortunately S3 is not supported")
                    base = request.host_url.rstrip("/")
                    file_url = f"{base}/output/{out_name}"
                stems_data[stem] = file_url
                logging.info(f"Stem '{stem}' final url: {file_url}")

            return jsonify({"stems": stems_data})

    except Exception as e:
        logging.exception("Unexpected error")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)


# ========== API: /api/process_audio ==========
import shutil
import subprocess
import json
from flask import send_file

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
PROCESSED_FOLDER = os.path.join(os.getcwd(), "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def safe_delete(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logging.error(f"Delete failed: {e}")

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    audio_file = request.files['audio']
    filename = f"{int(time.time())}-{uuid.uuid4().hex}.mp3"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(input_path)

    effects = request.form.get('effects') or (request.json.get('effects') if request.is_json else None)
    try:
        effects = json.loads(effects) if isinstance(effects, str) else effects
    except Exception:
        safe_delete(input_path)
        return jsonify({'error': 'Invalid effects JSON'}), 400

    if not effects or not isinstance(effects, dict) or not effects.keys():
        safe_delete(input_path)
        return jsonify({'error': 'No effects specified'}), 400

    output_path = os.path.join(PROCESSED_FOLDER, filename)
    filters = []
    # Port filter logic
    if effects.get('trimTime', {}).get('enabled'):
        start = effects['trimTime'].get('start', 0)
        end = effects['trimTime'].get('end')
        if end is not None:
            filters.append(f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS")
        else:
            filters.append(f"atrim=start={start},asetpts=PTS-STARTPTS")
    if effects.get('trimSilence', {}).get('enabled'):
        threshold = effects['trimSilence'].get('threshold', '-60dB')
        filters.append(f"silenceremove=start_periods=1:start_threshold={threshold}:start_silence=0.3:stop_periods=-1:stop_threshold={threshold}:stop_silence=0.3")
    if effects.get('louder', {}).get('enabled'):
        gain = effects['louder'].get('gain', 1.8)
        filters.append(f"volume={gain}")
    if effects.get('echo', {}).get('enabled'):
        delay = effects['echo'].get('delay', 800)
        decay = effects['echo'].get('decay', 0.3)
        filters.append(f"aecho=0.8:0.9:{delay}:{decay}")

    if not filters:
        safe_delete(input_path)
        return jsonify({'error': 'No valid effects enabled'}), 400

    ffmpeg_bin = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    args = ['-y', '-i', input_path]
    args += ['-af', ','.join(filters)]
    args += [output_path]
    try:
        proc = subprocess.run([ffmpeg_bin] + args, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(output_path):
            safe_delete(input_path)
            safe_delete(output_path)
            return jsonify({'error': 'FFmpeg failed'}), 500
        return send_file(output_path, as_attachment=True)
    finally:
        safe_delete(input_path)
        safe_delete(output_path)


# ========== API: /api/mix ==========
from werkzeug.utils import secure_filename
import shutil
import subprocess
from flask import send_file

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
PROCESSED_FOLDER = os.path.join(os.getcwd(), "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def safe_delete(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logging.error(f"Delete failed: {e}")

@app.route('/api/mix', methods=['POST'])
def api_mix():
    files = request.files.getlist('tracks')
    if not files or len(files) < 2:
        return jsonify({'error': 'Need at least 2 tracks'}), 400

    input_paths = []
    for f in files:
        fname = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}-{uuid.uuid4().hex}-{fname}")
        f.save(path)
        input_paths.append(path)

    output_path = os.path.join(PROCESSED_FOLDER, f"mixed-{int(time.time())}.mp3")
    ffmpeg_bin = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    args = []
    for p in input_paths:
        args += ['-i', p]
    filter_complex = f"{' '.join([f'[{i}:a]' for i in range(len(input_paths))])}amix=inputs={len(input_paths)}:duration=longest[a]"
    args += ['-filter_complex', filter_complex, '-map', '[a]', '-c:a', 'libmp3lame', '-b:a', '192k', output_path]
    try:
        proc = subprocess.run([ffmpeg_bin] + args, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(output_path):
            for p in input_paths:
                safe_delete(p)
            safe_delete(output_path)
            return jsonify({'error': 'Mixing error'}), 500
        return send_file(output_path, as_attachment=True)
    finally:
        for p in input_paths:
            safe_delete(p)
        safe_delete(output_path)


# ========== API: /api/trim-merge-audio ==========
import json

@app.route('/api/trim-merge-audio', methods=['POST'])
def trim_merge_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    audio_file = request.files['audio']
    filename = f"{int(time.time())}-{uuid.uuid4().hex}.mp3"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(input_path)

    exclude_ranges = request.form.get('exclude_ranges') or (request.json.get('exclude_ranges') if request.is_json else None)
    try:
        exclude_ranges = json.loads(exclude_ranges) if isinstance(exclude_ranges, str) else exclude_ranges
        if not isinstance(exclude_ranges, list):
            raise ValueError('exclude_ranges must be array')
    except Exception:
        safe_delete(input_path)
        return jsonify({'error': 'Invalid exclude_ranges JSON'}), 400

    output_path = os.path.join(PROCESSED_FOLDER, filename)
    ffmpeg_bin = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'

    # Get duration using ffprobe
    try:
        probe = subprocess.run([
            ffmpeg_bin.replace('ffmpeg', 'ffprobe'),
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ], capture_output=True, text=True)
        duration = float(probe.stdout.strip())
    except Exception:
        safe_delete(input_path)
        return jsonify({'error': 'Failed to read duration'}), 500

    exclude_ranges.sort(key=lambda r: r.get('start', 0))
    keep_ranges = []
    current = 0
    for r in exclude_ranges:
        start = max(0, r.get('start', 0))
        end = min(duration, r.get('end', duration))
        if start > current:
            keep_ranges.append({'start': current, 'end': start})
        current = max(current, end)
    if current < duration:
        keep_ranges.append({'start': current, 'end': duration})
    if not keep_ranges:
        safe_delete(input_path)
        return jsonify({'error': 'All audio removed'}), 400

    filter_parts = [
        f"[0:a]atrim=start={r['start']}:end={r['end']},asetpts=PTS-STARTPTS[a{i}]"
        for i, r in enumerate(keep_ranges)
    ]
    concat_inputs = ''.join([f"[a{i}]" for i in range(len(keep_ranges))])
    concat_part = f"{concat_inputs}concat=n={len(keep_ranges)}:v=0:a=1[out]"
    filter_complex = ';'.join(filter_parts + [concat_part])

    args = [
        '-y', '-i', input_path,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-acodec', 'mp3',
        output_path
    ]
    try:
        proc = subprocess.run([ffmpeg_bin] + args, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(output_path):
            safe_delete(input_path)
            safe_delete(output_path)
            return jsonify({'error': 'FFmpeg trim-merge failed'}), 500
        return send_file(output_path, as_attachment=True)
    finally:
        safe_delete(input_path)
        safe_delete(output_path)


# ========== API: /api/upload ==========
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    audio_file = request.files['audio']
    filename = f"{int(time.time())}-{uuid.uuid4().hex}.mp3"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)
    # Build public URL (assume /uploads is served statically)
    base_url = request.host_url.rstrip('/')
    file_url = None
    if S3_BUCKET:
        s3_key = filename
        try:
            file_url = upload_file_to_s3(file_path, S3_BUCKET, s3_key)
        except Exception as e:
            logging.error(f"S3 upload failed for {filename}, falling back to local URL: {e}")
    if not file_url:
        file_url = f"{base_url}/uploads/{filename}"
    return jsonify({'success': True, 'filename': filename, 'url': file_url})


# ========== API: /api/separate (sync & async) ==========
from flask import Response
import threading

separate_queue = []
SEPARATE_MAX_QUEUE = 8
DEMUCS_WORKERS = 2
active_workers = 0
separate_jobs = {}  # jobId -> { id, status, inputPath, result, error, createdAt, _resolve, _reject }
separate_subscribers = {}  # jobId -> [res, ...]

# Helper: run demucs separation (dummy placeholder, replace with real logic as needed)
def run_demucs(input_path, output_dir):
    # For demo, just copy input to output as 'vocals.wav', 'drums.wav', etc.
    os.makedirs(output_dir, exist_ok=True)
    for stem in ['vocals', 'drums', 'bass', 'other']:
        out_path = os.path.join(output_dir, f'{stem}.wav')
        with open(input_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            f_out.write(f_in.read())

# Worker pool

def process_separate_queue():
    global active_workers
    while active_workers < DEMUCS_WORKERS and separate_queue:
        job = separate_queue.pop(0)
        if not job:
            break
        active_workers += 1
        def worker(job):
            global active_workers
            try:
                if not job['inputPath']:
                    raise Exception('No input path for job')
                input_path = job['inputPath']
                output_dir = os.path.join(os.getcwd(), 'separated', job['id'])
                job['status'] = 'processing'
                run_demucs(input_path, output_dir)
                base_url = job.get('baseUrlFromReq')
                public_path = f"{base_url}/files/htdemucs/{job['id']}"
                job['status'] = 'done'
                job['result'] = {
                    'vocals': f"{public_path}/vocals.wav",
                    'drums': f"{public_path}/drums.wav",
                    'bass': f"{public_path}/bass.wav",
                    'other': f"{public_path}/other.wav"
                }
                # notify SSE subscribers
                subs = separate_subscribers.get(job['id'], [])
                payload = json.dumps({'job_id': job['id'], 'result': job['result']})
                for r in subs:
                    try:
                        r.write(f"event: done\ndata: {payload}\n\n")
                        r.close()
                    except Exception:
                        pass
                separate_subscribers.pop(job['id'], None)
                if job.get('_resolve'):
                    job['_resolve'](job['result'])
            except Exception as err:
                job['status'] = 'error'
                job['error'] = str(err)
                if job.get('_reject'):
                    job['_reject'](err)
                # notify SSE subscribers about error
                subs = separate_subscribers.get(job['id'], [])
                payload = json.dumps({'job_id': job['id'], 'error': job['error']})
                for r in subs:
                    try:
                        r.write(f"event: error\ndata: {payload}\n\n")
                        r.close()
                    except Exception:
                        pass
                separate_subscribers.pop(job['id'], None)
            finally:
                safe_delete(job['inputPath'])
                active_workers -= 1
                threading.Timer(0, process_separate_queue).start()
        threading.Thread(target=worker, args=(job,)).start()

# RunPod handler for /api/separate/async
def runpod_separate_async_handler(event):
    # event['input'] should contain the file data and metadata
    # Accept both 'audio' and 'file' as upload field
    file_data = event['input'].get('audio') or event['input'].get('file')
    if not file_data:
        return {'error': 'No file uploaded'}
       
@app.route('/api/separate/async', methods=['POST'])
def api_separate_async():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    if len(separate_queue) >= SEPARATE_MAX_QUEUE:
        return jsonify({'error': 'Server busy. Try again later'}), 429
    audio_file = request.files['audio']
    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.wav")
    audio_file.save(input_path)
    job = {
        'id': job_id,
        'status': 'queued',
        'inputPath': input_path,
        'createdAt': time.time(),
        'result': None,
        'error': None,
        'baseUrlFromReq': f"{request.scheme}://{request.host}"
    }
    separate_jobs[job_id] = job
    separate_queue.append(job)
    process_separate_queue()
    return {'success': True, 'job_id': job_id, 'status': 'queued'}

# Register handler with RunPod (handle mode for LB API)
runpod.serverless.handle(runpod_separate_async_handler)

@app.route('/api/separate', methods=['POST'])
def api_separate():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    if len(separate_queue) >= SEPARATE_MAX_QUEUE:
        return jsonify({'error': 'Server busy. Try again later'}), 429
    audio_file = request.files['audio']
    async_mode = request.args.get('async') == 'true' or request.form.get('async') == 'true'
    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.wav")
    audio_file.save(input_path)
    job = {
        'id': job_id,
        'status': 'queued',
        'inputPath': input_path,
        'createdAt': time.time(),
        'result': None,
        'error': None,
        'baseUrlFromReq': f"{request.scheme}://{request.host}"
    }
    job['_done'] = threading.Event()
    def resolve(result):
        job['result'] = result
        job['_done'].set()
    def reject(err):
        job['error'] = str(err)
        job['_done'].set()
    job['_resolve'] = resolve
    job['_reject'] = reject
    separate_jobs[job_id] = job
    separate_queue.append(job)
    process_separate_queue()
    if async_mode:
        return jsonify({'success': True, 'job_id': job_id, 'status': job['status']})
    job['_done'].wait()
    if job['status'] == 'done':
        return jsonify({'success': True, 'job_id': job_id, 'result': job['result']})
    else:
        return jsonify({'success': False, 'job_id': job_id, 'error': job['error']}), 500

@app.route('/api/separate/status/<job_id>', methods=['GET'])
def api_separate_status(job_id):
    job = separate_jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    return jsonify({'success': True, 'job_id': job_id, 'status': job['status'], 'error': job.get('error')})

@app.route('/api/separate/result/<job_id>', methods=['GET'])
def api_separate_result(job_id):
    job = separate_jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    if job['status'] == 'done':
        return jsonify({'success': True, 'job_id': job_id, 'result': job['result']})
    if job['status'] == 'error':
        return jsonify({'success': False, 'job_id': job_id, 'error': job['error']}), 500
    return jsonify({'success': False, 'job_id': job_id, 'status': job['status']}), 202

@app.route('/api/separate/stream/<job_id>', methods=['GET'])
def api_separate_stream(job_id):
    job = separate_jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    def event_stream():
        if job['status'] == 'done':
            yield f"event: done\ndata: {json.dumps({'job_id': job_id, 'result': job['result']})}\n\n"
        elif job['status'] == 'error':
            yield f"event: error\ndata: {json.dumps({'job_id': job_id, 'error': job['error']})}\n\n"
        else:
            separate_subscribers.setdefault(job_id, []).append(Response(event_stream(), mimetype='text/event-stream'))
            while True:
                time.sleep(1)
    return Response(event_stream(), mimetype='text/event-stream')

# ========== API: /api/merge_drums ==========
import requests

@app.route('/api/merge_drums', methods=['POST'])
def api_merge_drums():
    logging.info("Received request to /api/merge_drums")
    logging.info(f"Request JSON: {request.json}")
    try:
        input_urls = [
            (name, url.strip())
            for name, url in (request.json or {}).items()
            if isinstance(url, str) and url.strip()
        ]
        if not input_urls:
            logging.warning("No valid audio URLs provided in request.json")
            return jsonify({'success': False, 'error': 'No valid audio URLs provided'}), 400
        session_id = str(uuid.uuid4())
        temp_paths = []
        for name, url in input_urls:
            temp_path = os.path.join(TEMP_DIR, f"{session_id}_{name}.mp3")
            r = requests.get(url)
            if r.status_code != 200:
                logging.error(f"Failed to download file from {url}, status code: {r.status_code}")
                raise Exception('Failed to download file')
            with open(temp_path, 'wb') as f:
                f.write(r.content)
            temp_paths.append(temp_path)
            logging.info(f"Downloaded and saved {name} to {temp_path}")
        output_filename = f"{int(time.time())}-{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        ffmpeg_bin = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
        args = []
        for p in temp_paths:
            args += ['-i', p]
        filter = ''.join([f'[{i}:a]' for i in range(len(temp_paths))]) + f"amix=inputs={len(temp_paths)}:duration=longest:dropout_transition=2[out]"
        args += ['-filter_complex', filter, '-map', '[out]', '-b:a', '192k', output_path]
        proc = subprocess.run([ffmpeg_bin] + args, capture_output=True)
        if proc.returncode != 0 or not os.path.exists(output_path):
            logging.error(f"FFmpeg mixing error, return code: {proc.returncode}")
            for p in temp_paths:
                safe_delete(p)
            safe_delete(output_path)
            return jsonify({'success': False, 'error': 'Mixing error'}), 500
        logging.info(f"FFmpeg mixing succeeded, output at {output_path}")
        for p in temp_paths:
            safe_delete(p)
        file_url = f"{request.host_url.rstrip('/')}/uploads/{output_filename}"
        return jsonify({'success': True, 'mp3_url': file_url})
    except Exception as error:
        logging.exception(f"Exception in /api/merge_drums: {error}")
        return jsonify({'success': False, 'error': str(error) or 'Internal server error'}), 500

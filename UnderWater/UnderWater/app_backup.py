from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import threading
import time
import pygame
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained YOLO model
model = YOLO('best.pt')

# Class names from your training
CLASS_NAMES = ['eel', 'fish', 'jellyfish', 'lionfish', 'lobster']

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for video streaming
video_camera = None
detection_enabled = True
alarm_sound_enabled = True
last_detection_time = 0
alarm_cooldown = 3  # seconds between alarm sounds

# Initialize pygame mixer for alarm sound
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Generate a simple alarm sound programmatically
def generate_alarm_sound():
    """Generate a simple alarm sound programmatically"""
    sample_rate = 22050
    duration = 0.5
    frequency = 800
    
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2), dtype=np.int16)
    
    for i in range(frames):
        t = float(i) / sample_rate
        wave = np.sin(2 * np.pi * frequency * t)
        arr[i] = [int(wave * 32767), int(wave * 32767)]
    
    sound = pygame.sndarray.make_sound(arr)
    return sound

# Create alarm sound
alarm_sound = generate_alarm_sound()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def play_alarm():
    """Play alarm sound"""
    if alarm_sound_enabled:
        try:
            alarm_sound.play()
        except:
            pass

def detect_species(frame):
    """Detect species in a frame using YOLO model"""
    try:
        results = model(frame, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if hasattr(boxes, 'cpu'):
                        boxes = boxes.cpu()
                    
                    if hasattr(boxes, 'numpy'):
                        boxes = boxes.numpy()
                    
                    if hasattr(boxes, '__len__') and len(boxes) > 0:
                        for box in boxes:
                            try:
                                # Get bounding box coordinates
                                if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                    bbox = box.xyxy[0]
                                    if hasattr(bbox, 'cpu'):
                                        bbox = bbox.cpu()
                                    if hasattr(bbox, 'numpy'):
                                        bbox = bbox.numpy()
                                    
                                    if hasattr(bbox, '__len__') and len(bbox) == 4:
                                        x1, y1, x2, y2 = bbox
                                        
                                        # Get confidence score
                                        if hasattr(box, 'conf') and len(box.conf) > 0:
                                            conf = box.conf[0]
                                            if hasattr(conf, 'cpu'):
                                                conf = conf.cpu()
                                            if hasattr(conf, 'numpy'):
                                                conf = conf.numpy()
                                            confidence = float(conf)
                                            
                                            # Get class name
                                            if hasattr(box, 'cls') and len(box.cls) > 0:
                                                cls = box.cls[0]
                                                if hasattr(cls, 'cpu'):
                                                    cls = cls.cpu()
                                                if hasattr(cls, 'numpy'):
                                                    cls = cls.numpy()
                                                class_id = int(cls)
                                                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
                                                
                                                detections.append({
                                                    'class': class_name,
                                                    'confidence': confidence,
                                                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                                                })
                            except Exception as e:
                                continue
        return detections
    except Exception as e:
        return []

def process_video_simple(video_path):
    """Simple video processing - just extract first frame for detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file"
        
        # Just read the first frame for detection
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, "Could not read first frame from video"
        
        # Detect species in first frame only
        detections = detect_species(frame)
        
        # Play alarm if species detected
        if len(detections) > 0:
            play_alarm()
        
        # Draw detections on frame
        frame_with_detections = draw_detections(frame, detections)
        
        cap.release()
        
        # Return single frame with detections
        return [{
            'frame': frame_with_detections,
            'timestamp': "0.00s",
            'detections': detections
        }], detections
    except Exception as e:
        return None, f"Simple video processing error: {str(e)}"

def create_video_summary(processed_frames, output_path):
    """Create a summary video with detections"""
    if not processed_frames:
        return None
    
    # Get frame dimensions from first frame
    height, width = processed_frames[0]['frame'].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    for frame_data in processed_frames:
        out.write(frame_data['frame'])
    
    out.release()
    return output_path

def draw_detections(frame, detections):
    """Draw detection boxes and labels on frame"""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw background for label
        cv2.rectangle(frame, 
                     (int(x1), int(y1) - label_size[1] - 10),
                     (int(x1) + label_size[0], int(y1)),
                     (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

class VideoCamera:
    def __init__(self):
        self.video = None
        self.lock = threading.Lock()
        self.is_running = False
    
    def start(self):
        """Start video capture"""
        if self.video is None or not self.video.isOpened():
            self.video = cv2.VideoCapture(0)
            # Optimize camera settings for better performance
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Reduce auto exposure and focus for faster response
            self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.video.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.is_running = True
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.video and self.video.isOpened():
            self.video.release()
            self.video = None
    
    def get_frame(self):
        """Get frame from camera with detections"""
        global detection_enabled, last_detection_time
        
        if self.video is None or not self.video.isOpened() or not self.is_running:
            return None
        
        # Use grab() and retrieve() for faster frame capture
        if not self.video.grab():
            return None
            
        ret, frame = self.video.retrieve()
        if not ret:
            return None
        
        if detection_enabled:
            # Perform detection
            detections = detect_species(frame)
            
            # Check if any species detected and play alarm
            current_time = time.time()
            if len(detections) > 0 and (current_time - last_detection_time) > alarm_cooldown:
                play_alarm()
                last_detection_time = current_time
            
            # Draw detections on frame
            frame = draw_detections(frame, detections)
        
        return frame

# Initialize video camera
video_camera = VideoCamera()

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/detection')
def detection():
    """Detection page"""
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate_frames():
        video_camera.start()
        try:
            while True:
                frame = video_camera.get_frame()
                if frame is None:
                    break
                
                # Encode frame to JPEG with lower quality for faster transmission
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            video_camera.stop()
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded video temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            file.save(temp_path)

            # Process video for detections using simple approach
            processed_frames, all_detections = process_video_simple(temp_path)

            if processed_frames is None:
                error_msg = all_detections if all_detections else "Unknown video processing error"
                return jsonify({'error': f'Video processing failed: {error_msg}'}), 500

            # Get first frame with detections as preview
            preview_base64 = ""
            try:
                if len(processed_frames) > 0 and processed_frames[0]['frame'] is not None:
                    frame_rgb = cv2.cvtColor(processed_frames[0]['frame'], cv2.COLOR_BGR2RGB)
                    pil_preview = Image.fromarray(frame_rgb)
                    buffered = BytesIO()
                    pil_preview.save(buffered, format="JPEG")
                    preview_base64 = base64.b64encode(buffered.getvalue()).decode()
                    preview_base64 = f"data:image/jpeg;base64,{preview_base64}"
            except Exception as e:
                pass
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            response_data = {
                'success': True,
                'detections': all_detections,
                'preview_image': preview_base64,
                'summary_video': "",
                'total_detections': len(all_detections),
                'message': f"Found {len(all_detections)} species in video"
            }
            
            return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and detection"""
    try:
        # ... (rest of the function remains the same)
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read image
            image_stream = BytesIO()
            file.save(image_stream)
            image_stream.seek(0)
            
            # Convert to OpenCV format
            pil_image = Image.open(image_stream)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Perform detection
            detections = detect_species(frame)
            
            # Play alarm if species detected
            if len(detections) > 0:
                play_alarm()
            
            # Draw detections on image
            frame_with_detections = draw_detections(frame, detections)
            
            # Convert back to PIL Image
            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(frame_rgb)
            
            # Convert to base64 for JSON response
            buffered = BytesIO()
            pil_result.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'detections': detections,
                'image': f'data:image/jpeg;base64,{img_str}'
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off"""
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({'detection_enabled': detection_enabled})

@app.route('/toggle_alarm', methods=['POST'])
def toggle_alarm():
    """Toggle alarm sound on/off"""
    global alarm_sound_enabled
    alarm_sound_enabled = not alarm_sound_enabled
    return jsonify({'alarm_enabled': alarm_sound_enabled})

@app.route('/get_status')
def get_status():
    """Get current status"""
    return jsonify({
        'detection_enabled': detection_enabled,
        'alarm_enabled': alarm_sound_enabled
    })

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm():
    """Stop alarm sound immediately"""
    try:
        pygame.mixer.stop()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_alarm')
def test_alarm():
    """Test alarm sound"""
    play_alarm()
    return jsonify({'success': True})

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        if video_camera:
            video_camera.stop()
        pygame.mixer.quit()
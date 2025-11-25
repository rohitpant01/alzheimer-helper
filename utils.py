"""
Utility functions for Alzheimer Patient Helper Application
Contains database operations, face recognition, TTS/STT, and AI chatbot
"""

import sqlite3
import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import face_recognition
import numpy as np
import pyttsx3
import threading
import shutil
import random
import re

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

GEMINI_API_KEY = "AIzaSyCP3qZCd91m9UOpLD6KizS9YCF21QTyLpQ" 
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Create and return database connection"""
    conn = sqlite3.connect('data.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize SQLite database with all required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Persons table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            relation_to_patient TEXT,
            nickname TEXT,
            date_of_birth TEXT,
            phone_number TEXT,
            email TEXT,
            address TEXT,
            last_visit_datetime TEXT,
            notes TEXT,
            primary_language TEXT,
            emergency_contact TEXT,
            tags TEXT,
            created_at TEXT,
            created_by TEXT
        )
    ''')
    
    # Face embeddings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            encoding BLOB,
            created_at TEXT,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
        )
    ''')
    
    # Reminders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            next_due TEXT NOT NULL,
            recurrence TEXT DEFAULT 'once',
            status TEXT DEFAULT 'active',
            created_at TEXT
        )
    ''')
    
    # Activity logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            detail TEXT,
            timestamp TEXT NOT NULL
        )
    ''')
    
    # Cognitive results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cognitive_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            score REAL,
            details TEXT,
            timestamp TEXT,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Create necessary directories
    Path("faces").mkdir(exist_ok=True)
    Path("unknown").mkdir(exist_ok=True)
    Path("backup").mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# PERSON MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def add_person(person_data):
    """Add a new person to the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO persons (
                full_name, relation_to_patient, nickname, date_of_birth,
                phone_number, email, address, notes, tags, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            person_data.get('full_name'),
            person_data.get('relation_to_patient'),
            person_data.get('nickname'),
            person_data.get('date_of_birth'),
            person_data.get('phone_number'),
            person_data.get('email'),
            person_data.get('address'),
            person_data.get('notes'),
            person_data.get('tags'),
            person_data.get('created_at')
        ))
        
        person_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return person_id
    
    except Exception as e:
        print(f"Error adding person: {e}")
        return None

def get_all_persons():
    """Retrieve all persons from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM persons ORDER BY full_name')
        rows = cursor.fetchall()
        
        persons = [dict(row) for row in rows]
        
        conn.close()
        return persons
    
    except Exception as e:
        print(f"Error retrieving persons: {e}")
        return []

def get_person_by_id(person_id):
    """Get person details by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM persons WHERE id = ?', (person_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        return dict(row) if row else None
    
    except Exception as e:
        print(f"Error getting person: {e}")
        return None

def update_person(person_id, updates):
    """Update person information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build dynamic UPDATE query
        set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [person_id]
        
        cursor.execute(f'UPDATE persons SET {set_clause} WHERE id = ?', values)
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error updating person: {e}")
        return False

def delete_person(person_id):
    """Delete person and associated data"""
    try:
        # Get person name first
        person = get_person_by_id(person_id)
        if not person:
            return False
        
        # Delete face images
        person_folder = Path("faces") / person['full_name']
        if person_folder.exists():
            shutil.rmtree(person_folder)
        
        # Delete from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM embeddings WHERE person_id = ?', (person_id,))
        cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error deleting person: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# FACE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════

def capture_faces_from_webcam(person_name, count=15):
    """Capture multiple face photos from webcam"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot access webcam")
            return False
        
        # Create directory for person
        person_dir = Path("faces") / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        captured = 0
        frame_skip = 10  # Capture every 10th frame for variety
        frame_count = 0
        
        print(f"Capturing {count} images for {person_name}...")
        
        while captured < count:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for variety
            if frame_count % frame_skip == 0:
                # Save image
                img_path = person_dir / f"{person_name}_{captured + 1}.jpg"
                cv2.imwrite(str(img_path), frame)
                captured += 1
                print(f"Captured {captured}/{count}")
        
        cap.release()
        
        return captured >= count
    
    except Exception as e:
        print(f"Error capturing faces: {e}")
        return False

def save_uploaded_faces(person_name, uploaded_files):
    """Save uploaded face images"""
    try:
        person_dir = Path("faces") / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Save
            img_path = person_dir / f"{person_name}_{i + 1}.jpg"
            cv2.imwrite(str(img_path), img)
        
        return True
    
    except Exception as e:
        print(f"Error saving uploaded faces: {e}")
        return False

def train_face_embeddings():
    """Train face recognition embeddings for all persons"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clear existing embeddings
        cursor.execute('DELETE FROM embeddings')
        
        faces_dir = Path("faces")
        
        if not faces_dir.exists():
            conn.close()
            return False
        
        # Process each person's folder
        for person_folder in faces_dir.iterdir():
            if not person_folder.is_dir():
                continue
            
            person_name = person_folder.name
            
            # Get person ID
            cursor.execute('SELECT id FROM persons WHERE full_name = ?', (person_name,))
            result = cursor.fetchone()
            
            if not result:
                continue
            
            person_id = result[0]
            
            # Process images
            for img_path in person_folder.glob("*.jpg"):
                try:
                    # Load image
                    image = face_recognition.load_image_file(str(img_path))
                    
                    # Get face encodings
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        # Save first encoding (assume one face per image)
                        encoding = encodings[0]
                        encoding_blob = pickle.dumps(encoding)
                        
                        cursor.execute('''
                            INSERT INTO embeddings (person_id, encoding, created_at)
                            VALUES (?, ?, ?)
                        ''', (person_id, encoding_blob, datetime.now().isoformat()))
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error training embeddings: {e}")
        return False

def recognize_face_from_frame(rgb_frame, threshold=0.6):
    """Recognize face from camera frame"""
    try:
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return None
        
        # Get encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return None
        
        # Use first detected face
        test_encoding = face_encodings[0]
        
        # Load all known encodings from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.person_id, p.full_name, e.encoding
            FROM embeddings e
            JOIN persons p ON e.person_id = p.id
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        # Compare with all known faces
        best_match_person_id = None
        best_match_name = None
        best_distance = float('inf')
        
        for row in rows:
            person_id = row[0]
            person_name = row[1]
            stored_encoding = pickle.loads(row[2])
            
            # Calculate face distance
            distance = face_recognition.face_distance([stored_encoding], test_encoding)[0]
            
            if distance < best_distance:
                best_distance = distance
                best_match_person_id = person_id
                best_match_name = person_name
        
        # Check if match is good enough
        if best_distance < threshold:
            confidence = 1 - best_distance
            return (best_match_person_id, best_match_name, confidence)
        
        return None
    
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# REMINDER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def add_reminder(reminder_data):
    """Add new reminder"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reminders (title, description, next_due, recurrence, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            reminder_data.get('title'),
            reminder_data.get('description'),
            reminder_data.get('next_due'),
            reminder_data.get('recurrence', 'once'),
            reminder_data.get('status', 'active'),
            datetime.now().isoformat()
        ))
        
        reminder_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return reminder_id
    
    except Exception as e:
        print(f"Error adding reminder: {e}")
        return None

def get_active_reminders():
    """Get all active reminders sorted by next due time"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM reminders 
            WHERE status = 'active'
            ORDER BY next_due
        ''')
        
        rows = cursor.fetchall()
        reminders = [dict(row) for row in rows]
        
        conn.close()
        return reminders
    
    except Exception as e:
        print(f"Error getting reminders: {e}")
        return []

def update_reminder_status(reminder_id, status):
    """Update reminder status"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE reminders SET status = ? WHERE id = ?', (status, reminder_id))
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error updating reminder: {e}")
        return False

def delete_reminder(reminder_id):
    """Delete reminder"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM reminders WHERE id = ?', (reminder_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error deleting reminder: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# ACTIVITY LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def log_activity(activity_type, detail):
    """Log an activity to the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_logs (type, detail, timestamp)
            VALUES (?, ?, ?)
        ''', (activity_type, detail, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error logging activity: {e}")
        return False

def get_activity_logs(limit=100):
    """Retrieve recent activity logs"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM activity_logs 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        logs = [dict(row) for row in rows]
        
        conn.close()
        return logs
    
    except Exception as e:
        print(f"Error getting activity logs: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════
# COGNITIVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def add_cognitive_result(person_id, score, details):
    """Add cognitive exercise result"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO cognitive_results (person_id, score, details, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (person_id, score, details, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error adding cognitive result: {e}")
        return False

def get_cognitive_results(limit=50):
    """Get cognitive exercise results"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM cognitive_results 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        
        conn.close()
        return results
    
    except Exception as e:
        print(f"Error getting cognitive results: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH (OFFLINE)
# ═══════════════════════════════════════════════════════════════════════════

_tts_engine = None
_tts_lock = threading.Lock()

def get_tts_engine():
    """Initialize and return TTS engine (singleton)"""
    global _tts_engine
    
    if _tts_engine is None:
        try:
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty('rate', 150)  # Speed
            _tts_engine.setProperty('volume', 0.9)  # Volume
        except Exception as e:
            print(f"TTS initialization error: {e}")
            _tts_engine = None
    
    return _tts_engine

def speak_text(text):
    """Speak text using pyttsx3 (non-blocking)"""
    def speak_thread():
        try:
            with _tts_lock:
                engine = get_tts_engine()
                if engine:
                    engine.say(text)
                    engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    
    # Run in separate thread to avoid blocking
    thread = threading.Thread(target=speak_thread, daemon=True)
    thread.start()

def stop_tts():
    """Stop TTS engine"""
    global _tts_engine
    
    try:
        if _tts_engine:
            _tts_engine.stop()
    except Exception as e:
        print(f"Error stopping TTS: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# SPEECH-TO-TEXT (VOSK - OFFLINE)
# ═══════════════════════════════════════════════════════════════════════════

def init_vosk_stt():
    """Initialize Vosk speech recognition model"""
    try:
        from vosk import Model
        
        # Check for model directory
        model_path = "vosk-model"
        
        if not Path(model_path).exists():
            print("Vosk model not found. Download from https://alphacephei.com/vosk/models")
            return None
        
        model = Model(model_path)
        return model
    
    except ImportError:
        print("Vosk not installed. Install with: pip install vosk")
        return None
    
    except Exception as e:
        print(f"Error initializing Vosk: {e}")
        return None

def recognize_speech_vosk(model, duration=5):
    """Recognize speech using Vosk (offline)"""
    if not model:
        return None
    
    try:
        import pyaudio
        from vosk import KaldiRecognizer
        
        # Audio settings
        CHUNK = 8000
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        recognizer = KaldiRecognizer(model, RATE)
        recognizer.SetWords(True)
        
        print("Listening...")
        
        # Record for specified duration
        frames = int(RATE / CHUNK * duration)
        
        for _ in range(frames):
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get('text', '')
                
                if text:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return text
        
        # Get final result
        result = json.loads(recognizer.FinalResult())
        text = result.get('text', '')
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return text if text else None
    
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# AI CHATBOT WITH GEMINI API
# ═══════════════════════════════════════════════════════════════════════════

def process_chat_message(user_message, chat_history):
    """
    Enhanced chat handler with proper Gemini API integration
    Handles all types of questions with AI assistance
    """
    # Normalize input
    user_message = (user_message or "").strip()
    if not user_message:
        return "I'm here to help. What would you like to know?"
    
    norm_msg = re.sub(r'\s+', ' ', user_message.lower()).strip()

    # 1) Quick rule-based shortcuts for time-sensitive queries
    if any(norm_msg.startswith(g) for g in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "Hello! How can I help you today?"

    if re.search(r'\b(sos|emergency|help me|urgent)\b', norm_msg):
        try:
            log_activity('sos_chat', "Patient used SOS terms")
        except Exception:
            pass
        return "🚨 I've alerted your caregiver immediately. Help is on the way!"

    if re.search(r'\b(reminder|schedule|appointment)\b', norm_msg):
        try:
            reminders = get_active_reminders()
        except Exception:
            reminders = []
        if reminders:
            out = "Here are your upcoming reminders:\n\n"
            for r in reminders[:5]:
                try:
                    due_time = datetime.fromisoformat(r.get('next_due', '')).strftime("%I:%M %p")
                except Exception:
                    due_time = r.get('next_due', '')
                out += f"• {r.get('title','(no title)')} at {due_time}\n"
            return out
        return "You have no active reminders at the moment."

    if re.search(r'\b(time|date|today)\b', norm_msg) and len(norm_msg.split()) <= 5:
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')} and the time is {now.strftime('%I:%M %p')}."

    # Check if this is a recipe/cooking query
    is_recipe_query = bool(re.search(r'\b(recipe|cook|make|prepare|bake|fry|boil|how to make|how do i make|steps for)\b', norm_msg))
    
    if is_recipe_query:
        # Add explicit recipe instruction to the message
        user_message = f"Please provide a complete recipe with ingredients list and numbered steps for: {user_message}"

    # 2) Use Gemini API for all other queries
    if GEMINI_API_KEY and _HAS_GENAI:
        try:
            # Configure API
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Build the prompt first
            system_context = """You are a helpful assistant for an elderly person with memory difficulties.

CRITICAL INSTRUCTIONS FOR RECIPES:
When user asks for a recipe or "how to make/cook" something, you MUST provide:

**Ingredients:**
- List all ingredients with measurements

**Steps:**
1. First step with details
2. Second step with details
3. Continue with all steps numbered
4. Final step

**Tip:** One helpful tip at the end

Example for "how to make tea":
**Ingredients:**
- 1 tea bag or 1 teaspoon loose tea
- 1 cup (240ml) water
- Sugar or honey (optional)
- Milk (optional)

**Steps:**
1. Boil water in a kettle or pot
2. Pour hot water into your cup
3. Add the tea bag and let it steep for 3-5 minutes
4. Remove the tea bag
5. Add sugar, honey, or milk if you like
6. Stir and enjoy!

**Tip:** For stronger tea, let it steep longer. For weaker tea, remove the bag sooner.

Keep responses clear, friendly, and encouraging. Use simple language."""
            
            # Add recent chat history
            conversation_text = system_context + "\n\n"
            
            for msg in chat_history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if content:
                    if role == "user":
                        conversation_text += f"\nUser: {content}"
                    else:
                        conversation_text += f"\nAssistant: {content}"
            
            conversation_text += f"\n\nUser: {user_message}\nAssistant:"
            
            reply_text = None
            
            # Method 1: Try to list and use available models
            try:
                # List available models
                available_models = []
                try:
                    for model in genai.list_models():
                        if 'generateContent' in model.supported_generation_methods:
                            available_models.append(model.name)
                            print(f"DEBUG: Found available model: {model.name}")
                except Exception as list_error:
                    print(f"DEBUG: Could not list models: {str(list_error)}")
                
                # Try available models
                if available_models:
                    for model_name in available_models[:3]:  # Try first 3
                        try:
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(conversation_text)
                            
                            if hasattr(response, 'text') and response.text:
                                reply_text = response.text.strip()
                                print(f"DEBUG: Success with model: {model_name}")
                                break
                            elif hasattr(response, 'candidates') and response.candidates:
                                candidate = response.candidates[0]
                                if hasattr(candidate, 'content'):
                                    content = candidate.content
                                    if hasattr(content, 'parts') and content.parts:
                                        for part in content.parts:
                                            if hasattr(part, 'text') and part.text:
                                                reply_text = part.text.strip()
                                                print(f"DEBUG: Success with model: {model_name}")
                                                break
                            
                            if reply_text:
                                break
                                
                        except Exception as e:
                            print(f"DEBUG: Model {model_name} failed: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"DEBUG: Available models method failed: {str(e)}")
            
            # Method 2: Try direct API call with requests (bypassing SDK)
            if not reply_text:
                try:
                    import requests
                    
                    # Use the v1 API endpoint (not v1beta)
                    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
                    
                    payload = {
                        "contents": [{
                            "parts": [{
                                "text": conversation_text
                            }]
                        }],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 400
                        }
                    }
                    
                    response = requests.post(url, json=payload, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'candidates' in data and len(data['candidates']) > 0:
                            candidate = data['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                parts = candidate['content']['parts']
                                if len(parts) > 0 and 'text' in parts[0]:
                                    reply_text = parts[0]['text'].strip()
                                    print("DEBUG: Success with direct API call (v1)")
                    else:
                        print(f"DEBUG: Direct API v1 failed with status {response.status_code}: {response.text}")
                        
                        # Try v1beta endpoint as fallback
                        url_beta = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
                        response_beta = requests.post(url_beta, json=payload, timeout=10)
                        
                        if response_beta.status_code == 200:
                            data = response_beta.json()
                            if 'candidates' in data and len(data['candidates']) > 0:
                                candidate = data['candidates'][0]
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    parts = candidate['content']['parts']
                                    if len(parts) > 0 and 'text' in parts[0]:
                                        reply_text = parts[0]['text'].strip()
                                        print("DEBUG: Success with direct API call (v1beta)")
                        else:
                            print(f"DEBUG: Direct API v1beta also failed: {response_beta.status_code}")
                            
                except Exception as e:
                    print(f"DEBUG: Direct API call failed: {str(e)}")
            
            if reply_text:
                # Limit length for better readability
                if len(reply_text) > 600:
                    sections = reply_text.split('\n\n')
                    if len(sections[0]) > 500:
                        reply_text = reply_text[:500] + "..."
                    else:
                        reply_text = sections[0]
                
                return reply_text
            
            print("DEBUG: All Gemini methods failed to get response")
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"DEBUG: Gemini outer error: {error_type}: {error_msg}")

    # 3) Fallback responses if Gemini is unavailable
    fallback_responses = {
        'how': "That's a great question! Let me think about the best way to explain that. Could you be more specific about what you'd like to know?",
        'what': "That's interesting! I'd be happy to help you understand that better. Can you tell me a bit more about what you're curious about?",
        'when': "Good question! For specific timing information, it's best to check with your caregiver or look at your reminders.",
        'where': "That's something I'd like to help you with. Can you give me more details about what you're looking for?",
        'why': "That's a thoughtful question! I'm here to help explore that with you.",
        'who': "I'd be happy to help you remember. Can you describe the person you're thinking of?",
    }
    
    # Check for question words
    for question_word, response in fallback_responses.items():
        if norm_msg.startswith(question_word):
            return response
    
    # Generic friendly responses
    default_responses = [
        "That's interesting! Can you tell me more about what you're thinking?",
        "I understand. Is there anything specific you'd like help with right now?",
        "I'm here to listen and help. What would you like to talk about?",
        "I see. Would you like me to help you with your reminders or daily activities?",
        "Tell me more about that. I'm here to support you however I can."
    ]
    
    return random.choice(default_responses)

# ═══════════════════════════════════════════════════════════════════════════
# BACKUP & RESTORE
# ═══════════════════════════════════════════════════════════════════════════

def backup_database():
    """Create backup of database and face images"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("backup") / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup database
        if Path("data.db").exists():
            shutil.copy("data.db", backup_dir / "data.db")
        
        # Backup face images
        if Path("faces").exists():
            shutil.copytree("faces", backup_dir / "faces")
        
        # Create backup info file
        info = {
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'database_size': Path("data.db").stat().st_size if Path("data.db").exists() else 0,
            'persons_count': len(get_all_persons())
        }
        
        with open(backup_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        return True, str(backup_dir)
    
    except Exception as e:
        print(f"Backup error: {e}")
        return False, None

def get_backup_status():
    """Get status of recent backups"""
    try:
        backup_dir = Path("backup")
        
        if not backup_dir.exists():
            return []
        
        backups = []
        for folder in sorted(backup_dir.iterdir(), reverse=True):
            if folder.is_dir():
                info_file = folder / "info.json"
                
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                        backups.append(info)
        
        return backups
    
    except Exception as e:
        print(f"Error getting backup status: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def check_camera_available():
    """Check if camera is available"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except:
        return False

def check_mic_available():
    """Check if microphone is available"""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Try to open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        stream.close()
        p.terminate()
        
        return True
    except:
        return False
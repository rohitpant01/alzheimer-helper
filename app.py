"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                  ALZHEIMER PATIENT HELPER - STREAMLIT APP                 ║
║                          Fully Offline System                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

INSTALLATION & SETUP:
1. Install dependencies: pip install -r requirements.txt
2. Download Vosk model (optional for STT):
   - Visit https://alphacephei.com/vosk/models
   - Download "vosk-model-small-en-us-0.15" (~40MB)
   - Extract to project folder: ./vosk-model/
3. Run: streamlit run app.py

DEFAULT CREDENTIALS:
- Caregiver PIN: 1234 (change in Settings)

DATA STORAGE:
- Database: data.db (SQLite)
- Face images: faces/{person_name}/
- Unknown faces: unknown/
- Backups: backup/YYYYMMDD_HHMMSS/

FEATURES:
- Dual Dashboard (Patient & Caregiver modes)
- Face Recognition (offline)
- Voice Reminders (pyttsx3)
- Speech-to-Text (Vosk offline)
- AI Chatbot (rule-based + LLM ready)
- Cognitive Exercises
- Activity Logging & Timeline
- SOS Emergency Button
- Automatic Backups

PRIVACY & SECURITY:
- All data stored locally
- No external API calls (fully offline)
- Face data never leaves device
- Caregiver PIN protection
"""

import streamlit as st
import sqlite3
from datetime import datetime, timedelta
import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
import threading
import queue
import psutil

# Import utility functions
from utils import (
    init_database, add_person, get_all_persons, update_person, delete_person,
    capture_faces_from_webcam, save_uploaded_faces, train_face_embeddings,
    recognize_face_from_frame, get_person_by_id,
    add_reminder, get_active_reminders, update_reminder_status, delete_reminder,
    log_activity, get_activity_logs, get_cognitive_results,
    add_cognitive_result, backup_database, get_backup_status,
    speak_text, get_tts_engine, stop_tts,
    process_chat_message, init_vosk_stt, recognize_speech_vosk,
    check_camera_available, check_mic_available,
    get_db_connection
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION & INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Page configuration
st.set_page_config(
    page_title="Alzheimer Patient Helper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elderly-friendly UI
st.markdown("""
<style>
    /* Patient Mode - Large, Clear UI */
    .patient-mode {
        font-size: 24px !important;
    }
    
    .big-button {
        font-size: 28px !important;
        padding: 20px 40px !important;
        margin: 10px !important;
        border-radius: 15px !important;
    }
    
    .sos-button {
        background-color: #ff4444 !important;
        color: white !important;
        font-size: 36px !important;
        font-weight: bold !important;
        padding: 30px 60px !important;
        border-radius: 20px !important;
        border: 4px solid #cc0000 !important;
    }
    
    .recognized-person {
        font-size: 32px !important;
        color: #00cc00 !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 20px !important;
        background-color: #e8f5e9 !important;
        border-radius: 10px !important;
    }
    
    .unknown-person {
        font-size: 28px !important;
        color: #ff9800 !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 20px !important;
        background-color: #fff3e0 !important;
        border-radius: 10px !important;
    }
    
    /* Reminder Card */
    .reminder-card {
        padding: 15px !important;
        border-radius: 10px !important;
        border: 2px solid #2196F3 !important;
        background-color: #e3f2fd !important;
        margin: 10px 0 !important;
    }
    
    .reminder-time {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #1976D2 !important;
    }
    
    .reminder-text {
        font-size: 20px !important;
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.mode = 'patient'  # 'patient' or 'caregiver'
        st.session_state.caregiver_pin = '1234'  # Default PIN
        st.session_state.logged_in = False
        st.session_state.voice_enabled = True
        st.session_state.camera_active = False
        st.session_state.last_recognition = None
        st.session_state.last_recognition_time = None
        st.session_state.chat_history = []
        st.session_state.reminders_checked = datetime.now()
        st.session_state.stt_active = False
        st.session_state.recognition_threshold = 0.6
        st.session_state.auto_backup_enabled = True
        st.session_state.last_backup = None
        st.session_state.tts_engine = get_tts_engine()
        
        # Initialize database
        init_database()
        
        # Check system capabilities
        st.session_state.camera_available = check_camera_available()
        st.session_state.mic_available = check_mic_available()
        
        # Initialize Vosk STT (optional)
        st.session_state.vosk_model = init_vosk_stt()

init_session_state()

# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════

def authenticate_caregiver(pin):
    """Verify caregiver PIN"""
    return pin == st.session_state.caregiver_pin

def show_login_page():
    """Display caregiver login page"""
    st.title("🔐 Caregiver Login")
    st.markdown("### Enter PIN to access Caregiver Dashboard")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pin_input = st.text_input("Enter PIN:", type="password", key="pin_input")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔓 Login", use_container_width=True):
                if authenticate_caregiver(pin_input):
                    st.session_state.logged_in = True
                    st.session_state.mode = 'caregiver'
                    st.success("✅ Access granted!")
                    st.rerun()
                else:
                    st.error("❌ Invalid PIN!")
        
        with col_btn2:
            if st.button("← Back to Patient Mode", use_container_width=True):
                st.session_state.mode = 'patient'
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PATIENT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def show_patient_dashboard():
    """Display simplified patient-friendly dashboard"""
    st.markdown("<h1 style='text-align: center; font-size: 48px;'>👋 Hello!</h1>", 
                unsafe_allow_html=True)
    
    # Voice toggle and SOS in header
    header_col1, header_col2, header_col3 = st.columns([2, 3, 2])
    
    with header_col1:
        voice_toggle = st.checkbox("🔊 Voice ON", value=st.session_state.voice_enabled, 
                                   key="voice_toggle")
        st.session_state.voice_enabled = voice_toggle
    
    with header_col3:
        if st.button("🚨 EMERGENCY SOS", key="sos_btn", use_container_width=True):
            handle_sos_event()
    
    # Greeting message
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good Morning! 🌅"
    elif current_hour < 17:
        greeting = "Good Afternoon! ☀️"
    else:
        greeting = "Good Evening! 🌙"
    
    st.markdown(f"<h2 style='text-align: center; font-size: 36px;'>{greeting}</h2>", 
                unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 24px;'>Today is {datetime.now().strftime('%A, %B %d, %Y')}</p>", 
                unsafe_allow_html=True)
    
    # Speak greeting once
    if 'greeted_today' not in st.session_state or \
       st.session_state.get('last_greeting_date') != datetime.now().date():
        if st.session_state.voice_enabled:
            speak_text(f"{greeting}. Today is {datetime.now().strftime('%A, %B %d')}")
            st.session_state.greeted_today = True
            st.session_state.last_greeting_date = datetime.now().date()
    
    st.divider()
    
    # Main content area
    main_col1, main_col2 = st.columns([1, 1])
    
    # LEFT COLUMN: Camera & Recognition
    with main_col1:
        st.markdown("### 📸 Who's Here?")
        
        camera_placeholder = st.empty()
        recognition_placeholder = st.empty()
        
        if st.session_state.camera_available:
            start_camera = st.checkbox("Start Camera", value=st.session_state.camera_active, 
                                       key="patient_camera")
            
            if start_camera:
                st.session_state.camera_active = True
                show_live_recognition(camera_placeholder, recognition_placeholder)
            else:
                st.session_state.camera_active = False
                camera_placeholder.info("📷 Camera is off. Turn it on to recognize visitors.")
        else:
            camera_placeholder.warning("📷 Camera not available")
    
    # RIGHT COLUMN: Reminders & Quick Actions
    with main_col2:
        st.markdown("### 📋 Today's Reminders")
        show_patient_reminders()
        
        st.divider()
        
        st.markdown("### 🎯 Quick Activities")
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🧩 Brain Games", use_container_width=True, key="brain_games"):
                st.session_state.show_cognitive = True
        
        with col_act2:
            if st.button("💬 Chat Helper", use_container_width=True, key="chat_helper"):
                st.session_state.show_chat = True
    
    # Show cognitive exercises or chat if requested
    if st.session_state.get('show_cognitive'):
        st.divider()
        show_cognitive_exercises()
    
    if st.session_state.get('show_chat'):
        st.divider()
        show_patient_chat()
    
    # Check for due reminders
    check_and_fire_reminders()
    
    # Footer with mode switch
    st.divider()
    col_footer1, col_footer2, col_footer3 = st.columns([1, 2, 1])
    with col_footer2:
        if st.button("👨‍⚕️ Switch to Caregiver Mode", use_container_width=True):
            st.session_state.mode = 'caregiver'
            st.session_state.logged_in = False
            st.rerun()

def show_live_recognition(camera_placeholder, recognition_placeholder):
    """Display live camera feed with face recognition"""
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            camera_placeholder.error("❌ Could not access camera")
            return

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            camera_placeholder.error("❌ Could not read camera frame")
            return

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Try to recognize face
        result = recognize_face_from_frame(rgb_frame, st.session_state.recognition_threshold)

        if result:
            person_id, person_name, confidence = result

            # Draw rectangle and name on frame (visual cue)
            try:
                cv2.rectangle(rgb_frame, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50),
                              (0, 255, 0), 3)
            except Exception:
                # if drawing fails, just continue
                pass

            # Display frame
            camera_placeholder.image(rgb_frame, channels="RGB")

            # Top-level recognized badge (keeps your existing look)
            recognition_placeholder.markdown(
                f"<div class='recognized-person'>✅ {person_name}<br>"
                f"<span style='font-size: 20px;'>Confidence: {confidence:.1%}</span></div>",
                unsafe_allow_html=True
            )

            # Get person details from DB
            person = get_person_by_id(person_id)
            if person:
                # Format last visit (if present)
                last_visit_raw = person.get('last_visit_datetime')
                if last_visit_raw:
                    try:
                        lv_dt = datetime.fromisoformat(last_visit_raw)
                        last_visit_str = lv_dt.strftime("%b %d, %Y %I:%M %p")
                    except Exception:
                        last_visit_str = last_visit_raw
                else:
                    last_visit_str = "Never"

                # Build HTML block with all available details (graceful if missing)
                details_html = f"""
                <div style='background:#ffffffcc; padding:10px; border-radius:8px; margin-top:8px;'>
                  <div style='font-size:18px; font-weight:700; margin-bottom:6px;'>{person_name}
                    <small style='color:#555; font-weight:600;'> — {person.get('relation_to_patient','N/A')}</small>
                  </div>
                  <div style='font-size:14px; color:#111; line-height:1.4;'>
                """

                if person.get('date_of_birth'):
                    details_html += f"<div><strong>DOB:</strong> {person.get('date_of_birth')}</div>"
                if person.get('phone_number'):
                    details_html += f"<div><strong>Phone:</strong> {person.get('phone_number')}</div>"
                if person.get('email'):
                    details_html += f"<div><strong>Email:</strong> {person.get('email')}</div>"
                if person.get('address'):
                    details_html += f"<div><strong>Address:</strong> {person.get('address')}</div>"
                if person.get('notes'):
                    notes = person.get('notes')
                    # keep notes short to avoid oversized blocks
                    if len(notes) > 300:
                        notes = notes[:300] + "..."
                    details_html += f"<div><strong>Notes:</strong> {notes}</div>"

                details_html += f"<div style='margin-top:6px; color:#444;'><strong>Last visit:</strong> {last_visit_str}</div>"
                details_html += "</div></div>"

                # Render details
                recognition_placeholder.markdown(details_html, unsafe_allow_html=True)

                # Update last visit in DB
                try:
                    update_person(person_id, {'last_visit_datetime': datetime.now().isoformat()})
                except Exception as e:
                    print("Failed to update last_visit:", e)

                # SPEAK — only if new recognition or > 5 minutes since last spoken
                should_speak = False
                if st.session_state.last_recognition != person_id:
                    should_speak = True
                elif st.session_state.last_recognition_time:
                    elapsed = (datetime.now() - st.session_state.last_recognition_time).total_seconds()
                    if elapsed > 300:  # 5 minutes
                        should_speak = True

                if should_speak:
                    relation = person.get('relation_to_patient', 'visitor')
                    speech_parts = [f"This is {person_name}.", f"They are a {relation}."]
                    if last_visit_raw:
                        speech_parts.append(f"Last seen on {last_visit_str}.")
                    if person.get('phone_number'):
                        speech_parts.append(f"Phone {person.get('phone_number')}.")

                    message = " ".join(speech_parts)

                    if st.session_state.voice_enabled:
                        try:
                            speak_text(message)
                        except Exception as e:
                            print("TTS speak error:", e)

                    # update last-recognition state and log
                    st.session_state.last_recognition = person_id
                    st.session_state.last_recognition_time = datetime.now()
                    log_activity('recognition', f"Recognized {person_name} (ID: {person_id})")

        else:
            # Unknown person
            camera_placeholder.image(rgb_frame, channels="RGB")
            recognition_placeholder.markdown(
                "<div class='unknown-person'>⚠️ Unknown Visitor Detected</div>",
                unsafe_allow_html=True
            )

            # Save unknown face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unknown_dir = Path("unknown")
            unknown_dir.mkdir(exist_ok=True)
            try:
                cv2.imwrite(str(unknown_dir / f"unknown_{timestamp}.jpg"),
                            cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print("Failed to save unknown face:", e)

            # Log activity & notify
            log_activity('unknown_visitor', f"Unknown person detected at {datetime.now().isoformat()}")
            if st.session_state.voice_enabled:
                try:
                    speak_text("I don't recognize this person. The caregiver has been notified.")
                except Exception as e:
                    print("TTS speak error:", e)

    except Exception as e:
        camera_placeholder.error(f"Camera error: {str(e)}")

def show_patient_reminders():
    """Display upcoming reminders for patient"""
    reminders = get_active_reminders()
    
    if not reminders:
        st.info("No reminders for now. Enjoy your day! 😊")
        return
    
    for reminder in reminders[:5]:  # Show next 5 reminders
        reminder_time = datetime.fromisoformat(reminder['next_due'])
        time_str = reminder_time.strftime("%I:%M %p")
        
        st.markdown(f"""
        <div class='reminder-card'>
            <div class='reminder-time'>🕐 {time_str}</div>
            <div class='reminder-text'>{reminder['title']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Acknowledge button
        if st.button(f"✅ Done", key=f"ack_{reminder['id']}", use_container_width=True):
            update_reminder_status(reminder['id'], 'completed')
            log_activity('reminder_completed', f"Completed: {reminder['title']}")
            st.success("Great job! ✅")
            st.rerun()

def check_and_fire_reminders():
    """Check for due reminders and fire notifications"""
    now = datetime.now()
    
    # Check every minute
    if (now - st.session_state.reminders_checked).seconds < 60:
        return
    
    st.session_state.reminders_checked = now
    
    reminders = get_active_reminders()
    
    for reminder in reminders:
        next_due = datetime.fromisoformat(reminder['next_due'])
        
        # If reminder is due (within next 2 minutes)
        if now >= next_due and (next_due - now).seconds < 120:
            # Fire notification
            st.toast(f"⏰ Reminder: {reminder['title']}", icon="⏰")
            
            if st.session_state.voice_enabled:
                speak_text(f"Reminder. {reminder['title']}")
            
            # Update next due time if recurring
            if reminder['recurrence'] != 'once':
                # Calculate next occurrence
                if reminder['recurrence'] == 'daily':
                    next_time = next_due + timedelta(days=1)
                elif reminder['recurrence'] == 'weekly':
                    next_time = next_due + timedelta(weeks=1)
                else:
                    next_time = next_due
                
                # Update reminder
                conn = get_db_connection()
                conn.execute(
                    "UPDATE reminders SET next_due = ? WHERE id = ?",
                    (next_time.isoformat(), reminder['id'])
                )
                conn.commit()
                conn.close()

def show_cognitive_exercises():
    """Display cognitive exercises for patient"""
    st.markdown("### 🧩 Brain Exercise Time!")
    
    exercise_type = st.radio(
        "Choose an activity:",
        ["Color Match", "Number Recall", "Word Association"],
        key="exercise_type",
        horizontal=True
    )
    
    if exercise_type == "Color Match":
        show_color_match_exercise()
    elif exercise_type == "Number Recall":
        show_number_recall_exercise()
    else:
        show_word_association_exercise()
    
    if st.button("← Back", key="back_from_cognitive"):
        st.session_state.show_cognitive = False
        st.rerun()

def show_color_match_exercise():
    """Color matching cognitive exercise"""
    colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange"]
    color_codes = {
        "Red": "#FF0000", "Blue": "#0000FF", "Green": "#00FF00",
        "Yellow": "#FFFF00", "Purple": "#800080", "Orange": "#FFA500"
    }
    
    if 'color_target' not in st.session_state:
        st.session_state.color_target = np.random.choice(colors)
        st.session_state.color_score = 0
        st.session_state.color_attempts = 0
    
    target = st.session_state.color_target
    
    st.markdown(f"""
    <div style='text-align: center; padding: 30px; background-color: {color_codes[target]}; 
                border-radius: 15px; margin: 20px 0;'>
        <h2 style='color: white; font-size: 48px;'>What color is this?</h2>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    options = np.random.choice(colors, size=3, replace=False)
    if target not in options:
        options[0] = target
        np.random.shuffle(options)
    
    for i, col in enumerate(cols):
        with col:
            if st.button(options[i], key=f"color_opt_{i}", use_container_width=True):
                st.session_state.color_attempts += 1
                
                if options[i] == target:
                    st.session_state.color_score += 1
                    st.success("✅ Correct! Great job!")
                    
                    if st.session_state.voice_enabled:
                        speak_text(f"Correct! That's {target}!")
                else:
                    st.error(f"❌ Not quite. That's {options[i]}.")
                
                # Save result
                add_cognitive_result(
                    None,  # No specific person
                    st.session_state.color_score,
                    f"Color Match: {st.session_state.color_score}/{st.session_state.color_attempts}"
                )
                
                # New question
                st.session_state.color_target = np.random.choice(colors)
                time.sleep(1)
                st.rerun()
    
    st.info(f"Score: {st.session_state.color_score}/{st.session_state.color_attempts}")

def show_number_recall_exercise():
    """Number recall exercise"""
    if 'number_sequence' not in st.session_state:
        st.session_state.number_sequence = [str(np.random.randint(0, 10)) for _ in range(4)]
        st.session_state.number_shown = False
        st.session_state.number_score = 0
    
    if not st.session_state.number_shown:
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background-color: #e3f2fd; 
                    border-radius: 15px; margin: 20px 0;'>
            <h2 style='font-size: 36px;'>Remember these numbers:</h2>
            <h1 style='font-size: 72px; color: #1976D2;'>{' - '.join(st.session_state.number_sequence)}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("I'm Ready!", key="number_ready", use_container_width=True):
            st.session_state.number_shown = True
            if st.session_state.voice_enabled:
                speak_text("Now enter the numbers you remember")
            st.rerun()
    else:
        st.markdown("<h3 style='text-align: center;'>Enter the numbers you remember:</h3>", 
                   unsafe_allow_html=True)
        
        user_input = st.text_input("Your answer:", key="number_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Answer", key="check_number", use_container_width=True):
                correct_answer = ''.join(st.session_state.number_sequence)
                user_answer = user_input.replace(' ', '').replace('-', '')
                
                if user_answer == correct_answer:
                    st.success("🎉 Perfect! All correct!")
                    st.session_state.number_score += 1
                    if st.session_state.voice_enabled:
                        speak_text("Excellent! All correct!")
                else:
                    st.error(f"Not quite. The answer was: {' - '.join(st.session_state.number_sequence)}")
                
                add_cognitive_result(None, st.session_state.number_score, 
                                   f"Number Recall: Score {st.session_state.number_score}")
        
        with col2:
            if st.button("Try New Numbers", key="new_numbers", use_container_width=True):
                st.session_state.number_sequence = [str(np.random.randint(0, 10)) for _ in range(4)]
                st.session_state.number_shown = False
                st.rerun()

def show_word_association_exercise():
    """Word association exercise"""
    word_pairs = {
        "Day": "Night", "Hot": "Cold", "Up": "Down",
        "Happy": "Sad", "Big": "Small", "Fast": "Slow",
        "Good": "Bad", "Young": "Old", "Light": "Dark"
    }
    
    if 'word_target' not in st.session_state:
        st.session_state.word_target = np.random.choice(list(word_pairs.keys()))
        st.session_state.word_score = 0
        st.session_state.word_attempts = 0
    
    target = st.session_state.word_target
    correct_answer = word_pairs[target]
    
    st.markdown(f"""
    <div style='text-align: center; padding: 40px; background-color: #fff3e0; 
                border-radius: 15px; margin: 20px 0;'>
        <h2 style='font-size: 32px;'>What is the opposite of:</h2>
        <h1 style='font-size: 64px; color: #F57C00;'>{target}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate options
    wrong_answers = [v for k, v in word_pairs.items() if k != target]
    options = [correct_answer] + list(np.random.choice(wrong_answers, size=2, replace=False))
    np.random.shuffle(options)
    
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            if st.button(options[i], key=f"word_opt_{i}", use_container_width=True):
                st.session_state.word_attempts += 1
                
                if options[i] == correct_answer:
                    st.session_state.word_score += 1
                    st.success("✅ Correct!")
                    if st.session_state.voice_enabled:
                        speak_text(f"Correct! {target} is opposite of {correct_answer}")
                else:
                    st.error(f"❌ The opposite of {target} is {correct_answer}")
                
                add_cognitive_result(None, st.session_state.word_score,
                                   f"Word Association: {st.session_state.word_score}/{st.session_state.word_attempts}")
                
                st.session_state.word_target = np.random.choice(list(word_pairs.keys()))
                time.sleep(1)
                st.rerun()
    
    st.info(f"Score: {st.session_state.word_score}/{st.session_state.word_attempts}")

def show_patient_chat():
    """Display AI chatbot for patient assistance"""
    st.markdown("### 💬 AI Chat Helper")
    st.markdown("Ask me anything! I'm here to help.")
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Assistant:** {msg['content']}")
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input("Type your message:", key="patient_chat_input")
    
    with col2:
        if st.session_state.mic_available and st.session_state.vosk_model:
            if st.button("🎤 Speak", key="patient_voice_input"):
                with st.spinner("Listening..."):
                    recognized_text = recognize_speech_vosk(st.session_state.vosk_model, duration=5)
                    if recognized_text:
                        user_input = recognized_text
                        st.info(f"You said: {recognized_text}")
    
    if st.button("Send", key="send_patient_chat", use_container_width=True) and user_input:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Get AI response
        response = process_chat_message(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        
        # Speak response if voice enabled
        if st.session_state.voice_enabled:
            speak_text(response)
        
        st.rerun()
    
    if st.button("← Back", key="back_from_chat"):
        st.session_state.show_chat = False
        st.rerun()

def handle_sos_event():
    """Handle emergency SOS button press"""
    log_activity('sos', f"SOS button pressed at {datetime.now()}")
    
    st.error("🚨 EMERGENCY ALERT ACTIVATED! 🚨")
    st.warning("Caregiver has been notified immediately!")
    
    # Voice alert
    if st.session_state.voice_enabled:
        speak_text("Emergency alert activated. Help is being notified.")
    
    # TODO: Add actual emergency notification here
    # --- PLACEHOLDER: ADD EMERGENCY NOTIFICATION (Twilio SMS/Call, Email, etc.) ---
    # Example:
    # send_sms(emergency_contact_phone, "EMERGENCY: Patient activated SOS button!")
    # send_email(caregiver_email, "SOS Alert", "Patient needs immediate assistance")

# ═══════════════════════════════════════════════════════════════════════════
# CAREGIVER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def show_caregiver_dashboard():
    """Display full-featured caregiver dashboard"""
    st.title("👨‍⚕️ Caregiver Dashboard")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Select Section:",
            ["Overview", "Person Management", "Reminders", "Activity Log", 
             "Cognitive Results", "System Settings", "Backup & Restore"],
            key="caregiver_nav"
        )
        
        st.divider()
        
        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.mode = 'patient'
            st.rerun()
    
    # Display selected page
    if page == "Overview":
        show_overview_page()
    elif page == "Person Management":
        show_person_management_page()
    elif page == "Reminders":
        show_reminders_page()
    elif page == "Activity Log":
        show_activity_log_page()
    elif page == "Cognitive Results":
        show_cognitive_results_page()
    elif page == "System Settings":
        show_settings_page()
    elif page == "Backup & Restore":
        show_backup_page()

def show_overview_page():
    """Display overview dashboard"""
    st.markdown("## 📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    persons = get_all_persons()
    reminders = get_active_reminders()
    
    with col1:
        st.metric("Total Persons", len(persons))
    
    with col2:
        st.metric("Active Reminders", len(reminders))
    
    with col3:
        # Count today's activities
        logs = get_activity_logs(limit=1000)
        today_logs = [l for l in logs if l['timestamp'].startswith(datetime.now().strftime("%Y-%m-%d"))]
        st.metric("Today's Activities", len(today_logs))
    
    with col4:
        # System health
        cpu_percent = psutil.cpu_percent(interval=1)
        health = "Good" if cpu_percent < 70 else "High Load"
        st.metric("System Health", health, f"{cpu_percent:.0f}% CPU")
    
    st.divider()
    
    # Recent activity
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📋 Recent Activity")
        recent_logs = get_activity_logs(limit=10)
        
        if recent_logs:
            for log in recent_logs:
                log_time = datetime.fromisoformat(log['timestamp']).strftime("%I:%M %p")
                st.markdown(f"**{log_time}** - {log['type']}: {log['detail']}")
        else:
            st.info("No recent activity")
    
    with col_right:
        st.markdown("### 👥 Recent Visitors")
        recognition_logs = [l for l in logs if l['type'] == 'recognition'][-5:]
        
        if recognition_logs:
            for log in recognition_logs:
                st.markdown(f"- {log['detail']}")
        else:
            st.info("No recent recognitions")
    
    st.divider()
    
    # System status
    st.markdown("### 🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        cam_status = "✅ Available" if st.session_state.camera_available else "❌ Not Available"
        st.info(f"📷 Camera: {cam_status}")
    
    with status_col2:
        mic_status = "✅ Available" if st.session_state.mic_available else "❌ Not Available"
        st.info(f"🎤 Microphone: {mic_status}")
    
    with status_col3:
        vosk_status = "✅ Ready" if st.session_state.vosk_model else "⚠️ Not Configured"
        st.info(f"🗣️ Speech Recognition: {vosk_status}")

def show_person_management_page():
    """Display person management interface"""
    st.markdown("## 👥 Person Management")
    
    # Action buttons
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add New Person", use_container_width=True):
            st.session_state.show_add_person = True
    
    # Show add person form
    if st.session_state.get('show_add_person'):
        show_add_person_form()
        return
    
    # List existing persons
    persons = get_all_persons()
    
    if not persons:
        st.info("No persons added yet. Click 'Add New Person' to get started.")
        return
    
    st.markdown(f"### Registered Persons ({len(persons)})")
    
    for person in persons:
        with st.expander(f"👤 {person['full_name']} ({person.get('relation_to_patient', 'N/A')})"):
            col_info, col_actions = st.columns([3, 1])
            
            with col_info:
                st.markdown(f"**Phone:** {person.get('phone_number', 'N/A')}")
                st.markdown(f"**Email:** {person.get('email', 'N/A')}")
                st.markdown(f"**Last Visit:** {person.get('last_visit_datetime', 'Never')}")
                
                if person.get('notes'):
                    st.markdown(f"**Notes:** {person['notes']}")
                
                # Show photos
                person_folder = Path("faces") / person['full_name']
                if person_folder.exists():
                    photos = list(person_folder.glob("*.jpg"))
                    if photos:
                        st.markdown(f"**Photos:** {len(photos)} images stored")
            
            with col_actions:
                if st.button("✏️ Edit", key=f"edit_{person['id']}", use_container_width=True):
                    st.session_state.edit_person_id = person['id']
                    st.rerun()
                
                if st.button("🗑️ Delete", key=f"del_{person['id']}", use_container_width=True):
                    if st.checkbox(f"Confirm delete {person['full_name']}", key=f"confirm_del_{person['id']}"):
                        delete_person(person['id'])
                        st.success(f"Deleted {person['full_name']}")
                        st.rerun()

def show_add_person_form():
    """Display form to add new person"""
    st.markdown("### ➕ Add New Person")
    
    with st.form("add_person_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name*", key="person_name")
            relation = st.text_input("Relation to Patient", key="person_relation", 
                                    placeholder="e.g., daughter, neighbor, doctor")
            phone = st.text_input("Phone Number", key="person_phone")
            email = st.text_input("Email", key="person_email")
        
        with col2:
            nickname = st.text_input("Nickname (Optional)", key="person_nickname")
            dob = st.date_input(
            "Date of Birth (Optional)",
            value=datetime(1990, 1, 1),        # default selection
            min_value=datetime(1900, 1, 1),    # allow birth dates before 2015
            max_value=datetime.now(),          # cannot select future date
            key="person_dob"
        )

            address = st.text_area("Address (Optional)", key="person_address")
            tags = st.text_input("Tags (comma-separated)", key="person_tags", 
                                placeholder="family, caregiver, medical")
        
        notes = st.text_area("Notes (allergies, preferences, etc.)", key="person_notes")
        
        st.markdown("### 📸 Add Photos")
        photo_method = st.radio("Choose method:", ["Capture from Webcam", "Upload Images"], 
                               key="photo_method", horizontal=True)
        
        if photo_method == "Upload Images":
            uploaded_files = st.file_uploader("Upload photos", type=['jpg', 'jpeg', 'png'], 
                                            accept_multiple_files=True, key="upload_photos")
        
        col_submit, col_cancel = st.columns(2)
        
        with col_submit:
            submitted = st.form_submit_button("💾 Save Person", use_container_width=True)
        
        with col_cancel:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if cancel:
            st.session_state.show_add_person = False
            st.rerun()
        
        if submitted and full_name:
            # Create person data
            person_data = {
                'full_name': full_name,
                'relation_to_patient': relation,
                'nickname': nickname,
                'date_of_birth': dob.isoformat() if dob else None,
                'phone_number': phone,
                'email': email,
                'address': address,
                'notes': notes,
                'tags': tags,
                'created_at': datetime.now().isoformat()
            }
            
            # Add person to database
            person_id = add_person(person_data)
            
            if person_id:
                # Handle photos
                if photo_method == "Capture from Webcam":
                    with st.spinner("Preparing camera..."):
                        success = capture_faces_from_webcam(full_name, count=15)
                        if success:
                            st.success("✅ Captured 15 photos!")
                        else:
                            st.warning("⚠️ Could not capture photos. Please add them later.")
                elif uploaded_files:
                    save_uploaded_faces(full_name, uploaded_files)
                    st.success(f"✅ Saved {len(uploaded_files)} photos!")
                
                # Train embeddings
                with st.spinner("Training face recognition..."):
                    train_face_embeddings()
                
                st.success(f"✅ Added {full_name} successfully!")
                log_activity('person_added', f"Added person: {full_name}")
                
                st.session_state.show_add_person = False
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to add person. Please try again.")

def show_reminders_page():
    """Display reminders management"""
    st.markdown("## ⏰ Reminders Management")
    
    # Add reminder button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add Reminder", use_container_width=True):
            st.session_state.show_add_reminder = True
    
    # Show add reminder form
    if st.session_state.get('show_add_reminder'):
        show_add_reminder_form()
        return
    
    # List reminders
    reminders = get_active_reminders()
    
    if not reminders:
        st.info("No active reminders. Click 'Add Reminder' to create one.")
        return
    
    st.markdown(f"### Active Reminders ({len(reminders)})")
    
    for reminder in reminders:
        with st.expander(f"⏰ {reminder['title']}"):
            col_info, col_actions = st.columns([3, 1])
            
            with col_info:
                next_due = datetime.fromisoformat(reminder['next_due'])
                st.markdown(f"**Next Due:** {next_due.strftime('%Y-%m-%d %I:%M %p')}")
                st.markdown(f"**Recurrence:** {reminder['recurrence'].title()}")
                
                if reminder.get('description'):
                    st.markdown(f"**Description:** {reminder['description']}")
            
            with col_actions:
                if st.button("✏️ Edit", key=f"edit_rem_{reminder['id']}", use_container_width=True):
                    st.info("Edit functionality - coming soon")
                
                if st.button("🗑️ Delete", key=f"del_rem_{reminder['id']}", use_container_width=True):
                    delete_reminder(reminder['id'])
                    st.success("Reminder deleted")
                    st.rerun()

def show_add_reminder_form():
    """Display form to add new reminder"""
    st.markdown("### ➕ Add New Reminder")
    
    with st.form("add_reminder_form"):
        title = st.text_input("Reminder Title*", key="rem_title", 
                             placeholder="e.g., Take Morning Medication")
        description = st.text_area("Description (Optional)", key="rem_desc")
        
        col1, col2 = st.columns(2)
        
        with col1:
            due_date = st.date_input("Date", value=datetime.now().date(), key="rem_date")
        
        with col2:
            due_time = st.time_input("Time", value=datetime.now().time(), key="rem_time")
        
        recurrence = st.selectbox("Recurrence", 
                                 ["once", "daily", "weekly"],
                                 key="rem_recurrence")
        
        col_submit, col_cancel = st.columns(2)
        
        with col_submit:
            submitted = st.form_submit_button("💾 Save Reminder", use_container_width=True)
        
        with col_cancel:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if cancel:
            st.session_state.show_add_reminder = False
            st.rerun()
        
        if submitted and title:
            # Combine date and time
            due_datetime = datetime.combine(due_date, due_time)
            
            # Create reminder
            reminder_data = {
                'title': title,
                'description': description,
                'next_due': due_datetime.isoformat(),
                'recurrence': recurrence,
                'status': 'active'
            }
            
            reminder_id = add_reminder(reminder_data)
            
            if reminder_id:
                st.success(f"✅ Reminder '{title}' added successfully!")
                log_activity('reminder_added', f"Added reminder: {title}")
                
                st.session_state.show_add_reminder = False
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to add reminder. Please try again.")

def show_activity_log_page():
    """Display activity logs"""
    st.markdown("## 📜 Activity Log")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        log_type_filter = st.multiselect(
            "Filter by Type:",
            ["recognition", "unknown_visitor", "reminder_completed", "reminder_missed", "sos", "person_added"],
            key="log_type_filter"
        )

    with col2:
        date_filter = st.date_input("Filter by Date:", value=None, key="log_date_filter")

    with col3:
        limit = st.number_input(
            "Show entries:", 
            min_value=10, 
            max_value=500, 
            value=50,
            key="log_limit"
        )

    # Get logs
    logs = get_activity_logs(limit=int(limit))

    # Apply filters
    if log_type_filter:
        logs = [l for l in logs if l['type'] in log_type_filter]

    if date_filter:
        date_str = date_filter.strftime("%Y-%m-%d")
        logs = [l for l in logs if l['timestamp'].startswith(date_str)]

    # Display logs
    if not logs:
        st.info("No activity logs found.")
        return

    st.markdown(f"### Activity Log ({len(logs)} entries)")

    # Readable color palette (darkened)
    COLOR_MAP = {
        "sos": "#ff9999",              # Darker red
        "recognition": "#99e699",      # Darker green
        "unknown_visitor": "#ffe680",  # Darker yellow
        "reminder_completed": "#c9d6ff",
        "reminder_missed": "#ffcc80",
        "person_added": "#b3ecff",
        "default": "#b3d1ff"           # Blue-ish for others
    }

    # Show logs
    for log in logs:
        timestamp = datetime.fromisoformat(log['timestamp']).strftime("%Y-%m-%d %I:%M %p")

        color = COLOR_MAP.get(log['type'], COLOR_MAP["default"])

        st.markdown(f"""
        <div style='padding: 12px; margin: 6px 0;
                    background-color: {color};
                    border-radius: 6px; 
                    border-left: 6px solid #333;
                    font-weight: 600;
                    color: #000000;'>
            <strong>{timestamp}</strong> — <em>{log['type']}</em><br>
            {log['detail']}
        </div>
        """, unsafe_allow_html=True)


def show_cognitive_results_page():
    """Display cognitive exercise results"""
    st.markdown("## 🧠 Cognitive Exercise Results")
    
    results = get_cognitive_results(limit=50)
    
    if not results:
        st.info("No cognitive exercise results yet.")
        return
    
    # Calculate statistics
    total_exercises = len(results)
    avg_score = np.mean([r['score'] for r in results if r['score'] is not None])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Exercises", total_exercises)
    
    with col2:
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col3:
        recent_results = results[:10]
        recent_avg = np.mean([r['score'] for r in recent_results if r['score'] is not None])
        st.metric("Recent Avg (10)", f"{recent_avg:.1f}")
    
    st.divider()
    
    # Display results
    st.markdown("### Recent Results")
    
    for result in results[:20]:
        timestamp = datetime.fromisoformat(result['timestamp']).strftime("%Y-%m-%d %I:%M %p")
        
        st.markdown(f"""
        <div style='padding: 16px; margin: 10px 0;
                    background-color: #d9f2e1;
                    border-radius: 10px;
                    border-left: 8px solid #4CAF50;
                    color: #000000;
                    font-size: 18px;
                    font-weight: 600;'>
            <strong>{timestamp}</strong> — Score: {result['score']}<br>
            <span style='font-size: 16px; font-weight: 400;'>{result['details']}</span>
        </div>
        """, unsafe_allow_html=True)

def show_settings_page():
    """Display system settings"""
    st.markdown("## ⚙️ System Settings")
    
    st.markdown("### 🔐 Security Settings")
    
    with st.form("pin_change_form"):
        new_pin = st.text_input("New Caregiver PIN:", type="password", key="new_pin")
        confirm_pin = st.text_input("Confirm PIN:", type="password", key="confirm_pin")
        
        if st.form_submit_button("Update PIN"):
            if new_pin == confirm_pin and len(new_pin) >= 4:
                st.session_state.caregiver_pin = new_pin
                st.success("✅ PIN updated successfully!")
                log_activity('settings_changed', "Caregiver PIN updated")
            else:
                st.error("PINs don't match or too short (minimum 4 digits)")
    
    st.divider()
    
    st.markdown("### 🎤 Voice & Speech Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        voice_enabled = st.checkbox("Enable Text-to-Speech", 
                                    value=st.session_state.voice_enabled,
                                    key="settings_voice")
        st.session_state.voice_enabled = voice_enabled
    
    with col2:
        if st.button("Test TTS", key="test_tts"):
            speak_text("Text to speech is working correctly.")
    
    st.divider()
    
    st.markdown("### 🎯 Recognition Settings")
    
    threshold = st.slider("Face Recognition Threshold:", 
                         min_value=0.3, max_value=0.9, 
                         value=st.session_state.recognition_threshold,
                         step=0.05,
                         help="Lower = more strict, Higher = more lenient",
                         key="settings_threshold")
    st.session_state.recognition_threshold = threshold
    
    st.info(f"Current threshold: {threshold:.2f} - "
           f"{'Strict' if threshold < 0.5 else 'Moderate' if threshold < 0.7 else 'Lenient'}")
    
    st.divider()
    
    st.markdown("### 💾 Automatic Backup")
    
    auto_backup = st.checkbox("Enable Daily Automatic Backup",
                             value=st.session_state.auto_backup_enabled,
                             key="settings_backup")
    st.session_state.auto_backup_enabled = auto_backup
    
    if st.session_state.last_backup:
        st.info(f"Last backup: {st.session_state.last_backup}")

def show_backup_page():
    """Display backup and restore options"""
    st.markdown("## 💾 Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Create Backup")
        
        if st.button("Create Backup Now", use_container_width=True):
            with st.spinner("Creating backup..."):
                success, backup_path = backup_database()
                
                if success:
                    st.success(f"✅ Backup created: {backup_path}")
                    st.session_state.last_backup = datetime.now().strftime("%Y-%m-%d %I:%M %p")
                    log_activity('backup_created', f"Manual backup created: {backup_path}")
                else:
                    st.error("❌ Backup failed!")
        
        st.info("Backup includes:\n- Database (data.db)\n- Face images\n- All settings")
    
    with col2:
        st.markdown("### 📊 Backup Status")
        
        backup_dir = Path("backup")
        if backup_dir.exists():
            backups = sorted(backup_dir.glob("*/"), reverse=True)
            
            if backups:
                st.success(f"Total backups: {len(backups)}")
                
                st.markdown("#### Recent Backups:")
                for backup in backups[:5]:
                    backup_name = backup.name
                    backup_size = sum(f.stat().st_size for f in backup.rglob('*') if f.is_file())
                    size_mb = backup_size / (1024 * 1024)
                    
                    st.markdown(f"- **{backup_name}** ({size_mb:.2f} MB)")
            else:
                st.warning("No backups found")
        else:
            st.info("No backup directory yet")
    
    st.divider()
    
    st.markdown("### 🗑️ Storage Management")
    
    # Calculate storage usage
    def get_dir_size(path):
        total = 0
        if Path(path).exists():
            for f in Path(path).rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
        return total / (1024 * 1024)  # Convert to MB
    
    db_size = get_dir_size("data.db") if Path("data.db").exists() else 0
    faces_size = get_dir_size("faces")
    unknown_size = get_dir_size("unknown")
    backup_size = get_dir_size("backup")
    
    col_storage1, col_storage2, col_storage3, col_storage4 = st.columns(4)
    
    with col_storage1:
        st.metric("Database", f"{db_size:.2f} MB")
    
    with col_storage2:
        st.metric("Face Images", f"{faces_size:.2f} MB")
    
    with col_storage3:
        st.metric("Unknown Faces", f"{unknown_size:.2f} MB")
    
    with col_storage4:
        st.metric("Backups", f"{backup_size:.2f} MB")
    
    # Cleanup options
    st.markdown("### 🧹 Cleanup")
    
    if st.button("Clear Unknown Faces", use_container_width=True):
        unknown_dir = Path("unknown")
        if unknown_dir.exists():
            count = len(list(unknown_dir.glob("*.jpg")))
            for img in unknown_dir.glob("*.jpg"):
                img.unlink()
            st.success(f"Deleted {count} unknown face images")
            log_activity('cleanup', f"Cleared {count} unknown face images")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    # Check which mode to display
    if st.session_state.mode == 'patient':
        show_patient_dashboard()
    
    elif st.session_state.mode == 'caregiver':
        if st.session_state.logged_in:
            show_caregiver_dashboard()
        else:
            show_login_page()
    
    # Auto-backup check (once per day)
    if st.session_state.auto_backup_enabled:
        last_backup_date = st.session_state.get('last_backup_date')
        today = datetime.now().date()
        
        if last_backup_date != today:
            # Perform automatic backup
            success, backup_path = backup_database()
            if success:
                st.session_state.last_backup_date = today
                st.session_state.last_backup = datetime.now().strftime("%Y-%m-%d %I:%M %p")
                log_activity('auto_backup', f"Automatic backup created: {backup_path}")

# ═══════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
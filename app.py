import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
from scipy.signal import find_peaks
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import json
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def cleanup_resources():
    """Cleanup function to be called after processing"""
    gc.collect()  # Force garbage collection
    
    # Clear temporary files
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith('streamlit-'):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except:
                pass
    plt.close('all')  # Close all matplotlib figures

def load_all_results():
    """Load all analysis results from JSON files"""
    results = []
    results_dir = "analysis_results"
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    results.append(json.load(f))
    return results

def create_analysis_dataframe():
    """Convert results to pandas DataFrame"""
    results = load_all_results()
    if results:
        df = pd.DataFrame(results)
        df['session_date'] = pd.to_datetime(df['session_date'])
        return df
    return None

# Add tabs to separate video analysis and data management
tab1, tab2 = st.tabs(["Video Analysis", "Data Management"])

with tab1:
    # Video Analysis Tab
    st.title("Jump Analysis with Visual Overlays and Jump Classification")

    # Session Information Form
    with st.form(key='session_info'):
        col1, col2 = st.columns(2)
        with col1:
            athlete_name = st.text_input("Athlete Name")
            workout_number = st.number_input("Workout Number", min_value=1, value=1)
        with col2:
            session_date = st.date_input("Date")
            set_number = st.number_input("Set Number", min_value=1, value=1)
        
        submit_button = st.form_submit_button(label='Save Session Info')

    # Display session info in sidebar
    if submit_button or ('session_info_saved' in st.session_state and st.session_state.session_info_saved):
        st.session_state.session_info_saved = True
        st.session_state.athlete_name = athlete_name
        st.session_state.session_date = session_date
        st.session_state.workout_number = workout_number
        st.session_state.set_number = set_number
        
        with st.sidebar:
            st.write("### Current Session")
            st.write(f"Athlete: {athlete_name}")
            st.write(f"Date: {session_date}")
            st.write(f"Workout #: {workout_number}")
            st.write(f"Set #: {set_number}")

    # Only show video upload if session info is saved
    if 'session_info_saved' in st.session_state and st.session_state.session_info_saved:
        st.write("Upload a video to see pose keypoints, connections, and jump analysis.")
        
        # Video Upload Section
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

        if uploaded_file:
            try:
                # Create a dictionary to store session data
                session_data = {
                    'athlete_name': athlete_name,
                    'session_date': session_date.strftime('%Y-%m-%d'),
                    'workout_number': workout_number,
                    'set_number': set_number,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # Save uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_file.read())
                    video_path = temp_file.name

                st.success("Video uploaded successfully!")

                # Load the video
                cap = cv2.VideoCapture(video_path)

                # Prepare to save frames
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
                video_duration_minutes = total_frames / (fps * 60)

                # Temporary directory for frames
                frames_dir = tempfile.mkdtemp()

                # Variables for jump classification and counting
                hip_y_positions = []
                frame_count = 0

                # Process video frame by frame and collect hip positions
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert frame to RGB for MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame with MediaPipe Pose
                    results = pose.process(frame_rgb)

                    # Draw pose landmarks and connections
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        # Track Y-coordinate of the left hip
                        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                        hip_y_positions.append(left_hip.y)

                    # Save the frame
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1

                cap.release()

                # Convert hip_y_positions to numpy array
                hip_y_positions = np.array(hip_y_positions)
                
                # Perform jump detection
                peaks, properties = find_peaks(-hip_y_positions, prominence=0.01)
                
                # Create and display the jump detection graph
                st.write("### Jump Detection Graph")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Set threshold and separate big and small jumps
                big_jump_threshold = 0.07
                displacements = properties['prominences']
                big_jump_indices = peaks[displacements >= big_jump_threshold]
                small_jump_indices = peaks[displacements < big_jump_threshold]
                
                # Plot the hip positions and detected jumps
                ax.plot(-hip_y_positions, label='Hip Y-Position', color='blue', alpha=0.7)
                
                if len(big_jump_indices) > 0:
                    ax.plot(big_jump_indices, -hip_y_positions[big_jump_indices], "rx",
                           label="Big Jumps", markersize=10, markeredgewidth=2)
                
                if len(small_jump_indices) > 0:
                    ax.plot(small_jump_indices, -hip_y_positions[small_jump_indices], "yo",
                           label="Small Jumps (Pogos)", markersize=8, markeredgewidth=2)

                ax.set_title('Hip Y-Position with Detected Jumps')
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Hip Height (inverted Y-Coordinate)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Update session data with analysis results
                session_data.update({
                    'total_jumps': len(peaks),
                    'big_jumps': len(big_jump_indices),
                    'small_jumps': len(small_jump_indices),
                    'video_duration': video_duration_minutes,
                    'jumps_per_minute': (len(peaks) / video_duration_minutes) if video_duration_minutes > 0 else 0
                })

                # Save session data
                results_dir = "analysis_results"
                os.makedirs(results_dir, exist_ok=True)
                filename = f"{session_data['athlete_name']}_{session_data['session_date']}_{session_data['workout_number']}_{session_data['set_number']}.json"
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(session_data, f)

                # Create frame-by-frame counters
                jumps_by_frame = {frame: {'big': 0, 'small': 0, 'total': 0} 
                                for frame in range(frame_count)}

                # Determine which frames contain jumps and their types
                current_big_jumps = 0
                current_small_jumps = 0
                
                for peak_idx, prominence in zip(peaks, displacements):
                    if prominence >= big_jump_threshold:
                        current_big_jumps += 1
                    else:
                        current_small_jumps += 1
                    
                    # Update all frames after this jump with the new counts
                    for frame in range(peak_idx, frame_count):
                        jumps_by_frame[frame] = {
                            'big': current_big_jumps,
                            'small': current_small_jumps,
                            'total': current_big_jumps + current_small_jumps
                        }

                # Process saved frames again to add text overlays with running counts
                for i in range(frame_count):
                    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                    frame = cv2.imread(frame_path)
                    
                    if frame is not None:
                        # Get current counts for this frame
                        current_counts = jumps_by_frame[i]
                        
                        # Calculate reps per minute based on current frame
                        current_time_minutes = (i / fps) / 60
                        reps_per_minute = current_counts["total"] / current_time_minutes if current_time_minutes > 0 else 0
                        
                        def put_text_with_background(img, text, position):
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.5
                            thickness = 3
                            color = (0, 255, 0)
                            
                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                            cv2.rectangle(img, 
                                        (position[0] - 10, position[1] - text_height - 10),
                                        (position[0] + text_width + 10, position[1] + 10),
                                        (0, 0, 0),
                                        -1)
                            cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

                        # Add text with backgrounds showing running counts
                        put_text_with_background(frame, f'Big Jumps: {current_counts["big"]}', (50, 50))
                        put_text_with_background(frame, f'Pogos: {current_counts["small"]}', (50, 100))
                        put_text_with_background(frame, f'Total Jumps: {current_counts["total"]}', (50, 150))
                        put_text_with_background(frame, f'Reps/min: {reps_per_minute:.1f}', (50, 200))

                        # Save the frame with overlays
                        cv2.imwrite(frame_path, frame)

                # Use FFmpeg to compile frames into video
                output_video_path = "output_with_overlays.mp4"
                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(frames_dir, "frame_%04d.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    output_video_path
                ]
                subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Display the processed video
                st.write("### Processed Video with Visual Overlays and Jump Classification")
                st.video(output_video_path)

            finally:
                # Cleanup
                cleanup_resources()
                if os.path.exists(video_path):
                    os.remove(video_path)
                for frame_file in os.listdir(frames_dir):
                    os.remove(os.path.join(frames_dir, frame_file))
                os.rmdir(frames_dir)
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)

    else:
        st.info("Please fill in and save the session information above before uploading a video.")

    # Add a reset button at the bottom
    if st.button("Reset Session"):
        if 'session_info_saved' in st.session_state:
            del st.session_state.session_info_saved
        st.experimental_rerun()

with tab2:
    # Data Management Tab
    st.title("Analysis Data Management")
    
    # Load and display data
    df = create_analysis_dataframe()
    
    if df is not None:
        # Data filtering options
        st.subheader("Filter Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_athlete = st.multiselect(
                "Select Athletes",
                options=sorted(df['athlete_name'].unique()),
                default=sorted(df['athlete_name'].unique())
            )
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(df['session_date'].min(), df['session_date'].max()),
                key="date_range"
            )
        with col3:
            selected_workouts = st.multiselect(
                "Workout Numbers",
                options=sorted(df['workout_number'].unique()),
                default=sorted(df['workout_number'].unique())
            )

        # Filter DataFrame
        filtered_df = df[
            (df['athlete_name'].isin(selected_athlete)) &
            (df['session_date'].dt.date >= date_range[0]) &
            (df['session_date'].dt.date <= date_range[1]) &
            (df['workout_number'].isin(selected_workouts))
        ]

        # Summary Statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sessions", len(filtered_df))
        with col2:
            st.metric("Avg Jumps/Session", round(filtered_df['total_jumps'].mean(), 1))
        with col3:
            st.metric("Avg Big Jumps", round(filtered_df['big_jumps'].mean(), 1))
        with col4:
            st.metric("Avg Small Jumps", round(filtered_df['small_jumps'].mean(), 1))

        # Visualizations
        st.subheader("Performance Trends")
        
        # Time series plot
        fig_time = plt.figure(fig

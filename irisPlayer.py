import cv2
import mediapipe as mp
import numpy as np
import threading
from tkinter import *
from tkinter import filedialog, messagebox
import pygame.mixer as mixer
import pygame
import os
import tensorflow as tf
import time

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='iris_gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Define the eye and iris landmark indices, including iris centers
right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
left_iris_indices = [474, 475, 476, 477, 473]  # Include iris center point 468
right_iris_indices = [469, 470, 471, 472, 468]  # Include iris center point 473

# Combine all the indices
eye_iris_indices = left_eye_indices + left_iris_indices +right_eye_indices +  right_iris_indices

# Function to normalize landmarks
def normalize_landmarks(landmarks, indices):
    # Compute the midpoint between the two eyes
    left_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices], axis=0)
    right_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices], axis=0)
    mid_point = (left_eye_center + right_eye_center) / 2.0

    normalized = np.array([[landmarks[i].x - mid_point[0], landmarks[i].y - mid_point[1]] for i in indices])
    return normalized.flatten()

# Labels for gestures
labels = ['volume up', 'volume down', 'next', 'prev', 'center', 'play', 'pause', 'stop']

last_event_time = 0
# debounce_time = 2
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Initializing the mixer
mixer.init()
pygame.init()  # You need to initialize pygame to handle events
song_stopped_manually = False
# Function to load songs from a directory
def load(listbox, directory=None):
    if directory:
        if not os.path.exists(directory):
            messagebox.showwarning("Directory not found", f"The directory {directory} does not exist.")
            return False
        os.chdir(directory)
    else:
        os.chdir(filedialog.askdirectory(title='Open a songs directory'))
    tracks = os.listdir()
    listbox.delete(0, END)  # Clear the listbox before loading new songs
    for track in tracks:
        listbox.insert(END, track)

    return True


# Track the active button globally
active_button = None

def set_active_button(btn):
    global active_button
    # Reset the background of the previously active button if there is one
    if active_button:
        active_button.config(bg='Gray80')

    # Set the new active button's background to Gray60
    btn.config(bg='Gray60')

    # Update the active button
    active_button = btn


# Modified functions to include active button tracking
def play_song(song_name: StringVar, songs_list: Listbox, status: StringVar, pause_btn: Button, btn):
    set_active_button(btn)
    song_name.set(songs_list.get(ACTIVE))
    mixer.music.load(songs_list.get(ACTIVE))
    mixer.music.play()
    status.set("Song PLAYING")
    pause_btn["text"] = "Pause"
    print(f"Now playing: {song_name.get()}")

def stop_song(status: StringVar, pause_btn: Button, btn):
    global song_stopped_manually
    song_stopped_manually = True
    set_active_button(btn)
    mixer.music.stop()
    status.set("Song STOPPED")
    pause_btn["text"] = "Pause"

def toggle_pause(status: StringVar, pause_btn: Button, btn):
    if pause_btn["text"] == "Pause": # If music is playing
        mixer.music.pause() # Pause the music
        status.set("Song PAUSED") # Update status label
        pause_btn["text"] = "Resume" # Change button text to "Resume"
        set_active_button(pause_btn) # Highlight Pause button
    else: # If music is paused
        mixer.music.unpause() # Unpause the music
        status.set("Song PLAYING") # Update status
        pause_btn["text"] = "Pause" # Change button text back to "Pause"
        set_active_button(play_btn)  # Highlight the Play button

def next_song(song_name: StringVar, songs_list: Listbox, status: StringVar, pause_btn: Button, btn):
    current_selection = songs_list.curselection()
    next_selection = (current_selection[0] + 1) % songs_list.size()
    songs_list.selection_clear(0, END)
    songs_list.activate(next_selection)
    songs_list.selection_set(next_selection)
    songs_list.see(next_selection)
    play_song(song_name, songs_list, status, pause_btn, btn)
    set_active_button(play_btn)  # Highlight the Play button

def previous_song(song_name: StringVar, songs_list: Listbox, status: StringVar, pause_btn: Button, btn):
    current_selection = songs_list.curselection()
    prev_selection = (current_selection[0] - 1) % songs_list.size()
    songs_list.selection_clear(0, END)
    songs_list.activate(prev_selection)
    songs_list.selection_set(prev_selection)
    songs_list.see(prev_selection)
    play_song(song_name, songs_list, status, pause_btn, btn)
    set_active_button(play_btn)  # Highlight the Play button

# Volume control functions
def volume_up(slider):
    current_volume = mixer.music.get_volume()
    new_volume = min(current_volume + 0.1, 1.0)
    mixer.music.set_volume(new_volume)
    slider.set(new_volume * 100)

def volume_down(slider):
    current_volume = mixer.music.get_volume()
    new_volume = max(current_volume - 0.1, 0.0)
    mixer.music.set_volume(new_volume)
    slider.set(new_volume * 100)
def check_for_song_end(song_name: StringVar, songs_list: Listbox, status: StringVar, pause_btn: Button):
    global song_stopped_manually
    # This function checks for the song end event
    for event in pygame.event.get():
        if event.type == SONG_END:
            if not song_stopped_manually:
                # If song wasn't stopped manually, move to the next song
                next_song(song_name, songs_list, status, pause_btn, play_btn)
            # Reset the flag after checking it
            song_stopped_manually = False

    root.after(100, check_for_song_end, song_name, songs_list, status, pause_btn)

def auto_next_song(song_name: StringVar, songs_list: Listbox, status: StringVar, pause_btn: Button):
    global song_stopped_manually
    if not song_stopped_manually:  # Only proceed if the song wasn't stopped manually
        next_song(song_name, songs_list, status, pause_btn, play_btn)

SONG_END = pygame.USEREVENT + 1
mixer.music.set_endevent(SONG_END)  # Trigger this event when the song ends
# Creating the master GUI
root = Tk()
root.geometry('700x320')  # Adjusted for the new layout
root.title('Music Player')
# root.iconbitmap('music1.ico')
# All the frames
song_frame = LabelFrame(root, text='Current Song', bg='Gray72', width=400, height=90)
song_frame.place(x=0, y=0)

button_frame = LabelFrame(root, text='Control', bg='Gray72', width=400, height=170)
button_frame.place(y=170)

volume_frame = LabelFrame(root, text='Volume', bg='Gray72', width=400, height=80)
volume_frame.place(y=90)

listbox_frame = LabelFrame(root, text='Playlist', bg='Gray72')
listbox_frame.place(x=400, y=0, height=320, width=300)

# All StringVar variables
current_song = StringVar(root, value='<Not selected>')
song_status = StringVar(root, value='<Not Available>')

# Playlist ListBox
playlist = Listbox(listbox_frame, font=('Helvetica', 11), selectbackground='Gold')

scroll_bar = Scrollbar(listbox_frame, orient=VERTICAL)
scroll_bar.pack(side=RIGHT, fill=BOTH)

playlist.config(yscrollcommand=scroll_bar.set)
scroll_bar.config(command=playlist.yview)
playlist.pack(fill=BOTH, expand=True, padx=5, pady=5)

# SongFrame Labels
Label(song_frame, bg='Gray72', font=('Times', 10, 'bold')).place(x=5, y=20)

song_lbl = Label(song_frame, textvariable=current_song, bg='Gray72', font=("Times", 12, 'bold'), anchor="w",
                 justify="left", width=45)
song_lbl.place(x=5, y=18)

# Buttons with active button tracking
prev_btn = Button(button_frame, text='Prev', bg='Gray80', font=("Georgia", 11, 'bold'), width=6,
                  command=lambda: previous_song(current_song, playlist, song_status, pause_btn, prev_btn))
prev_btn.grid(row=0, column=0, padx=2, pady=10)

pause_btn = Button(button_frame, text='Pause', bg='Gray80', font=("Georgia", 11, 'bold'), width=7,
                   command=lambda: toggle_pause(song_status, pause_btn, pause_btn))
pause_btn.grid(row=0, column=1, padx=2, pady=10)

stop_btn = Button(button_frame, text='Stop', bg='Gray80', font=("Georgia", 11, 'bold'), width=7,
                  command=lambda: stop_song(song_status, pause_btn, stop_btn))
stop_btn.grid(row=0, column=2, padx=2, pady=10)

play_btn = Button(button_frame, text='Play', bg='Gray80', font=("Georgia", 11, 'bold'), width=7,
                  command=lambda: play_song(current_song, playlist, song_status, pause_btn, play_btn))
play_btn.grid(row=0, column=3, padx=2, pady=10)

next_btn = Button(button_frame, text='Next', bg='Gray80', font=("Georgia", 11, 'bold'), width=6,
                  command=lambda: next_song(current_song, playlist, song_status, pause_btn, next_btn))
next_btn.grid(row=0, column=4, padx=2, pady=10)

load_btn = Button(button_frame, text='Load Directory', bg='Gray80', font=("Georgia", 11, 'bold'), width=30,
                  command=lambda: load(playlist))
load_btn.grid(row=1, column=0, columnspan=5, pady=12)

# Volume control buttons and slider
vol_down_btn = Button(volume_frame, text='Down', bg='Gray80', font=("Georgia", 11, 'bold'), width=7,
                      command=lambda: volume_down(vol_slider))
vol_down_btn.grid(row=0, column=0, padx=5, pady=10)

vol_slider = Scale(volume_frame, from_=0, to=100, orient=HORIZONTAL, bg='Gray72', font=("Georgia", 10, 'bold'),
                   showvalue=True, length=200, command=lambda v: mixer.music.set_volume(float(v) / 100))
vol_slider.set(50)  # Start at 50%
vol_slider.grid(row=0, column=1, padx=5, pady=10)

vol_up_btn = Button(volume_frame, text='Up', bg='Gray80', font=("Georgia", 11, 'bold'), width=7,
                    command=lambda: volume_up(vol_slider))
vol_up_btn.grid(row=0, column=2, padx=5, pady=10)

# Label at the bottom that displays the state of the music
Label(root, textvariable=song_status, bg='Gray72', font=('Times', 12), justify=LEFT).pack(side=BOTTOM, fill=X)

# Loading the default directory on startup
default_directory = r'C:\music'
if not os.path.exists(default_directory):
    os.makedirs(default_directory)
    print(f"Created directory: {default_directory}")
# Load songs from the default directory if it exists
if load(playlist, default_directory):
    playlist.selection_set(0)
    playlist.activate(0)
    current_song.set(playlist.get(0))
check_for_song_end(current_song, playlist, song_status, pause_btn)

cap = cv2.VideoCapture(0)


def process_frame():
    global last_event_time  # Ensure last_event_time is global
    eye_move = ""
    debounce_time = 2  # Set debounce time in seconds
    consecutive_count = 0
    required_consecutive_detections = 10  # Number of consecutive detections required
    last_predicted_label = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB as MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find faces and iris landmarks
        result = face_mesh.process(rgb_frame)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:

                # Normalize the landmarks
                normalized_landmarks = normalize_landmarks(face_landmarks.landmark, eye_iris_indices)
                input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(1, -1)

                # Set the tensor to point to the input data to be inferred
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get the result
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_label = np.argmax(output_data[0])

                # Event counting logic
                if predicted_label == last_predicted_label:
                    consecutive_count += 1
                else:
                    consecutive_count = 1  # Reset the counter if the prediction changes
                    last_predicted_label = predicted_label  # Update the last prediction

                # Only act when the detection is the same for the required consecutive frames
                if consecutive_count == required_consecutive_detections:
                    eye_move = labels[predicted_label]
                    consecutive_count = 0  # Reset the counter after detection

                    current_time = time.time()  # Get the current time
                    if current_time - last_event_time > debounce_time:
                        last_event_time = current_time  # Update last event time

                        if eye_move == "play":
                            play_song(current_song, playlist, song_status, pause_btn, play_btn)
                        elif eye_move == "stop":
                            stop_song(song_status, pause_btn, stop_btn)
                        elif eye_move == "pause":
                            toggle_pause(song_status, pause_btn, pause_btn)
                        elif eye_move == "next":
                            next_song(current_song, playlist, song_status, pause_btn, next_btn)
                        elif eye_move == "prev":
                            previous_song(current_song, playlist, song_status, pause_btn, prev_btn)
                        elif eye_move == "volume up":
                            volume_up(vol_slider)
                        elif eye_move == "volume down":
                            volume_down(vol_slider)
                        elif eye_move == "center":
                            continue  # No action for center

            # Display the predicted gesture
            cv2.putText(frame, f'Gesture: {eye_move}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Start the MediaPipe and OpenCV processing in a separate thread
threading.Thread(target=process_frame, daemon=True).start()

# Start the Tkinter event loop
root.mainloop()

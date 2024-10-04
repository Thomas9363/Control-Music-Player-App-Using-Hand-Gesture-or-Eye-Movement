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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils #use drawing utility
handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1) #define landmark style
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1) #define connection style

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='hand_gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

last_event_time = 0
# Gesture mapping
gesture_names = ["volume up", "volume down", "prev", "next", "nothing", "stop", "pause", "play"]
width = 640
height = 480
#function find the bounding box
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    # Take the first landmark as the reference point (0, 0)
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
    return normalized.flatten()


last_event_time = 0
debounce_time = 3

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
    if pause_btn["text"] == "Pause":
        mixer.music.pause()
        status.set("Song PAUSED")
        pause_btn["text"] = "Resume"
        set_active_button(pause_btn)
    else:
        mixer.music.unpause()
        status.set("Song PLAYING")
        pause_btn["text"] = "Pause"
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
song_frame = LabelFrame(root, text='Current Song', bg='Gray72', width=400, height=80)
song_frame.place(x=0, y=0)

button_frame = LabelFrame(root, text='Control', bg='Gray72', width=400, height=170)
button_frame.place(y=170)

volume_frame = LabelFrame(root, text='Volume', bg='Gray72', width=400, height=80)
volume_frame.place(y=80)

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

if load(playlist, default_directory):
    playlist.selection_set(0)
    playlist.activate(0)
    current_song.set(playlist.get(0))# Integrating the hand gesture control with MediaPipe and OpenCV
check_for_song_end(current_song, playlist, song_status, pause_btn)

cap = cv2.VideoCapture(0)

def process_frame():
    global last_event_time, prev_time, prev_fps  # Ensure last_gesture_time is global
    gesture_name = ""
    debounce_time = 2  # Set debounce time in seconds
    consecutive_count = 0
    required_consecutive_detections = 20  # Number of consecutive detections required
    last_predicted_class = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB as MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame to find hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Normalize the landmarks
                normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

                # Reshape and prepare input data
                input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(input_details[0]['shape'])

                # Set the input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get the output tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Interpret the results
                predicted_class = np.argmax(output_data)
                # Event counting logic
                if predicted_class == last_predicted_class:
                    consecutive_count += 1
                else:
                    consecutive_count = 1  # Reset the counter if the prediction changes
                    last_predicted_class = predicted_class  # Update the last prediction

                # Only act when the detection is the same for the required consecutive frames
                if consecutive_count == required_consecutive_detections:
                    gesture_name = gesture_names[predicted_class]

                    consecutive_count = 0  # Reset the counter after detection
                    current_time = time.time()  # Get the current time
                    if current_time - last_event_time > debounce_time:
                        last_event_time = current_time  # Update last event time



                # print(labels[predicted_label])
                        if gesture_name =="play":
                            play_song(current_song, playlist, song_status, pause_btn, play_btn)
                        elif gesture_name=="stop":
                            stop_song(song_status, pause_btn, stop_btn)
                        elif gesture_name=="pause":
                            toggle_pause(song_status, pause_btn, pause_btn)
                        elif gesture_name=="next":
                            next_song(current_song, playlist, song_status, pause_btn, next_btn)
                        elif gesture_name=="prev":
                            previous_song(current_song, playlist, song_status, pause_btn, prev_btn)
                        elif gesture_name=="volume up":
                            volume_up(vol_slider)
                        elif gesture_name=="volume down":
                            volume_down(vol_slider)

        # Draw the hand landmarks on the frame
                mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, handLmsStyle,
                              handConStyle)  # draw landmarks styles
                brect = calc_bounding_rect(frame, hand_landmarks)  # Calculate the bounding rectangle
                cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0),
                      1)  # Draw the bounding rectangle
            # Display the predicted gesture on the frame
                cv2.putText(frame, f'Gesture: {gesture_name}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)


        # Display the frame
        cv2.imshow('Iris Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Start the MediaPipe and OpenCV processing in a separate thread
threading.Thread(target=process_frame, daemon=True).start()

# Start the Tkinter event loop
root.mainloop()

# Importing all the necessary modules
from tkinter import *
from tkinter import filedialog, messagebox
import pygame.mixer as mixer
import pygame
import os

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

if load(playlist, default_directory):
    playlist.selection_set(0)
    playlist.activate(0)
    current_song.set(playlist.get(0))

# Finalizing the GUI
root.update()
check_for_song_end(current_song, playlist, song_status, pause_btn)
root.mainloop()

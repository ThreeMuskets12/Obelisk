import PySimpleGUI as sg

# Define initial window size and maximum window size
initial_size = (540, 960)  # 9:16 aspect ratio
max_size = (1080, 1920)    # Maximum size allowed, also 9:16 aspect ratio

aspect_ratio = 9 / 16  # 9:16 aspect ratio

# Create layout with centered text
layout = [
    [sg.Text("Hello World", justification='center', key='-TEXT-', text_color='white', background_color='black')]
]

# Create the window
window = sg.Window("DISPLAY OUTPUT", layout, resizable=True, size=initial_size,
                   background_color='black', element_justification='center',
                   finalize=True, location=(0, 0), no_titlebar=False, margins=(0, 0))

# Set the window max size
window.TKroot.minsize(0, 0)  # no minimum size
window.TKroot.maxsize(*max_size)

# Event loop to manage resizing and window events
while True:
    event, values = window.read(timeout=100)
    if event == sg.WIN_CLOSED:
        break

    # Get the current window size
    current_size = window.size
    current_width, current_height = current_size

    # Initialize new_width and new_height with the current dimensions
    new_width = current_width
    new_height = current_height

    # Enforce the aspect ratio by recalculating one of the dimensions
    if current_width / current_height != aspect_ratio:
        # Adjust dimensions to maintain the 9:16 aspect ratio
        if current_width / current_height > aspect_ratio:
            new_width = int(current_height * aspect_ratio)
        else:
            new_height = int(current_width / aspect_ratio)

        # Resize the window to the new dimensions with the correct aspect ratio
        window.TKroot.geometry(f"{new_width}x{new_height}")

    # Calculate the scaling factor for the font size based on the window size
    scale_factor = min(new_width / initial_size[0], new_height / initial_size[1])

    # Resize the text font size proportionally
    font_size = int(48 * scale_factor)
    window['-TEXT-'].update(font=("Helvetica", font_size))

window.close()

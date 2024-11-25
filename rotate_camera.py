import time
from servo_motor import SetAngle, Cleanup
from driver import get_character_count

# Define the positions of each chalkboard in STC
# Update positions while testings
def RotateCamera(position):
    if position == 1:
        print("Position 1 - 0 degrees")
        SetAngle(0)
    elif position == 2:
        print("Position 2 - 60 degrees")
        SetAngle(60)
    elif position == 3:
        print("Position 3 - 120 degrees")
        SetAngle(120)
    elif position == 4:
        print("Position 4 - 180 degrees")
        SetAngle(180)
    else:
        print("Invalid position")

# Ask William if this is necessary based on how characters are being fed
def read_previous_count():
    try:
        with open("character_count.txt", "r") as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return -1  # File not exist

def save_previous_count(count):
    with open("character_count.txt", "w") as file:
        file.write(str(count))

# William's code will provide the number of TOTAL characters on each chalkboard
def start_motor():
    time.sleep(2)  # Begin camera movement after 2 seconds
    previous_count = read_previous_count()  # Initialize the previous count
    current_count = 0    # Initialize the current count
    position = 1         # Start at position 1 (0 degrees)
    no_change = 0

    while True:
        current_count = get_character_count()

        print(f"Number of total characters on chalkboard {position}: {current_count}")

        # STOP ROTATING - Lecture completed
        # Stop rotating if 
        if current_count == 0:
            no_characters_counter += 1
        else:
            no_characters_counter = 0

        # Check if we've reached the threshold of consecutive 0 character counts
        if no_characters_counter >= 5:  # If we detect 0 characters for 5 checks (e.g., 50 seconds)
            print("No change detected after 5 pictures. Stopping rotation.")
            Cleanup()   # Clean up GPIO and stop PWM
            return False

        # ROTATE TO NEXT POSITION
        # Check if the count has stopped changing based on tolerance
        if abs(current_count - previous_count) == 0:
            print(f"Count stopped changing. Rotating.")
            # Move to the next position in clockwise motion
            position = position % 4 + 1
            RotateCamera(position)

            # Reset previous count for next cycle
            save_previous_count(current_count)

            # Exit the loop or break if you only want one rotation after detecting stability
            break
        
    return True

if __name__ == "__main__":
    main()

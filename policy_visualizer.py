import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()

# Set up the width and height of each tile
image_w = 100
image_h = 100

# Read in all of the images to create the grid
file = 'Images/jerry.jpg'
image = Image.open(file)
jerry = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/trap.jpg'
image = Image.open(file)
trap = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/0001.png'
image = Image.open(file)
im_0001 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/0010.png'
image = Image.open(file)
im_0010 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/0011.png'
image = Image.open(file)
im_0011 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/0100.png'
image = Image.open(file)
im_0100 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/0101.png'
image = Image.open(file)
im_0101 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/0110.png'
image = Image.open(file)
im_0110 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/0111.png'
image = Image.open(file)
im_0111 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1000.png'
image = Image.open(file)
im_1000 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1001.png'
image = Image.open(file)
im_1001 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1010.png'
image = Image.open(file)
im_1010 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1011.png'
image = Image.open(file)
im_1011 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1100.png'
image = Image.open(file)
im_1100 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1101.png'
image = Image.open(file)
im_1101 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1110.png'
image = Image.open(file)
im_1110 = ImageTk.PhotoImage(image.resize((image_w, image_h)))

file = 'Images/Arrows/1111.png'
image = Image.open(file)
im_1111 = ImageTk.PhotoImage(image.resize((image_w, image_h)))


# Read the policy to visualize from a file
filename = 'policy_out'
with open(filename, 'r') as f:
	lines = f.readlines()

# The first line of the file contains the grid dimensions (row, col)
dimensions = [int(x) for x in lines[0].split(' ')]
tile = None

for r in range(dimensions[0]):
	for c in range(dimensions[1]):
		index = r*dimensions[1] + c
		if (lines[index + 1] == 'Trap\n'):
			tile = trap
		elif (lines[index + 1] == 'Cheese\n'):
			tile = jerry
		else:
			mask = int(lines[index + 1])
			if (mask == 1):
				tile = im_0001
			elif (mask == 2):
				tile = im_0010
			elif (mask == 3):
				tile = im_0011
			elif (mask == 4):
				tile = im_0100
			elif (mask == 5):
				tile = im_0101
			elif (mask == 6):
				tile = im_0110
			elif (mask == 7):
				tile = im_0111
			elif (mask == 8):
				tile = im_1000
			elif (mask == 9):
				tile = im_1001
			elif (mask == 10):
				tile = im_1010
			elif (mask == 11):
				tile = im_1011
			elif (mask == 12):
				tile = im_1100
			elif (mask == 13):
				tile = im_1101
			elif (mask == 14):
				tile = im_1110
			elif (mask == 15):
				tile = im_1111
		tk.Label(root, image=tile, borderwidth=1, relief='solid').grid(row=r, column=c)

# Start up the GUI, containing the visualized policy
root.mainloop()
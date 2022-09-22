import tkinter as tk
import numpy as np
import os
import shutil
from PIL import Image, ImageTk

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_padding = 100
window_width = screen_width - 2*window_padding
window_height = screen_height - 2*window_padding
root.geometry(f'{window_width}x{window_height}+{window_padding}+{window_padding}')
root.configure(bg='cyan')
canvas = tk.Canvas(root, width = window_width, height = window_height,bg='white')  
canvas.configure()
canvas.pack()

print(window_width/window_height,window_width,window_height)
rows = 3
columns = 4
image_width = window_width / columns
image_height = window_height/ rows

# source_dir ='./dicom/NIH_1stPA_Norm_Fib/Fibrosis' 
# dest_dir ='./dicom/NIH_1stPA_Norm_Fib/screened_out/Fibrosis' 

source_dir ='./dicom/NIH_1stPA_Norm_Fib/No Finding' 
dest_dir ='./dicom/NIH_1stPA_Norm_Fib/screened_out/No Finding' 

file_names = os.listdir(source_dir)
total_image_count = len(file_names)
file_names_iter = iter(file_names) 

image_tags = np.zeros((rows,columns),dtype=np.int8)
images_to_reject = np.zeros((rows,columns),dtype=np.int8)
current_file_names = np.empty((rows,columns),dtype="U16")
images=[]

batch_number=0

def reset_vars():
  global image_tags, images_to_reject, current_file_names
  image_tags = np.zeros((rows,columns),dtype=np.int8)
  images_to_reject = np.zeros((rows,columns),dtype=np.int8)
  current_file_names = np.empty((rows,columns),dtype="U16")

def load_more_images():
  global images, batch_number, total_image_count
  images=[]
  batch_number += 1
  root.title(f'Screening batch {batch_number}, image {rows*columns*batch_number} of {total_image_count} images in: ||||| {source_dir} ==> {dest_dir} ')
  # current_file_names=[]
  for i in range(0,columns):
    for j in range(0,rows):
      try:
        file_name = next(file_names_iter)
        print(file_name)
        if(file_name):
          # image= tk.PhotoImage(file=f"{source_dir}/{file_name}")
          PIL_image = Image.open(f"{source_dir}/{file_name}")
          PIL_image = PIL_image.resize((int(image_width),int(image_height)), Image.ANTIALIAS)
          image = ImageTk.PhotoImage(PIL_image)
          images.append(image)
          current_file_names[j][i] = file_name 
          # current_file_names.append(file_name)
          xyz=canvas.create_image(i*image_width,j*image_height, anchor='nw',image=image)
          image_tags[j][i]=xyz 
      finally:
        pass

reset_vars()
load_more_images()
# print(image_tags)
# print(images_to_reject)
# print(current_file_names)
key_map = {'q':0,'w':1,'e':2,'r':3,'a':4,'s':5,'d':6,'f':7,'z':8,'x':9,'c':10,'v':11}
# image= tk.PhotoImage(file="pictures/TrapFit.png")  
# image.config(color=(1,0,0))
# xyz=canvas.create_image(20,20, anchor='nw',image=image)
# myimage= tk.Image #tk.Image("png","pictures/818_heatmap.png")
# L1=tk.Label(root, width=200, height=200, borderwidth=2, border = 2)
# L1.place(x=0,y=355)
   
# def down(e):
#   # canvas.delete(xyz)
#   # # if m == 0:
#   print( e.char, e )

def up(e):
  global image_tags, images_to_reject, current_file_names
  # MARK FILES TO REJECT
  if e.char in key_map:
    print(e.char,key_map[e.char])
    n, m = key_map[e.char]%columns, key_map[e.char]//columns
    print(m,n)
    if(images_to_reject[m][n] ==0 ):
      tag=canvas.create_text( n*image_width,m*image_height,text="X", fill="red",anchor='nw', font=('Helvetica 20 bold'))
      images_to_reject[m][n]=tag
    else:
      canvas.delete(images_to_reject[m][n])
      images_to_reject[m][n]=0
    print(images_to_reject)
    print(np.array(images_to_reject).flatten())
    print(np.array(current_file_names).flatten())

  if(e.char == ' '):
    # MOVE REJECTED FILES
    #DELETE ALL THE IMAGES IN THE CANVAS
    image_to_reject_flat = np.array(images_to_reject).flatten()
    file_names_flat = np.array(current_file_names).flatten()
    for index in range(len(file_names_flat)):
      print(index, image_to_reject_flat[index])
      if(image_to_reject_flat[index] != 0):
        shutil.move(f"{source_dir}/{file_names_flat[index]}",f"{dest_dir}/{file_names_flat[index]}")
        canvas.delete(image_to_reject_flat[index]) # DELETE THE RED X's FROM CANVAS
    # DELETE THE IMAGE FROM CANVAS
    for tag in iter(np.array(image_tags).flatten()):
      canvas.delete(tag)
    reset_vars()
    load_more_images()
    # image_tags=[]

  # canvas.create_text(0, 0, text="HELLO WORLD", fill="red",anchor='nw', font=('Helvetica 15 bold'))


# root.bind('<KeyPress>', down)
root.bind('<KeyRelease>', up)
root.mainloop()
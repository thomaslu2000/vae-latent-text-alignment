from PIL import Image
import numpy as np
#give image input as numpy array
#given rgb input as an array of RGB values like np.array([0,255,255])

def colormetricx(imager, color):
  im = Image.fromarray(imager)
  im.save("a.png")
  im = Image.open("a.png")
  arr = np.array(im.getdata())
  unique_colors, counts = np.unique(arr.reshape(-1, arr.shape[1]), axis=0, return_counts=True)
  r1 = color[0]
  g1 = color[1]
  b1 = color[2]
  r2 = unique_colors[0][0]
  g2 = unique_colors[0][1]
  b2 = unique_colors[0][2]

  d=np.sqrt((np.absolute(r2-r1))^2+(np.absolute(g2-g1))^2+(np.absolute(b2-b1))^2)
  p=d/np.sqrt((255)^2+(255)^2+(255)^2)
  return (1-p)*100

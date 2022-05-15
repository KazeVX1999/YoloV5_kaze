from PIL import Image
import imagehash


def compareImage(img1, img2):
  image1= imagehash.average_hash(Image.open(img1))
  image2 = imagehash.average_hash(Image.open(img2))

  threshold = 15
  if image1 - image2 < threshold:
    print('images are similar')
  else:
    print('images are not similar')


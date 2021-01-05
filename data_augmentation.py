from PIL import Image
import PIL
import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import cv2
from scipy.ndimage.morphology import distance_transform_edt as bwdist


rootDirGt = "/content/drive/MyDrive/IP - 7th sem/SK-SMALL/SK506/groundTruth/train/"
listData = sorted(os.listdir(rootDirGt))

trainDirImg = "/content/drive/MyDrive/IP - 7th sem/SK-SMALL/SK506/images/train/"
listimages = sorted(os.listdir(trainDirImg))

output_dir = "/content/drive/My Drive/IP - 7th sem/SK-SMALL/SK506/aug_data/groundTruth/"
img_dir = "/content/drive/My Drive/IP - 7th sem/SK-SMALL/SK506/aug_data/images/"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

file1 = open("/content/drive/My Drive/IP - 7th sem/SK-SMALL/SK506/aug_data/pairList.txt", 'w') 
for scale in tqdm([0.8, 1, 1.2]):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    for angle in tqdm([0, 90, 180, 270]):
        # os.makedirs(output_dir + str(scale) + "/o/" + str(angle) + "/f/", exist_ok=True)
        # os.makedirs(img_dir + str(scale) + "/o/" + str(angle) + "/f/", exist_ok=True)
        for flip in tqdm([0, 1, 2]):
            # os.makedirs(output_dir + str(scale) + "/o/" + str(angle) + "/f/" + str(flip), exist_ok=True)
            # os.makedirs(img_dir + str(scale) + "/o/" + str(angle) + "/f/" + str(flip), exist_ok=True)
            for i in range(len(listData)):
                targetName = listData[i]
                gtpath = output_dir + str(scale) + str(angle) + str(flip) + "_"
                impath = img_dir + str(scale) + str(angle) + str(flip) + "_"
                name = targetName.replace(".mat", ".png")
                
                imageTrain = cv2.imread(trainDirImg + listimages[i])
                itemGround = loadmat(rootDirGt + targetName)
                edge, skeleton = itemGround['edge'], itemGround['symmetry']
                dist = 2*bwdist(1 - edge)
                appl = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
                img = appl(dist,skeleton)
                img = Image.fromarray(img.astype(np.uint8))
                img = img.rotate(angle, PIL.Image.BICUBIC, True)
                imageTrain = Image.fromarray(imageTrain)
                imageTrain = imageTrain.rotate(angle, PIL.Image.BICUBIC, True)
                new_size = (int(np.ceil(scale*img.size[0])), int(np.ceil(scale*img.size[1])))
                img = img.resize(new_size)

                new_size_train = (int(np.ceil(scale*imageTrain.size[0])), int(np.ceil(scale*imageTrain.size[1])))
                imageTrain = imageTrain.resize(new_size_train)
                if flip == 1:
                    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                    imageTrain = imageTrain.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                elif flip == 2:
                    img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                    imageTrain = imageTrain.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                img.save(gtpath + name)
                imageTrain.save(impath+ listimages[i])

                file1.write(impath+ listimages[i] + "," + gtpath + name + "\n")
                """
                ground = Image.open(path + name)
                img = Image.open(path.replace("ed_scale","im_scale") + name.replace(".png", ".jpg"))
                
                ground_s = ground.size
                img_s = img.size
                if (ground_s != img_s):
                    print("Image and groundTruth sizes do not fit!")
                """

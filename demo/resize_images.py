from PIL import Image
import os

# path = "../pose_images/Database_Proyecto/DownwardDog/Right/"
# path = "../pose_images/Database_Proyecto/Plank/Right/"
# path = "../pose_images/Database_Proyecto/Tree/Right/"
path = "../pose_images/Database_Proyecto/WarriorII/Right/"
dirs = os.listdir(path)
final_size = 512;


def resize_aspect_fit():

    for item in dirs:
        if os.path.isfile(path + item):
            print(os.path.isfile(path + item))
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            size = im.size
            ratio = float(final_size) / max(size)
            new_image_size = tuple([int(x * ratio) for x in size])
            im = im.resize(new_image_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (final_size, final_size))
            new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))
            # new_im.save(f + '_downward_right_resized.jpeg', 'JPEG', quality=90)
            # new_im.save(f + '_plank_right_resized.jpeg', 'JPEG', quality=90)
            # new_im.save(f + '_tree_right_resized.jpeg', 'JPEG', quality=90)
            new_im.save(f + '_warrior_right_resized.jpeg', 'JPEG', quality=90)
            print("RESIZE DONE")


resize_aspect_fit()

import os
import random
from utils.motion_process import *


def get_motion_blur_image(input: np.ndarray, motion_angle: int, motion_dis: int):
    PSF = straight_motion_psf((input.shape[0], input.shape[1]), motion_angle, motion_dis)
    R, G, B = cv2.split(input)
    blurred_image_R, blurred_image_G, blurred_image_B = \
        np.abs(make_blurred(R, PSF, 1e-3)), np.abs(make_blurred(G, PSF, 1e-3)), np.abs(make_blurred(B, PSF, 1e-3))
    blurred_image = cv2.merge([blurred_image_R, blurred_image_G, blurred_image_B])

    return np.uint8(blurred_image)


train_img_dir, val_img_dir, test_img_dir = \
    'L:/7_UAV_crack_segmentation/img_dir/train', \
    'L:/7_UAV_crack_segmentation/img_dir/val', \
    'L:/7_UAV_crack_segmentation/img_dir/test'

train_blur_img_dir, val_blur_img_dir, test_blur_img_dir = \
    'L:/7_UAV_crack_segmentation/blur_img_dir/train', \
    'L:/7_UAV_crack_segmentation/blur_img_dir/val', \
    'L:/7_UAV_crack_segmentation/blur_img_dir/test'

train_random_blur_img_dir, val_random_blur_img_dir, test_random_blur_img_dir = \
    'L:/7_UAV_crack_segmentation/random_blur_img_dir/train', \
    'L:/7_UAV_crack_segmentation/random_blur_img_dir/val', \
    'L:/7_UAV_crack_segmentation/random_blur_img_dir/test'

# for m, n in zip([train_img_dir, val_img_dir, test_img_dir],
#                 [train_blur_img_dir, val_blur_img_dir, test_blur_img_dir]):
#     for file in os.listdir(m):
#         image = cv2.imread(os.path.join(m, file))
#         blur_image = get_motion_blur_image(image, 45, 10)
#
#         cv2.imwrite(os.path.join(n, file), blur_image)

# for m, n in zip([train_img_dir, val_img_dir, test_img_dir],
#                 [train_random_blur_img_dir, val_random_blur_img_dir, test_random_blur_img_dir]):
#     for file in os.listdir(m):
#         image = cv2.imread(os.path.join(m, file))
#
#         i = random.randint(0, 180)
#         j = random.randint(7, 15)
#
#         random_blur_image = get_motion_blur_image(image, i, j)
#
#         cv2.imwrite(os.path.join(n, file), random_blur_image)

train_earth_img_dir, val_earth_img_dir, test_earth_img_dir = \
    'L:/6_UAV_earthquake_segmentation/img_dir/train', \
    'L:/6_UAV_earthquake_segmentation/img_dir/val', \
    'L:/6_UAV_earthquake_segmentation/img_dir/test'

train_earth_blur_img_dir, val_earth_blur_img_dir, test_earth_blur_img_dir = \
    'L:/6_UAV_earthquake_segmentation/blur_img_dir/train', \
    'L:/6_UAV_earthquake_segmentation/blur_img_dir/val', \
    'L:/6_UAV_earthquake_segmentation/blur_img_dir/test'

train_earth_random_blur_img_dir, val_earth_random_blur_img_dir, test_earth_random_blur_img_dir = \
    'L:/6_UAV_earthquake_segmentation/random_blur_img_dir/train', \
    'L:/6_UAV_earthquake_segmentation/random_blur_img_dir/val', \
    'L:/6_UAV_earthquake_segmentation/random_blur_img_dir/test'

# for m, n in zip([train_earth_img_dir, val_earth_img_dir, test_earth_img_dir],
#                 [train_earth_blur_img_dir, val_earth_blur_img_dir, test_earth_blur_img_dir]):
#     for file in os.listdir(m):
#         image = cv2.imread(os.path.join(m, file))
#         blur_image = get_motion_blur_image(image, 45, 10)
#
#         cv2.imwrite(os.path.join(n, file), blur_image)
#
# for m, n in zip([train_earth_img_dir, val_earth_img_dir, test_earth_img_dir],
#                 [train_earth_random_blur_img_dir, val_earth_random_blur_img_dir, test_earth_random_blur_img_dir]):
#     for file in os.listdir(m):
#         image = cv2.imread(os.path.join(m, file))
#
#         i = random.randint(0, 180)
#         j = random.randint(7, 15)
#
#         random_blur_image = get_motion_blur_image(image, i, j)
#
#         cv2.imwrite(os.path.join(n, file), random_blur_image)


for m in os.listdir('L:/7_UAV_crack_segmentation/ann_dir/'):
    for n in os.listdir(os.path.join('L:/7_UAV_crack_segmentation/ann_dir/', m)):
        image = cv2.imread(os.path.join('L:/7_UAV_crack_segmentation/ann_dir/', m, n))
        # image = np.uint8(image[:, :, 0] > 127.5)
        image = cv2.resize(image, (512, 512))
        image = np.uint8(image[:, :, 0] > 127.5)
        cv2.imwrite(os.path.join('L:/7_UAV_crack_segmentation/ann_dir/', m, n), image)

for m in os.listdir('L:/6_UAV_earthquake_segmentation/ann_dir/'):
    for n in os.listdir(os.path.join('L:/6_UAV_earthquake_segmentation/ann_dir/', m)):
        image = cv2.imread(os.path.join('L:/6_UAV_earthquake_segmentation/ann_dir/', m, n))
        # image = np.uint8(image[:, :, 0] > 127.5)
        image = cv2.resize(image, (512, 512))
        image = np.uint8(image[:, :, 0] > 127.5)
        cv2.imwrite(os.path.join('L:/6_UAV_earthquake_segmentation/ann_dir/', m, n), image)

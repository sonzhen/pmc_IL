import numpy as np


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def encode_npy_to_pil(bev_array):
    """
    将多通道二值 BEV 语义图压缩编码为 3 通道 RGB 图像（无损压缩）
    
    原理：利用位运算，每个 uint8 通道可存储 5 个二值通道：
        R 通道: bit7=ch0, bit6=ch1, bit5=ch2, bit4=ch3, bit3=ch4
        G 通道: bit7=ch5, bit6=ch6, bit5=ch7, bit4=ch8, bit3=ch9
        B 通道: bit7=ch10, bit6=ch11, bit5=ch12, bit4=ch13, bit3=ch14
    共可编码 15 个二值语义通道 → 1 张 RGB 图像
    
    Args:
        bev_array: numpy array [C, W, H]，C 个二值语义通道（0 或 1）
    Returns:
        img: numpy array [3, W, H]，压缩后的 RGB 图像（uint8）
    """
    c, w, h = bev_array.shape

    img = np.zeros([3, w, h]).astype('uint8')     # 输出 3 通道 RGB
    bev = np.ceil(bev_array).astype('uint8')      # 确保值为 0/1 整数

    for i in range(c):
        if 0 <= i <= 4:
            # 通道 0~4 → 编码到 R 通道的 bit7~bit3
            img[0] = img[0] | (bev[i] << (8 - i - 1))
        elif 5 <= i <= 9:
            # 通道 5~9 → 编码到 G 通道的 bit7~bit3
            img[1] = img[1] | (bev[i] << (8 - (i - 5) - 1))
        elif 10 <= i <= 14:
            # 通道 10~14 → 编码到 B 通道的 bit7~bit3
            img[2] = img[2] | (bev[i] << (8 - (i - 10) - 1))

    return img

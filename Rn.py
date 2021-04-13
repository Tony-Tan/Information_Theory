# MacKay, David J. C. Information Theory, Inference, and Learning Algorithms.
# Cambridge, UK ; New York: Cambridge University Press, 2003.
# page 7. R3 algorithm

import cv2
import copy
import numpy as np

if __name__ == '__main__':
    R = 3
    width = 300
    height = 300
    src = cv2.imread('Rn/Dilbert_Image.png', 0)
    src = cv2.resize(src, (height, width))
    _, src = cv2.threshold(src, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    # correct probability f
    f = 0.1
    encoder_t = [copy.deepcopy(src) for i in range(R)]
    receive_r = [np.zeros((width, height, 1), np.uint8) for i in range(R)]
    # transmit
    for i in range(R):
        r_i = receive_r[i]
        for y in range(height):
            for x in range(width):
                if np.random.rand(1)[0] < 0.1:
                    # flip the pixel
                    if encoder_t[i][y][x] == 255:
                        r_i[y][x] = 0
                    else:
                        r_i[y][x] = 255
                else:
                    r_i[y][x] = encoder_t[i][y][x]
    # decoding
    hat_s = np.zeros((width, height, 1), np.uint8)
    for y in range(height):
        for x in range(width):
            pixel_value_sum = 0.0
            for i in range(R):
                pixel_value_sum += receive_r[i][y][x]
            if pixel_value_sum > 255.0*R/2.:
                hat_s[y][x] = 255
            else:
                hat_s[y][x] = 0
    noise_remain = 0
    for y in range(height):
        for x in range(width):
            if hat_s[y][x] != src[y][x]:
                noise_remain += 1
    print('flip rate of the entire image: ', noise_remain/(width*height))
    cv2.imshow('s', src)
    cv2.imshow('t', cv2.hconcat(encoder_t[0:2]))
    cv2.imshow('receive', cv2.hconcat(receive_r[0:2]))
    cv2.imshow('hat_s', hat_s)
    cv2.waitKey(0)

import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Initialize Pygame
pygame.init()

FONT = pygame.font.Font(None, 36)
WIDTH, HEIGHT = 640, 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

BOUNDRY = 5
img_cnt = 0
PREDICT = True
IMAGESAVE = False

# Load your new character model
MODEL = load_model('path/to/ml/model/character_model.h5')

# Create labels dictionary for letters A-Z
LABELS = {i: chr(65 + i) for i in range(26)}  # 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Character Recognizer")

isDrawing = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and isDrawing:
            xcord, ycord = event.pos
            pygame.draw.circle(screen, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            isDrawing = True

        if event.type == MOUSEBUTTONUP:
            isDrawing = False

            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRY, 0), min(WIDTH, number_xcord[-1] + BOUNDRY)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRY, 0), min(HEIGHT, number_ycord[-1] + BOUNDRY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(screen))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f'character_image_{img_cnt}.png', img_arr)
                img_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255.0

                image = image.reshape(1, 28, 28, 1)

                prediction = MODEL.predict(image)
                predicted_class = np.argmax(prediction)
                label = LABELS[predicted_class]

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                screen.blit(textSurface, textRectObj)

        if event.type == KEYDOWN:
            if event.unicode == 'n':
                screen.fill(BLACK)

    pygame.display.update()

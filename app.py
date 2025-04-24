import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Initialize Pygame
pygame.init()


FONT = pygame.font.Font(None, 36)
# Set up the display
WIDTH, HEIGHT = 640, 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

BOUNDRY = 5
img_cnt = 0
PREDICT = True

IMAGESAVE = False
MODEL = load_model('/home/abu/Desktop/ml/digit-recognisation/basemodel.h5')

LABELS = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Digit Recognizer")

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
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRY, 0), min(WIDTH, number_ycord[-1] + BOUNDRY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(screen))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f'image.png')
                img_cnt += 1
            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                screen.blit(textSurface, textRectObj)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    screen.fill(BLACK)

        pygame.display.update()
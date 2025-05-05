import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force CPU

import pygame
import sys
import numpy as np
import cv2
from keras.models import load_model

# -----------------------------
# Config
# -----------------------------
WIDTH, HEIGHT = 300, 300
WHITE, BLACK, RED = (255,255,255), (0,0,0), (255,0,0)
SYMBOL_LABELS = {0: '+', 1: '-', 2: '*', 3: '/', 4: '='}
MODEL_PATH = '/home/abu/Desktop/ml/digit-recognisation/symbol_model-v1.h5'
BOUNDARY = 5
FONT_SIZE = 48

def pad_to_square(img):
    h, w = img.shape
    m = max(h, w)
    square = np.zeros((m, m), dtype=img.dtype)
    dy, dx = (m - h)//2, (m - w)//2
    square[dy:dy+h, dx:dx+w] = img
    return square

# -----------------------------
# Load model
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Pygame setup
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Symbol Recognizer')
FONT = pygame.font.Font(None, FONT_SIZE)
screen.fill(BLACK)

# -----------------------------
# Drawing state
# -----------------------------
drawing = False
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(BLACK)
last_pred = ''

# -----------------------------
# Main Loop
# -----------------------------
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        if event.type == pygame.MOUSEMOTION and drawing:
            x,y = event.pos
            pygame.draw.circle(screen, WHITE, (x,y), 4)
            pygame.draw.circle(canvas, WHITE, (x,y), 4)
        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                arr = pygame.surfarray.array3d(canvas)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                nz = cv2.findNonZero(gray)
                if nz is not None:
                    x,y,w,h = cv2.boundingRect(nz)
                    x, y = max(x-BOUNDARY,0), max(y-BOUNDARY,0)
                    w, h = min(w+2*BOUNDARY, WIDTH-x), min(h+2*BOUNDARY, HEIGHT-y)
                    roi = gray[y:y+h, x:x+w]
                    if roi.shape[0] > 5 and roi.shape[1] > 5:
                        # pad, dilate, resize
                        sq = pad_to_square(roi)
                        sq = cv2.dilate(sq, np.ones((3,3),np.uint8), iterations=1)
                        img28 = cv2.resize(sq, (28,28), interpolation=cv2.INTER_AREA)
                        inp = img28.astype(np.float32)/255.0
                        inp = inp.reshape(1,28,28,1)
                        pred = model.predict(inp)
                        last_pred = SYMBOL_LABELS[np.argmax(pred)]
                        # debug save
                        # cv2.imwrite('debug_symbol.png', sq)
            if event.key == pygame.K_c:
                screen.fill(BLACK)
                canvas.fill(BLACK)
                last_pred = ''
    # render prediction
    if last_pred:
        surf = FONT.render(last_pred, True, RED, BLACK)
        rect = surf.get_rect(center=(WIDTH//2, HEIGHT-30))
        screen.blit(surf, rect)
    pygame.display.update()

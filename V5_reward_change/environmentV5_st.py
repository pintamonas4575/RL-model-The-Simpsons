import numpy as np
import math
from PIL import Image
import cv2

class Scratch_Game_Environment5_Streamlit:
    """
    Versión adaptada para Streamlit del entorno Scratch&Win.
    Renderiza una imagen compuesta con fondo, emojis y cuadrícula de colores.
    """
    def __init__(self, frame_size: int, scratching_area: tuple[int, int, int, int], background_path: str = "utils/space.jpg"):
        self.scratched_count = 0
        self.FRAME_SIZE = frame_size
        self.rect_x, self.rect_y = scratching_area[0], scratching_area[1]
        self.rect_width, self.rect_height = scratching_area[2], scratching_area[3]
        self.number_of_rows = math.ceil(self.rect_height / self.FRAME_SIZE)
        self.number_of_columns = math.ceil(self.rect_width / self.FRAME_SIZE)
        self.total_squares = self.number_of_rows * self.number_of_columns
        self.frames_mask = [-1] * self.total_squares  # -1 no rascado, 0 malo, 1 bueno
        self.emoji_paths = ["emojis/axe.png", "emojis/axe.png", "emojis/axe.png"]
        self.emoji_images = [Image.open(path) for path in self.emoji_paths]
        self.background_path = background_path
        self.background_image = Image.open(self.background_path).resize((self.rect_width, self.rect_height))
        self._setup_environment_and_contours()

    def _setup_environment_and_contours(self):
        # 1. Crea imagen auxiliar blanca del área de juego
        aux_img = Image.new("RGBA", (self.rect_width, self.rect_height), (255,255,255,255))

        # 2. Calcula posiciones centradas de los emojis (bboxes)
        emoji_width, emoji_height = self.emoji_images[0].size
        gap = 40
        total_width = 3 * emoji_width + 2 * gap
        start_x = (self.rect_width - total_width) // 2
        y = (self.rect_height - emoji_height) // 2
        self.emoji_bboxes = []
        for i, emoji_img in enumerate(self.emoji_images):
            x = start_x + i * (emoji_width + gap)
            self.emoji_bboxes.append((x, y, x + emoji_width, y + emoji_height))
            aux_img.paste(emoji_img, (x, y), emoji_img.convert("RGBA"))

        # 3. Saca contornos y establece good_frames, y dibuja cuadrícula de colores inicial
        gray = cv2.cvtColor(np.asarray(aux_img), cv2.COLOR_RGBA2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        background = np.ones_like(np.asarray(aux_img)) * 255
        numpy_contours = cv2.drawContours(background, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
        contour_mask = Image.fromarray(numpy_contours).convert("L")
        contour_mask = np.array(contour_mask)

        self.good_frames_idx = set()
        self.squares_image = Image.new("RGBA", (self.rect_width, self.rect_height), (0,0,0,0))
        idx = 0
        for row in range(self.number_of_rows):
            for col in range(self.number_of_columns):
                y0 = row * self.FRAME_SIZE
                x0 = col * self.FRAME_SIZE
                frame_area = contour_mask[y0:y0+self.FRAME_SIZE, x0:x0+self.FRAME_SIZE]
                if np.sum(frame_area == 0) > 10:  # píxeles negros
                    self.good_frames_idx.add(idx)
                    color = (255, 0, 0, 255)  # rojo si es bueno
                else:
                    color = (0, 0, 255, 255)  # azul si es malo
                square = Image.new("RGBA", (self.FRAME_SIZE, self.FRAME_SIZE), color)
                self.squares_image.paste(square, (x0, y0), square)
                idx += 1

    def get_window_image_and_save(self, with_grid=True):
        """
        Devuelve la imagen compuesta: fondo, emojis y cuadrícula de colores.
        La cuadrícula muestra el estado de cada frame.
        """
        # 1. Fondo
        base = self.background_image.copy().convert("RGBA")
        # 2. Emojis
        for (x1, y1, x2, y2), emoji_img in zip(self.emoji_bboxes, self.emoji_images):
            base.paste(emoji_img, (x1, y1), emoji_img.convert("RGBA"))
        # 3. Cuadrícula de frames (con colores)
        if with_grid:
            base = Image.alpha_composite(base, self.squares_image)
        return base

    def scratch_frame(self, idx):
        if self.frames_mask[idx] != -1:
            return 0, False  # Ya rascado
        if idx in self.good_frames_idx:
            self.frames_mask[idx] = 1
        else:
            self.frames_mask[idx] = 0
        self.scratched_count += 1
        game_done = all(x != -1 for x in self.frames_mask)
        numero_de_0s = self.frames_mask.count(0)
        numero_de_1s = self.frames_mask.count(1)
        recompensa_por_0s = -2 * numero_de_0s
        recompensa_por_1s = 3 * numero_de_1s
        recompensa_total = recompensa_por_0s + recompensa_por_1s
        return recompensa_total, game_done


env = Scratch_Game_Environment5_Streamlit(frame_size=40, scratching_area=(0, 0, 700, 350), background_path="utils/space.jpg")
# print("Frames totales:", env.total_squares)
# print("Frames buenos:", env.good_frames_idx)
# print("Estado inicial:", env.frames_mask)
frame_idx = 9
reward, done = env.scratch_frame(frame_idx)
print(f"Scratch frame {frame_idx} -> reward: {reward}, game done: {done}")
# print("Estado tras rascar:", env.frames_mask)
env.get_window_image_and_save(with_grid=True).show()
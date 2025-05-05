import os

# -----------------------------------------------------------------------------
# Функция для генерации твёрдой ванны (solid) с боковыми стенками и дном
# -----------------------------------------------------------------------------
def generate_solid_init(path, w=512, h=512,
                        wall_thickness=4,   # толщина боковых стенок (в ячейках)
                        floor_thickness=4   # толщина дна (в ячейках)
                       ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"S {w} {h}\n")
        for j in range(h):
            for i in range(w):
                if i < wall_thickness or i >= w - wall_thickness or j >= h - floor_thickness:
                    phi = -1.0
                else:
                    phi = 1.0
                f.write(f"{phi:.6f} ")
            f.write("\n")

# -----------------------------------------------------------------------------
# Функция для генерации прямоугольного распределения жидкости (rectangle)
# -----------------------------------------------------------------------------
def generate_fluid_rectangle_init(path, w=512, h=512,
                                  wall_thickness=4,
                                  floor_thickness=4,
                                  fill_width_frac=1.0,    # доля внутренней ширины
                                  fill_height_frac=0.4    # доля внутренней высоты
                                 ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    interior_width = w - 2 * wall_thickness
    interior_height = h - floor_thickness
    # x от стены до x_end
    x_end = wall_thickness + int(fill_width_frac * interior_width)
    # порог по y (от верха) для заполнения
    y_start = int((1.0 - fill_height_frac) * interior_height)

    with open(path, 'w') as f:
        f.write(f"F {w} {h}\n")
        for j in range(h):
            for i in range(w):
                in_x = (i >= wall_thickness and i < x_end)
                in_y = (j >= y_start and j < interior_height)
                phi = -1.0 if (in_x and in_y) else 1.0
                f.write(f"{phi:.6f} ")
            f.write("\n")

# -----------------------------------------------------------------------------
# Функция для генерации жидкости: прямоугольный слой + шарик воды сверху
# -----------------------------------------------------------------------------
def generate_fluid_with_ball_init(path, w=512, h=512,
                                  wall_thickness=4,
                                  floor_thickness=4,
                                  fill_width_frac=1.0,       # для прямоугольной части
                                  fill_height_frac=0.4,
                                  ball_center_frac=(0.5,0.2), # (по x, по y) в долях от интерьера
                                  ball_radius_frac=0.1       # радиус шарика
                                 ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    interior_width = w - 2 * wall_thickness
    interior_height = h - floor_thickness

    # Параметры прямоугольного слоя
    x_end = wall_thickness + int(fill_width_frac * interior_width)
    y_start = int((1.0 - fill_height_frac) * interior_height)

    # Параметры шарика
    cx = wall_thickness + ball_center_frac[0] * interior_width
    cy = ball_center_frac[1] * interior_height
    r = ball_radius_frac * min(interior_width, interior_height)

    with open(path, 'w') as f:
        f.write(f"F {w} {h}\n")
        for j in range(h):
            for i in range(w):
                # прямоугольник внизу
                in_rect = (i >= wall_thickness and i < x_end and j >= y_start and j < interior_height)
                # шарик
                dx = i - cx
                dy = j - cy
                in_ball = (dx*dx + dy*dy <= r*r)
                phi = -1.0 if (in_rect or in_ball) else 1.0
                f.write(f"{phi:.6f} ")
            f.write("\n")

# -----------------------------------------------------------------------------
# Запуск генерации тестов

w, h = 100, 100
    # Солид-ванна (один файл для всех тестов)
generate_solid_init('res/solid_init.txt', w=w, h=h,
                       wall_thickness=4, floor_thickness=4)

    # Тест 1: обычный прямоугольный слой воды внизу (40% высоты)
generate_fluid_rectangle_init('res/fluid_init_rect.txt', w=w, h=h,
                                  wall_thickness=4, floor_thickness=4,
                                  fill_width_frac=1.0, fill_height_frac=0.4)

    # Тест 2: узкий столб слева у дна (50% ширины, 0% высоты)
generate_fluid_rectangle_init('res/fluid_init_column.txt', w=w, h=h,
                                  wall_thickness=4, floor_thickness=4,
                                  fill_width_frac=0.5, fill_height_frac=0.5)

    # Тест 3: стол воды + шарик, падающий сверху
generate_fluid_with_ball_init('res/fluid_init_ball.txt', w=w, h=h,
                                  wall_thickness=4, floor_thickness=4,
                                  fill_width_frac=1.0, fill_height_frac=0.4,
                                  ball_center_frac=(0.5, 0.4), ball_radius_frac=0.1)
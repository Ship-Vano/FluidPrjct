import matplotlib.pyplot as plt
import numpy as np

def generate_combined_labels_simple(solid_path, fluid_path, output_path):
    # Читаем solid данные
    with open(solid_path, 'r') as f_solid:
        header_solid = f_solid.readline().strip().split()
        w = int(header_solid[1])
        h = int(header_solid[2])
        solid_data = []
        for line in f_solid:
            solid_data.extend(map(float, line.strip().split()))

    # Читаем fluid данные
    with open(fluid_path, 'r') as f_fluid:
        header_fluid = f_fluid.readline().strip().split()
        fluid_data = []
        for line in f_fluid:
            fluid_data.extend(map(float, line.strip().split()))

    # Генерируем буквенные метки
    with open(output_path, 'w') as f_out:
        f_out.write(f"{w} {h}\n")  # Первая строка: ширина и высота

        for j in range(h):
            for i in range(w):
                idx = j * w + i
                if solid_data[idx] < 0:
                    f_out.write("S ")
                elif fluid_data[idx] < 0:
                    f_out.write("F ")
                else:
                    f_out.write("A ")
            f_out.write("\n")  # Новая строка для каждого ряда


def plot_simple_labels(filepath):
    with open(filepath, 'r') as f:
        w, h = map(int, f.readline().split())
        data = []
        for line in f:
            symbols = line.strip().split()
            if symbols:
                data.append([0 if c == 'A' else 1 if c == 'F' else 2 for c in symbols])

    plt.imshow(data, cmap='viridis', vmin=0, vmax=2)
    plt.colorbar(ticks=[0, 1, 2], label='AIR (A), FLUID (F), SOLID (S)')
    plt.show()

# Пример использования:
generate_combined_labels_simple(
    'res/solid_init.txt',
    'res/fluid_init_ball.txt',
    'res/labels_simple.txt'
)
plot_simple_labels('res/labels_simple.txt')
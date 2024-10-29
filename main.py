import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

from PIL import Image


# 1. Виведення первинного кольорового зображення на екран
def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. Виведення на екран матриці значень яскравості зображення
def display_brightness_matrix(image):
    image_array = np.array(image)
    brightness_matrix=np.mean(image_array,axis=2)
    for row in brightness_matrix:
        row =[str(int(value)) for value in row]
        print(" ".join(row)+"\n")

# 3. Побудова гістограми яскравості кольорового зображення
def plot_color_histogram(image):
    color = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])

        # Перетворюємо гістограму на одномірний масив для побудови
        hist = hist.ravel()

        # Побудова стовпчастої гістограми для кожного кольору
        plt.bar(np.arange(256) + i * 0.2, hist, width=0.2, color=col, label=f'{col.upper()} Channel')

    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([-0.5, 255.5])
    plt.xticks(np.arange(0, 256, 25))  # Встановлення підписів на осі X
    plt.legend()
    plt.show()


# 4. Зміна кольоровості
# Бінаризація
def binarize_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

# Перехід до відтінків сірого
def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Негатив
def negative_image(image):
    return cv2.bitwise_not(image)

# Побудова гістограми для зображення в градаціях сірого
def plot_grayscale_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Перетворюємо гістограму на одномірний масив для побудови
    hist = hist.ravel()

    # Побудова стовпчастої гістограми
    plt.bar(range(256), hist, color='black', width=1.0)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.show()



# Меню вибору дій
def menu(image):
    while True:
        print("\nОберіть дію:")
        print("1 - Виведення первинного кольорового зображення")
        print("2 - Виведення матриці яскравості")
        print("3 - Побудова гістограми яскравості кольорового зображення")
        print("4 - Бінаризація зображення")
        print("5 - Перехід до відтінків сірого")
        print("6 - Негатив")
        print("7 - Побудова гістограми зображення в градаціях сірого")
        print("0 - Вихід")

        choice = input("Ваш вибір: ")

        if choice == '1':
            show_image(image, "Original Image")
        elif choice == '2':
            display_brightness_matrix(image)
        elif choice == '3':
            plot_color_histogram(image)
        elif choice == '4':
            binary_image = binarize_image(image)
            show_image(binary_image, "Binary Image")
        elif choice == '5':
            grayscale = grayscale_image(image)
            show_image(grayscale, "Grayscale Image")
        elif choice == '6':
            negative = negative_image(image)
            show_image(negative, "Negative Image")
        elif choice == '7':
            plot_grayscale_histogram(image)
        elif choice == '0':
            print("Вихід з програми...")
            break
        else:
            print("Неправильний вибір, спробуйте ще раз.")

# Основна функція для виконання завдань
def main():
    # Використання Tkinter для вибору файлу зображення
    root = Tk()
    root.withdraw()  # Приховати основне вікно Tkinter
    image_path = filedialog.askopenfilename(title="Оберіть зображення", filetypes=[("Зображення", "*.jpg *.jpeg *.png")])

    if image_path:
        image = cv2.imread(image_path)

        # Запуск меню вибору дій
        menu(image)
    else:
        print("Зображення не було обрано.")

if __name__ == '__main__':
    main()

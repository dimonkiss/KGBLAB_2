import cv2
import numpy as np


def weighted_median_filter(image, kernel_size=3, weights=None):
    if weights is None:
        # Якщо ваги не задані, використовуємо однакові ваги
        weights = np.ones((kernel_size, kernel_size))

    # Нормалізуємо ваги
    weights = weights / np.sum(weights)

    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    output_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Вирізаємо регіон для фільтрації
            region = padded_image[i:i + kernel_size, j:j + kernel_size]

            # Обчислюємо значення пікселів та їх ваги
            values = region.flatten()
            weight_values = weights.flatten()

            # Обчислюємо зважене медіанне значення
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weight_values[sorted_indices]

            cumulative_weights = np.cumsum(sorted_weights)
            median_index = np.searchsorted(cumulative_weights, 0.5)

            output_image[i, j] = sorted_values[median_index]

    return output_image


# Завантаження зображення
image_primary = image = cv2.imread("C:\\Users\\dklxw\\OneDrive\\Desktop\\download.jfif")
image = cv2.imread("C:\\Users\\dklxw\\OneDrive\\Desktop\\download.jfif", cv2.IMREAD_GRAYSCALE)

# Визначення ваг
weights = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]])

# Застосування зваженого медіанного фільтра
filtered_image = weighted_median_filter(image, kernel_size=3, weights=weights)

# Відображення зображень
cv2.imshow('Original Image', image_primary)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

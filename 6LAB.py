import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def save_image(data, filename):
    img = Image.fromarray(data.astype(np.uint8))
    img.save(filename)

def add_gaussian_noise(image, variance):
    noise = np.random.normal(0, np.sqrt(variance), image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image

def add_impulse_noise(image, probability):
    noisy_image = image.copy()
    num_impulses = int(probability * noisy_image.size // 3)  # divide by 3 for RGB
    x_coords = np.random.randint(0, noisy_image.shape[1], num_impulses)
    y_coords = np.random.randint(0, noisy_image.shape[0], num_impulses)

    for i in range(num_impulses):
        noisy_image[y_coords[i], x_coords[i]] = np.random.choice([0, 255], size=3)

    return noisy_image

def add_spatially_non_stationary_noise(image, brightness_variance):
    noisy_image = image.astype(np.float32).copy()
    h, w, _ = noisy_image.shape
    brightness_variance_map = np.random.normal(0, brightness_variance, (h, w))
    noisy_image += (brightness_variance_map[..., np.newaxis] * 255)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_spatially_non_stationary_noise_with_coords(image):
    noisy_image = image.astype(np.float32).copy()
    h, w, _ = noisy_image.shape

    # Create a variance map based on coordinates
    x_coords = np.linspace(0, 1, w)
    y_coords = np.linspace(0, 1, h)
    x, y = np.meshgrid(x_coords, y_coords)

    # Variance depends on coordinates (can be adjusted)
    brightness_variance_map = np.sqrt(x * y)  # For example, variance depends on x and y
    noise = np.random.normal(0, 1, image.shape) * brightness_variance_map[..., np.newaxis]

    noisy_image += noise * 255
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def generate_noisy_images(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    # Create noisy images
    noisy_images = []

    # 1. Additive noise (variance 0.01)
    noisy_images.append(add_gaussian_noise(np.array(image), 0.01))

    # 2. Additive noise (variance 0.1)
    noisy_images.append(add_gaussian_noise(np.array(image), 0.1))

    # 3. Impulse noise (first)
    noisy_images.append(add_impulse_noise(np.array(image), 0.02))

    # 4. Impulse noise (second)
    noisy_images.append(add_impulse_noise(np.array(image), 0.05))  # Can change probability

    # 5. Spatially non-stationary additive noise (depends on brightness)
    noisy_images.append(add_spatially_non_stationary_noise(np.array(image), 0.05))

    # 6. Spatially non-stationary additive noise (depends on brightness)
    noisy_images.append(add_spatially_non_stationary_noise(np.array(image), 0.1))

    # 7. Spatially non-stationary additive noise (variance - function of spatial coordinates)
    noisy_images.append(add_spatially_non_stationary_noise_with_coords(np.array(image)))

    # 8. Spatially non-stationary additive noise (variance - function of spatial coordinates, again for testing)
    noisy_images.append(add_spatially_non_stationary_noise_with_coords(np.array(image)))

    return noisy_images

def apply_filter(image, filter_type):
    if filter_type == 1:  # Linear filter 1
        kernel = np.ones((3, 3), np.float32) / 9
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 2:  # Linear filter 2
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 3:  # Linear filter 3
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 4:  # Non-linear filter 1 (median)
        return cv2.medianBlur(image, 3)
    elif filter_type == 5:  # Non-linear filter 2
        return cv2.medianBlur(image, 5)
    elif filter_type == 6:  # Non-linear filter 3
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif filter_type == 7:  # Non-linear filter 4
        return cv2.bilateralFilter(image, 5, 75, 75)
    elif filter_type == 8:  # Non-linear filter 5
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif filter_type == 9:  # Non-linear filter 6
        return cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
    elif filter_type == 10:  # Non-linear filter 7
        return cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)

    return image

def calculate_distortion(original, filtered):
    return np.sum(np.abs(original - filtered)) / original.size

def main():
    image_path = "C:\\Users\\dklxw\\OneDrive\\Desktop\\download.jfif"
    noisy_images = generate_noisy_images(image_path)

    # Store distortion results in a table
    results = []

    for i in range(len(noisy_images)):  # For each noisy image
        noisy_image = noisy_images[i].astype(np.uint8)
        original_image = np.array(Image.open(image_path).convert("RGB").resize((128, 128))).astype(np.uint8)

        for filter_choice in range(1, 11):  # All 10 filters
            filtered_image = apply_filter(noisy_image, filter_choice)
            distortion = calculate_distortion(original_image, filtered_image)

            results.append({
                "Зображення": f"{i + 1}",
                "Фільтр": filter_choice,
                "Спотворення": distortion
            })

    # Create DataFrame to display results
    results_df = pd.DataFrame(results)
    print(results_df)

    # Visualize distortion results graphically
    plt.figure(figsize=(12, 6))
    plt.title("Оцінка спотворення для кожного зашумленого зображення")
    plt.xticks(rotation=45)
    plt.bar(results_df["Зображення"] + " - " + results_df["Фільтр"].astype(str), results_df["Спотворення"])
    plt.xlabel("Зображення - Фільтр")
    plt.ylabel("Спотворення")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

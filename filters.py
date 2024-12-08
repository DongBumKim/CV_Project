
from PIL import Image
import random
import numpy as np
import cv2

# Define the probabilistic wrapper
class RandomApplyFilter:
    def __init__(self, filter_cls, p=0.5):
        """
        Wrapper to apply a filter with a given probability.

        Args:
            filter_cls: The filter class to apply (e.g., EdgePreservingFilter, SobelFilter).
            p: Probability of applying the filter.
        """
        self.filter = filter_cls
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.filter(img)  # Apply the filter
        return img  # Return the original image

# Define custom filters
class EdgePreservingFilter:
    def __call__(self, img):
        img_np = np.array(img)
        filtered_img = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
        return Image.fromarray(filtered_img)

class SobelFilter:
    def __call__(self, img):
        img_np = np.array(img)
        sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobel_x, sobel_y)
        sobel = np.uint8(sobel / sobel.max() * 255)  # Normalize to 0-255
        return Image.fromarray(sobel)

class GaussianBlurFilter:
    def __call__(self, img):
        img_np = np.array(img)
        blurred_img = cv2.GaussianBlur(img_np, (5, 5), 0)  # (5, 5) kernel size
        return Image.fromarray(blurred_img)


class HistogramEqualizationFilter:
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:  # RGB image
            for i in range(3):  # Apply histogram equalization to each channel
                img_np[:, :, i] = cv2.equalizeHist(img_np[:, :, i])
        else:  # Grayscale image
            img_np = cv2.equalizeHist(img_np)
        return Image.fromarray(img_np)


class AddNoiseFilter:
    def __call__(self, img):
        img_np = np.array(img)

        # Use a unique seed for each image (e.g., hash of image content or index)
        seed = hash(img.tobytes()) % (2**32 - 1)
        rng = np.random.default_rng(seed)  # Create a seeded random number generator

        mean = 0
        stddev = 10
        noise = rng.normal(mean, stddev, img_np.shape).astype(np.int16)  # Ensure unique noise per image
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)  # Add noise and clip to valid range
        return Image.fromarray(noisy_img)



class SharpenFilter:
    def __call__(self, img):
        img_np = np.array(img)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])  # Sharpening kernel
        sharpened_img = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(sharpened_img)

class TextureEnhancementFilter:
    def __call__(self, img):
        img_np = np.array(img)

        # Use a unique seed for each image (e.g., hash of image content or index)
        seed = hash(img.tobytes()) % (2**32 - 1)
        rng = np.random.default_rng(seed)  # Create a seeded random number generator

        texture = rng.integers(0, 50, img_np.shape, dtype=np.uint8)  # Generate deterministic random texture
        textured_img = cv2.addWeighted(img_np, 0.8, texture, 0.2, 0)
        return Image.fromarray(np.clip(textured_img, 0, 255))

# In terms of reducing the domain gap
class LowPassFilter:
    def __init__(self, filter_size=30):
        """
        Args:
            filter_size: Size of the low-pass filter mask. Larger size allows more frequencies, smaller size removes more details.
        """
        self.filter_size = filter_size

    def __call__(self, img):
        img_np = np.array(img.convert("L"))  # Convert to grayscale
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)

        # Define the low-pass filter mask
        rows, cols = img_np.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-self.filter_size:crow+self.filter_size, ccol-self.filter_size:ccol+self.filter_size] = 1

        # Apply the mask and inverse Fourier Transform
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.abs(np.fft.ifft2(f_ishift))

        # Normalize the result to 0-255 and convert back to RGB
        img_back = np.uint8(img_back / img_back.max() * 255)
        img_back_rgb = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_back_rgb)

class AdaptiveEdgePreservingFilter:
    def __init__(self, d=9, sigma_color=75, sigma_space=75, sobel_weight=0.3):
        """
        Args:
            d: Diameter of each pixel neighborhood for bilateral filter.
            sigma_color: Filter sigma in the color space.
            sigma_space: Filter sigma in the coordinate space.
            sobel_weight: Weight of the Sobel edge gradient when combined with the smoothed image.
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.sobel_weight = sobel_weight

    def __call__(self, img):
        img_np = np.array(img)  # Convert PIL image to NumPy array

        # Apply edge-preserving bilateral filter
        edge_preserved = cv2.bilateralFilter(img_np, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)

        # Apply Sobel edge detection
        gray = cv2.cvtColor(edge_preserved, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)  # Normalize to 0-255

        # Blend the edge-preserved image with the Sobel gradient
        adaptive_filtered = cv2.addWeighted(edge_preserved, 1 - self.sobel_weight,
                                            cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2RGB), self.sobel_weight, 0)
        # Normalize to 0-255 range and ensure output is uint8
        adaptive_filtered = np.clip(adaptive_filtered, 0, 255).astype(np.uint8)
        return Image.fromarray(adaptive_filtered)

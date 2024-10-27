import tkinter as tk
from tkinter import filedialog, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, feature, segmentation, color
from skimage.filters import sobel
from scipy import ndimage as ndi
from PIL import Image, ImageTk
import cv2
import numba as nb

"""
#fonction pour self.show_segment_by_edge
@nb.jit(nopython=True)
def compute_edges(image_gray, filtered_image):
    Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    My = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    for i in range(1, image_gray.shape[0] - 1):
        for j in range(1, image_gray.shape[1] - 1):
            Gx = np.sum(Mx * image_gray[i - 1:i + 2, j - 1:j + 2])
            Gy = np.sum(My * image_gray[i - 1:i + 2, j - 1:j + 2])
            filtered_image[i, j] = np.sqrt(Gx**2 + Gy**2)
"""

class ImageHistogramApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Segmentation et Histogram d'image")
        self.master.configure(bg="#f0f0f0")

        self.header_label = tk.Label(master, text="Segmentation et Histogram d'image", font=("Arial", 20, "bold"),
                                     bg="#333333", fg="#ffffff", pady=10)
        self.header_label.pack(fill="x")

        self.menu_bar = tk.Menu(master)

        # Menu Fichier
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Charger Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=master.quit)
        self.menu_bar.add_cascade(label="Fichier", menu=file_menu)

        # Menu Affichage
        display_menu = tk.Menu(self.menu_bar, tearoff=0)
        display_menu.add_command(label="Afficher Histogramme Couleur", command=self.show_histogram_color)
        display_menu.add_command(label="Afficher Histogramme Niveau De Gris", command=self.show_histogram_gray)
        self.menu_bar.add_cascade(label="Affichage", menu=display_menu)

        # Menu Segmentation
        segment_menu = tk.Menu(self.menu_bar, tearoff=0)

        # Sous-menu Par Seuil
        threshold_menu = tk.Menu(segment_menu, tearoff=0)
        threshold_menu.add_command(label="Moyenne", command=lambda: self.segment_by_threshold("Moyenne"))
        threshold_menu.add_command(label="Médiane", command=lambda: self.segment_by_threshold("Médiane"))
        threshold_menu.add_command(label="Otsu", command=lambda: self.segment_by_threshold("Otsu"))
        threshold_menu.add_command(label="Manuel", command=self.ask_manual_threshold)

        segment_menu.add_cascade(label="Par Seuil", menu=threshold_menu)
        segment_menu.add_command(label="Par Edge", command=self.segment_by_edge)
        segment_menu.add_command(label="Par Region (Watershed)", command=self.show_segment_by_watershed)
        
        self.menu_bar.add_cascade(label="Segmentation", menu=segment_menu)

        master.config(menu=self.menu_bar)

        self.image_label = tk.Label(master, bg="#f0f0f0")
        self.image_label.pack(pady=20)

        self.image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = io.imread(file_path)
            self.show_image(self.image)

    def show_image(self, image):
        image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        img_tk = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(img_tk)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def show_histogram_gray(self):
        if self.image is not None:
            # Convertir l'image en niveaux de gris
            image_gray = np.mean(self.image, axis=2, dtype=np.uint8)
            hist_gray, _ = exposure.histogram(image_gray)

            plt.figure(figsize=(8, 6))
            plt.plot(hist_gray, color='#666666', alpha=0.7)
            plt.title('Histogramme de l\'Image en Niveaux de Gris', color='#333333')
            plt.xlabel('Niveaux de Gris')
            plt.ylabel('Nombre de Pixels')
            plt.tight_layout()
            plt.show()

    def show_histogram_color(self):
        if self.image is not None:
            # Calcul de l'histogramme en couleur (RGB)
            hist_r, _ = exposure.histogram(self.image[:, :, 0])
            hist_g, _ = exposure.histogram(self.image[:, :, 1])
            hist_b, _ = exposure.histogram(self.image[:, :, 2])

            plt.figure(figsize=(8, 6))
            plt.plot(hist_r, color='#ff4d4d', alpha=0.7, label='Rouge')
            plt.plot(hist_g, color='#66ff66', alpha=0.7, label='Vert')
            plt.plot(hist_b, color='#4da6ff', alpha=0.7, label='Bleu')
            plt.title('Histogramme des Canaux RGB', color='#333333')
            plt.xlabel('Intensité')
            plt.ylabel('Nombre de Pixels')
            plt.legend()
            plt.tight_layout()
            plt.show()

    def segment_by_threshold(self, method):
        if self.image is not None:
            image_gray = np.mean(self.image, axis=2, dtype=np.uint8)

            if method == "Moyenne":
                threshold = np.mean(image_gray)
            elif method == "Médiane":
                threshold = np.median(image_gray)
            elif method == "Otsu":
                threshold = filters.threshold_otsu(image_gray)

            segmented_image = image_gray > threshold

            plt.imshow(segmented_image, cmap=plt.cm.gray)
            plt.title(f'Image Segmentée (Seuillage {method})', color='#333333')
            plt.axis('off')
            plt.show()

    def ask_manual_threshold(self):
        if self.image is not None:
            threshold = simpledialog.askinteger("Seuillage Manuel", "Entrez le seuil :")
            if threshold is not None:
                image_gray = np.mean(self.image, axis=2, dtype=np.uint8)
                segmented_image = image_gray > threshold

                plt.imshow(segmented_image, cmap=plt.cm.gray)
                plt.title(f'Image Segmentée (Seuillage Manuel)', color='#333333')
                plt.axis('off')
                plt.show()

    """
    def show_segment_by_edge(self):
        if self.image is not None:
            # Convertir l'image en niveaux de gris
            image_gray = np.mean(self.image, axis=2, dtype=np.uint8)
            filtered_image = np.zeros_like(image_gray)

            compute_edges(image_gray, filtered_image)

            threshold = 100
            segmented_image = np.zeros_like(image_gray)
            segmented_image[filtered_image > threshold] = 255

            plt.imshow(segmented_image, cmap=plt.cm.gray)
            plt.title('Image Segmentée (Détection de Contours)', color='#333333')
            plt.axis('off')
            plt.show()
    """
    def segment_by_edge(self):
        if self.image is not None:
            # Convertir l'image en niveaux de gris
            image_gray = np.mean(self.image, axis=2, dtype=np.uint8)

            # Détection des contours avec la méthode Canny
            edges = feature.canny(image_gray)

            plt.imshow(edges, cmap=plt.cm.gray)
            plt.title('Image Segmentée (Détection de Contours)', color='#333333')
            plt.axis('off')
            plt.show()
            
    def show_segment_by_watershed(self):
        if self.image is not None:
            # Convertir l'image en RGB si elle n'est pas déjà en RGB
            if self.image.ndim == 2:
                self.image = np.stack((self.image,) * 3, axis=-1)
            elif self.image.shape[2] == 4:
                self.image = self.image[:, :, :3]  # Garder seulement les trois premiers canaux RGB

            # Convertir l'image en niveaux de gris
            image_gray = np.mean(self.image, axis=2, dtype=np.uint8)

            # Calculer la carte d'élévation avec la méthode Sobel
            elevation_map = sobel(image_gray)

            # Appliquer un seuil pour segmenter en régions
            markers = np.zeros_like(image_gray)
            markers[image_gray < 30] = 1
            markers[image_gray > 150] = 2

            segmentation_coins = segmentation.watershed(elevation_map, markers)

            # Remplir les trous et labeliser les régions
            segmentation_coins = ndi.binary_fill_holes(segmentation_coins - 1)
            labeled_coins, _ = ndi.label(segmentation_coins)

            # Utiliser label2rgb avec l'image convertie en RGB
            image_label_overlay = color.label2rgb(labeled_coins, image=self.image, bg_label=0)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(elevation_map, cmap=plt.cm.gray)
            axes[0].set_title('Carte d\'Élévation', color='#333333')
            axes[0].axis('off')

            axes[1].imshow(image_label_overlay)
            axes[1].set_title('Image Segmentée par Watershed', color='#333333')
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()


def main():
    root = tk.Tk()
    app = ImageHistogramApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

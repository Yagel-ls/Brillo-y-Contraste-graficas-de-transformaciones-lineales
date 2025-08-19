import cv2
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# 1. Cargar la imagen y manejar errores
# ======================================================================
try:
    img = cv2.imread('input.jpg')
    if img is None:
        raise FileNotFoundError("No se pudo cargar la imagen. Asegúrate de que 'input.jpg' exista en el mismo directorio.")
except FileNotFoundError as e:
    print(e)
    exit()

# ======================================================================
# 2. Ajuste de Brillo y Contraste por separado
# ======================================================================
# Parámetros
alpha_contraste = 1.8  # Solo contraste
beta_brillo = 50     # Solo brillo

# Imagen ajustada solo con contraste
img_contraste = cv2.convertScaleAbs(img, alpha=alpha_contraste, beta=0)

# Imagen ajustada solo con brillo
img_brillo = cv2.convertScaleAbs(img, alpha=1.0, beta=beta_brillo)

# Preparar las imágenes para mostrarlas con Matplotlib (BGR a RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
img_contraste_rgb = cv2.cvtColor(img_contraste, cv2.COLOR_BGR2RGB) if len(img_contraste.shape) == 3 else img_contraste
img_brillo_rgb = cv2.cvtColor(img_brillo, cv2.COLOR_BGR2RGB) if len(img_brillo.shape) == 3 else img_brillo

# ======================================================================
# 3. Gráficas de las transformaciones (Lineal)
# ======================================================================
x_original = np.arange(256)

# Gráfica de Transformación de Brillo
y_brillo = np.clip(1.0 * x_original + beta_brillo, 0, 255)

# Gráfica de Transformación de Contraste
y_contraste = np.clip(alpha_contraste * x_original + 0, 0, 255)

# ======================================================================
# 4. Visualizar todo en una sola ventana con espaciado mejorado
# ======================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 14)) # Aumento del tamaño de la figura
fig.suptitle('Análisis de Brillo y Contraste', fontsize=18)

# Subgráfico A: Imagen Original
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('A) Imagen Original')
axes[0, 0].axis('off')

# Subgráfico B: Imagen con Brillo Ajustado
axes[0, 1].imshow(img_brillo_rgb)
axes[0, 1].set_title(f'B) Imagen con Brillo (+{beta_brillo})')
axes[0, 1].axis('off')

# Subgráfico C: Imagen con Contraste Ajustado
axes[1, 0].imshow(img_contraste_rgb)
axes[1, 0].set_title(f'C) Imagen con Contraste (x{alpha_contraste})')
axes[1, 0].axis('off')

# Subgráfico D: Gráfica de Transformaciones Lineales
axes[1, 1].plot(x_original, y_brillo, label=f'Brillo (β={beta_brillo})', color='green')
axes[1, 1].plot(x_original, y_contraste, label=f'Contraste (α={alpha_contraste})', color='blue')
axes[1, 1].plot(x_original, x_original, '--', label='Identidad (Sin cambio)', color='gray')
axes[1, 1].set_title('D) Gráficas de Transformaciones Lineales')
axes[1, 1].set_xlabel('Valor de píxel de entrada')
axes[1, 1].set_ylabel('Valor de píxel de salida')
axes[1, 1].grid(True)
axes[1, 1].legend()

# Ajuste automático de los subplots con relleno extra
plt.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])
plt.show()
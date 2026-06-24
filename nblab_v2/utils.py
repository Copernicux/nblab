import os
import ast
import json
import argparse
from datetime import datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

def str2bool(val):
    if isinstance(val, bool): return val
    return str(val).lower() in ("yes", "true", "t", "y", "1")

def get_args():
    parsed_args = get_parsed_args()
    args = {arg: getattr(parsed_args, arg) for arg in vars(parsed_args)}
    for key, value in args.items():
        # FORZAMOS ESCÁNER BOOLEANO REAL:
        if key in ["save", "plot", "save_gif", "live_anim"]:
            args[key] = str2bool(value)
        else:
            try: args[key] = ast.literal_eval(value)
            except: continue
    return args

def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-beam", default="beams/beam_v1_256.mat")
    parser.add_argument("-resolution", default=256)
    parser.add_argument("-window_size", default=20.0)
    parser.add_argument("-loss_sigma", default=100.0)
    
    parser.add_argument("-save", default=True)
    parser.add_argument("-plot", default=True)
    parser.add_argument("-save_gif", default=True)
    parser.add_argument("-live_anim", default=False)
    parser.add_argument("-anim_step", default=20)

    parser.add_argument("-N", default=40)
    parser.add_argument("-epochs", default=1200)
    parser.add_argument("-learning_rate", default=0.01)

    # --- NUEVOS HIPERPARÁMETROS DE REGULARIZACIÓN (Vías 1 y 2) ---
    parser.add_argument("-gamma_fourier", default=0.05) # Peso de la pérdida en el espacio k
    parser.add_argument("-lambda_sparse", default=0.001) # Fuerza del "silenciador" de modos
    parser.add_argument("-lambda_tv", default=0.001)     # Fuerza del "alisador" de frentes

    return parser.parse_args()




def load_beam_target(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo no existe: {filepath}")
    
    mat_data = sio.loadmat(filepath)
    if 'Uo' not in mat_data:
        raise KeyError(f"Falta la llave 'Uo' en {filepath}")
    
    matrix = mat_data['Uo']
    if np.iscomplexobj(matrix):
        return np.abs(matrix)**2
    return matrix.astype(np.float32)

def cartesian_to_polar(x, y):
    alpha = np.sqrt(x**2 + y**2)
    beta = np.arctan2(y, x)
    beta = np.where(beta < 0, beta + 2.0 * np.pi, beta)
    return alpha, beta


def save_evolution_gif(target_I, frames_list, output_path, step=20, fps=15):
    print("Compilando animación GIF de alta simetría...")
    I_max = float(np.max(target_I))
    if I_max == 0: I_max = 1.0 # Protección anti-división por cero

    # Forzamos una figura ancha (10 x 4.8 pulgadas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))
    
    # PANEL IZQUIERDO (Objetivo estático)
    im1 = ax1.imshow(target_I, cmap='hot', origin='lower', vmin=0.0, vmax=I_max, aspect='equal')
    ax1.set_title("Objetivo: |Ψ_obj|²", fontsize=12, pad=12)
    ax1.axis('off')
    
    # PANEL DERECHO (Reconstrucción dinámica)
    im2 = ax2.imshow(frames_list[0], cmap='hot', origin='lower', vmin=0.0, vmax=I_max, aspect='equal')
    title_dyn = ax2.set_title("Reconstrucción | Época: 0", fontsize=12, pad=12)
    ax2.axis('off')
    
    # BARRA DE COLOR ANCLADA ESTABLEMENTE AL PANEL DERECHO
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Intensidad absoluta |Ψ|²", rotation=270, labelpad=15)

    plt.tight_layout()

    def update(idx):
        im2.set_data(frames_list[idx])
        title_dyn.set_text(f"Reconstrucción | Época: {idx * step}")
        return [im2, title_dyn]

    ani = FuncAnimation(fig, update, frames=len(frames_list), blit=False)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_experiment(alpha, beta, k_t, loss_history, target_I, pred_I, pred_phase, frames_buffer, args, source_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = os.path.splitext(os.path.basename(source_name))[0]
    output_dir = os.path.join("outputs", f"{timestamp}_{clean_name}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. JSON
    data_export = {
        "metadata": {
            "source_file": source_name,
            "timestamp": timestamp,
            "N_modes": int(args["N"]),
            "loss_sigma": float(args["loss_sigma"]),
            "learning_rate": float(args["learning_rate"]),
            "gamma_fourier": float(args["gamma_fourier"]),
            "lambda_sparse": float(args["lambda_sparse"]),
            "lambda_tv": float(args["lambda_tv"]),
            "k_t_inferred": float(k_t),
            "final_apodized_loss": float(loss_history[-1]),
            "resolution": int(target_I.shape[0]),
            "window_size": float(args["window_size"])
        },
        "coefficients": [{"n": int(i+1), "alpha": float(alpha[i]), "beta_rad": float(beta[i])} for i in range(len(alpha))]
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(data_export, f, indent=4)

    # 2. Resumen PNG
    plot_paper_summary(target_I, pred_I, pred_phase, alpha, beta, loss_history, k_t, save_path=os.path.join(output_dir, "paper_summary.png"), show=args.get("plot", True))

    # 3. SECCIÓN GIF INTERROGADA
    print(f"\n[RASTREO UTILS] Entrando a bloque GIF. Frames recibidos: {len(frames_buffer)}")
    
    if len(frames_buffer) > 0:
        gif_path = os.path.join(output_dir, "evolution.gif")
        print(f"[RASTREO UTILS] Ruta de destino calculada: {gif_path}")
        
        try:
            save_evolution_gif(target_I, frames_buffer, gif_path, step=int(args["anim_step"]))
            print("[RASTREO UTILS] ¡ÉXITO! El motor de Matplotlib/Pillow terminó de escribir el archivo.")
        except Exception as e:
            print(f"\n[!!!] EXCEPCIÓN FANTASMA CAPTURADA EN PILLOW [!!!]")
            print(f"Clase de error : {type(e).__name__}")
            print(f"Mensaje exacto : {e}\n")
    else:
        print("[RASTREO UTILS] ERROR: Los fotogramas llegaron con longitud cero a utils.py.")
    return output_dir


def plot_paper_summary(target_intensity, pred_intensity, pred_phase, alpha, beta, loss_history, k_t, save_path=None, show=True):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0, 0].imshow(target_intensity, cmap='hot', origin='lower')
    axs[0, 0].set_title("Objetivo: |Ψ_obj|²")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(pred_intensity, cmap='hot', origin='lower')
    axs[0, 1].set_title(f"Reconstrucción: |Ψ|² (k_t={k_t:.2f})")
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(pred_phase, cmap='twilight', origin='lower')
    axs[0, 2].set_title("Fase Espacial: arg(Ψ)")
    axs[0, 2].axis('off')
    
    n_indices = np.arange(1, len(alpha) + 1)
    axs[1, 0].plot(n_indices, alpha, 'k-o', markersize=3)
    axs[1, 0].set_title("Espectro Amplitud (α_n)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    axs[1, 1].plot(n_indices, beta, 'k-o', markersize=3)
    axs[1, 1].set_title("Espectro Fase (β_n)")
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    axs[1, 2].plot(np.log10(loss_history), 'k-')
    axs[1, 2].set_title("Convergencia: log10(γ)")
    axs[1, 2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    if show: plt.show()
    plt.close()
import os
import cv2
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

class NondiffractingInstancePINN:
    def __init__(self, target_intensity_matrix, args):
        self.args = args
        self.target_intensity = tf.constant(target_intensity_matrix, dtype=tf.float32)
        self.N = int(args["N"])
        self.res = int(target_intensity_matrix.shape[0]) 
        
        # Pesos de las regularizaciones
        self.gamma_fourier = float(args["gamma_fourier"])
        self.lambda_sparse = float(args["lambda_sparse"])
        self.lambda_tv = float(args["lambda_tv"])
        
        self.x_coeffs = tf.Variable(tf.random.normal([self.N], mean=0.0, stddev=0.1), dtype=tf.float32)
        self.y_coeffs = tf.Variable(tf.random.normal([self.N], mean=0.0, stddev=0.1), dtype=tf.float32)
        self.k_t = tf.Variable(4.0, dtype=tf.float32, trainable=True)
        self.trainable_variables = [self.x_coeffs, self.y_coeffs, self.k_t]
        
        w_size = float(args["window_size"])
        line = tf.linspace(-w_size / 2.0, w_size / 2.0, self.res)
        X, Y = tf.meshgrid(line, line)
        self.r = tf.sqrt(X**2 + Y**2)
        self.theta = tf.math.atan2(Y, X)
        
        n_indices = tf.range(1, self.N + 1, dtype=tf.float32)
        self.phi_n = (2.0 * np.pi * n_indices) / float(self.N)
        
        sigma = float(args["loss_sigma"])
        self.w_r = (1.0 / sigma) * tf.sqrt(2.0 / np.pi) * tf.exp(-(self.r**2) / (2.0 * sigma**2))
        
        # --- PRE-CÁLCULO VÍA 2: Espectro de Fourier Objetivo Normalizado ---
        fft_t = tf.signal.fft2d(tf.cast(self.target_intensity, tf.complex64))
        mag_t = tf.abs(fft_t)
        self.target_fft_norm = mag_t / (tf.reduce_max(mag_t) + 1e-8)
        
        # --- NUEVO: TRANSMISOR DE TENSORBOARD ---
        log_dir = os.path.join("outputs", "tb_live", datetime.datetime.now().strftime("%H%M%S"))
        self.tb_writer = tf.summary.create_file_writer(log_dir)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=float(args["learning_rate"]))
        self.loss_history = []
        self.frames_buffer = []

    def compute_complex_field(self):
        delta_angle = self.phi_n[None, None, :] - self.theta[:, :, None]
        phase_argument = self.k_t * self.r[:, :, None] * tf.cos(delta_angle)
        plane_waves = tf.complex(tf.cos(phase_argument), tf.sin(phase_argument))
        c_n = tf.complex(self.x_coeffs, self.y_coeffs)
        return tf.reduce_sum(plane_waves * c_n[None, None, :], axis=-1)

    def coarse_kt_sweep(self, kt_min=2, kt_max=12.0, steps=150):
        print("\n[!] Realizando barrido grueso de frecuencia k_t...")
        best_kt = float('inf')
        min_loss = float('inf')
        test_values = np.linspace(kt_min, kt_max, steps)
        
        backup_x, backup_y = self.x_coeffs.numpy(), self.y_coeffs.numpy()
        self.x_coeffs.assign(tf.ones([self.N]) * 0.1)
        self.y_coeffs.assign(tf.zeros([self.N]))
        
        for val in test_values:
            self.k_t.assign(val)
            psi = self.compute_complex_field()
            I_test = tf.math.real(psi)**2 + tf.math.imag(psi)**2
            err = tf.reduce_sum(self.w_r * tf.abs(I_test - self.target_intensity)).numpy()
            if err < min_loss:
                min_loss, best_kt = err, val
                
        self.x_coeffs.assign(backup_x)
        self.y_coeffs.assign(backup_y)
        self.k_t.assign(best_kt)
        print(f"[✓] k_t anclado en: {best_kt:.2f} (Loss base: {min_loss:.1f})")

    @tf.function
    def optimization_step(self):
        with tf.GradientTape() as tape:
            psi_field = self.compute_complex_field()
            pred_intensity = tf.math.real(psi_field)**2 + tf.math.imag(psi_field)**2
            
            # 1. Pérdida Espacial Apodizada (Paper original)
            error_espacial = tf.abs(pred_intensity - self.target_intensity)
            loss_spatial = tf.reduce_sum(self.w_r * error_espacial)
            
            # 2. Pérdida Vía 2: Dominio de Fourier (k-space)
            fft_pred = tf.signal.fft2d(tf.cast(pred_intensity, tf.complex64))
            mag_pred = tf.abs(fft_pred)
            mag_pred_norm = mag_pred / (tf.reduce_max(mag_pred) + 1e-8)
            loss_fourier = tf.reduce_mean(tf.abs(mag_pred_norm - self.target_fft_norm))
            
            # 3. Pérdidas Vía 1: Sparsity (L1) y Total Variation Circular (TV)
            alpha_n = tf.sqrt(self.x_coeffs**2 + self.y_coeffs**2 + 1e-8)
            loss_sparse = tf.reduce_mean(alpha_n)
            
            alpha_roll = tf.roll(alpha_n, shift=-1, axis=0)
            loss_tv = tf.reduce_mean(tf.abs(alpha_roll - alpha_n))
            
            # --- GRAN ENSAMBLE GENERAL ---
            loss_total = (
                loss_spatial + 
                (self.gamma_fourier * loss_fourier) + 
                (self.lambda_sparse * loss_sparse) + 
                (self.lambda_tv * loss_tv)
            )
            
        grads = tape.gradient(loss_total, self.trainable_variables)
        
        # Esteroide al gradiente de k_t para evitar bloqueo espectral
        grads_list = list(grads)
        grads_list[2] = grads_list[2] * 15.0 
        
        self.optimizer.apply_gradients(zip(grads_list, self.trainable_variables))
        self.k_t.assign(tf.maximum(self.k_t, 0.2))
        
        # Devolvemos el diccionario de autopsia
        desglose = {
            "total": loss_total, 
            "spatial": loss_spatial, 
            "fourier": loss_fourier, 
            "sparse": loss_sparse, 
            "tv": loss_tv
        }
        return desglose, pred_intensity, psi_field

    def fit(self):
        self.coarse_kt_sweep()
        epochs = int(self.args["epochs"])
        step = int(self.args["anim_step"])
        
        do_live = bool(self.args["live_anim"])
        do_gif = bool(self.args["save_gif"])

        # Pre-empaquetamos el objetivo en formato monitor (0-255 uint8) para el Live
        if do_live:
            target_np = self.target_intensity.numpy()
            I_max_global = float(np.max(target_np))
            if I_max_global == 0: I_max_global = 1.0
            
            target_u8 = np.clip((target_np / I_max_global) * 255.0, 0, 255).astype(np.uint8)
            target_hot = cv2.applyColorMap(target_u8, cv2.COLORMAP_HOT)
            # Barra vertical gris divisoria de 2 píxeles de ancho
            linea_divisoria = np.full((self.res, 2, 3), 100, dtype=np.uint8)

        for epoch in range(epochs):
            losses, pred_I, psi_field = self.optimization_step()
            self.loss_history.append(losses["spatial"].numpy()) 
            
            # --- 1. TRANSMISIÓN OPENCV EN VIVO (DOBLE PANEL WIDESCREEN) ---
            if do_live and (epoch % step == 0 or epoch == epochs - 1):
                pred_np = pred_I.numpy()
                pred_u8 = np.clip((pred_np / I_max_global) * 255.0, 0, 255).astype(np.uint8)
                pred_hot = cv2.applyColorMap(pred_u8, cv2.COLORMAP_HOT)
                
                # Pegamos: [ Objetivo | Linea | Simulación ]
                panel_doble = np.hstack([target_hot, linea_divisoria, pred_hot])
                
                # Estiramos a resolución de televisor (1024 x 512 px)
                panel_doble_big = cv2.resize(panel_doble, (1024, 512), interpolation=cv2.INTER_NEAREST)
                
                # Textos de telemetría superpuestos en blanco
                cv2.putText(panel_doble_big, "OBJETIVO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(panel_doble_big, f"EPOCA: {epoch:04d} | kt: {self.k_t.numpy():.2f}", (512 + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                
                cv2.imshow(f"NBLAB SALA DE CONTROL -> {self.args['beam']}", panel_doble_big)
                cv2.waitKey(1)

            # --- 2. RECOLECTOR PARA EL GIF FINAL ---
            if do_gif and (epoch % step == 0 or epoch == epochs - 1):
                self.frames_buffer.append(pred_I.numpy().copy())

            if epoch % 100 == 0 or epoch == epochs - 1:
                t_tot, t_sp = losses['total'].numpy(), losses['spatial'].numpy()
                print(f"Época {epoch:04d} | Tot: {t_tot:.2f} (Sp:{t_sp:.2f}) | kt: {self.k_t.numpy():.2f}")
        
        if do_live: cv2.destroyAllWindows()

        final_I = pred_I.numpy()
        final_phase = tf.math.atan2(tf.math.imag(psi_field), tf.math.real(psi_field)).numpy()
        alpha, beta = utils.cartesian_to_polar(self.x_coeffs.numpy(), self.y_coeffs.numpy())
        
        return final_I, final_phase, alpha, beta, self.loss_history, self.frames_buffer

if __name__ == "__main__":
    args = utils.get_args()
    beam_input = args["beam"]

    print(f"\n--- 1. Cargando objetivo: '{beam_input}' ---")
    target_matrix = utils.load_beam_target(beam_input) if beam_input.endswith(".mat") else np.ones((256,256))

    print("\n--- 2. Entrenando PINN Híbrido ---")
    pinn = NondiffractingInstancePINN(target_matrix, args)
    pred_I, pred_phase, alpha, beta, losses, frames = pinn.fit()

    if args["save"]:
        utils.save_experiment(alpha, beta, pinn.k_t.numpy(), losses, target_matrix, pred_I, pred_phase, frames, args, beam_input)
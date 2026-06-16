
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('tkAgg')
from scipy.optimize import differential_evolution
import numpy as np 
import pandas as pd 
import utils as ut 
import time 
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
import tensorflow_probability as tfp
# from Models.NLBModelConvNN import NLBModelConvNN, CustomFit
# from Models.ConvNN2 import ConvNN as NLBModel
# from Models.ConvNN2 import CustomFit
from Models.nn_modelv2 import NLBModel, CustomFit
# from Losses.CustomGaussianLoss import CustomGaussianLoss

class NBLab:

    def __init__(self, beam, args):
        self.beam = beam
        self.args = args
        self.model = None
        self.epochs = self.args["epochs"]
        self.batch_size = self.args["batch_size"]
        self.num_samples = self.args["num_samples"]
        self.window_size = self.args["window_size"]
        self.resolution = self.args["resolution"]
        self.problem_size = self.args["problem_size"]
        
    def get_data(self):
        X = np.array([np.array([self.beam[0]]) for i in range(self.num_samples)]).reshape(self.num_samples, self.resolution, self.resolution, 1)
        # Rotate arbitrarily some samples
        # for i in range(self.num_samples//2):
        #    X[i] = np.rot90(X[i], k=1, axes=(0,1))
        #    X[i+self.num_samples//2] = np.rot90(X[i+self.num_samples//2], k=2, axes=(0,1))
        # X = np.array([self.beam[0] for i in range(self.num_samples)]).reshape(self.num_samples, self.resolution, self.resolution, 1)
        y = np.array([self.beam[1] for i in range(self.num_samples)]).reshape(self.num_samples, self.resolution, self.resolution, 1)
        return {"X" : X, "y" : y}

    def generate_random_alpha_beta(self):
        N = self.problem_size
        return np.random.random((N,2))

    def train_model(self, data, callbacks=None):
        X, y = data["X"], data["y"]
        
        # --- CAMBIO DE OPTIMIZADOR ---
        opt = tf.keras.optimizers.Adam(learning_rate=self.args["learning_rate"])
        
        model = NLBModel(self.args)
        training = CustomFit(model)
        training.compile(optimizer=opt, loss=self.CustomGaussianLoss)
        
        if callbacks is None:
            callbacks = []
            
        history = training.fit(
            X, y, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=False,
            callbacks=callbacks
        )
        self.model = training
        return training, history
    
    def custom_gaussian_loss_for_optimization(self, alpha_beta):
        # Separar y reformatear alpha y beta
        N = self.problem_size
        alpha, beta = np.split(alpha_beta, 2)
        alpha = alpha.reshape((1, N))  # Reformateando para que sea un tensor 2D
        beta = beta.reshape((1, N))    # Reformateando para que sea un tensor 2D

        # Reconstruir el haz de luz utilizando alpha y beta
        real_pred, imag_pred = self.get_pred_beam(alpha, beta)
        int_pred = np.abs((np.square(real_pred) + np.square(imag_pred)))
        # Usar los datos de entrenamiento o un objetivo definido para la comparación
        int_true = self.beam[1].reshape(1, *self.beam[1].shape)  # Asumiendo que 'beam[1]' contiene el objetivo

        # Calcular la pérdida utilizando CustomGaussianLoss
        loss = np.sum(np.sum(self.gaussian_mask() * np.abs(int_true - int_pred)))
        print("Pérdida calculada:", loss)
        return loss

    
    def genetic_optimization(self, initial_alpha_beta, bounds, maxiter=2):
        # Utiliza el algoritmo de evolución diferencial para encontrar los parámetros óptimos
        result = differential_evolution(
        self.custom_gaussian_loss_for_optimization, 
        bounds, 
        popsize=3, 
        maxiter=2, 
        mutation=(0.0),  # Rango de factores de mutación
        recombination=0.9,  # Probabilidad de recombinación
        disp=True
    )
        return result.x

    def search(self, model):
        # Obtener valores iniciales de alpha y beta de la predicción del modelo
        y_pred = model.predict(np.array([self.beam[0].reshape(self.resolution, self.resolution, 1)]))
        alpha_init, beta_init = y_pred[0,:y_pred.shape[1]//2], y_pred[0,y_pred.shape[1]//2:]

        # Combinar alpha y beta en un solo array para la optimización
        initial_alpha_beta = np.concatenate([alpha_init, beta_init])

        # # Establecer límites estrechos alrededor de los valores iniciales
        # tolerance = 0.1  # Esto define qué tan estrecho es el rango alrededor de los valores iniciales
        # bounds = [(val - tolerance, val + tolerance) for val in initial_alpha_beta]

        # # Aplicar algoritmo genético para optimizar alpha y beta
        # optimized_alpha_beta = self.genetic_optimization(initial_alpha_beta, bounds)

        # # Separar alpha y beta optimizados
        # alpha_opt, beta_opt = np.split(optimized_alpha_beta, 2)

        return alpha_init, beta_init


    def pred_beam(self, model, sample):
        return model.predict(sample)[0,:,:,:]

    def get_sample(self):
        sample = np.array([ self.generate_nb_mode(n) for n in range(self.problem_size)])
        return np.array([sample] * 2).reshape(2, self.resolution, self.resolution, self.problem_size, 2, 2)
    

    def plot_beam(self, beam):
        beam = beam[0] 
        # beam = beam[:,:,0] 
        intensity = np.sqrt(np.abs(beam))
        plt.imshow(intensity)
        plt.show()

    def plot_alpha_beta(self, alpha, beta):
        plt.plot(alpha)
        plt.ylabel(r'$\alpha_n$', fontsize=25, fontweight='bold', family='serif')
        plt.xlabel("N", fontsize=25, fontweight='bold', family='serif')
        plt.show() 

        plt.plot(beta)
        plt.ylabel(r'$\beta_n$',fontsize=25, fontweight='bold', family='serif')

        plt.xlabel("N", fontsize=25, fontweight='bold', family='serif')
        plt.show() 
        

    def plot_compare_beams(self, beam_a, beam_b):

        beam_a = beam_a[:,:,0] #+ 1j*beam_a[:,:,1]
        beam_b = beam_b[:,:,0] #+ 1j*beam_b[:,:,1]
        intensity_a = np.sqrt(np.abs(beam_a))
        # Rotamos solo la mitad inferior 180 grados en sentido horario
        intensity_b = np.sqrt(np.abs(beam_b))
        res = intensity_a - intensity_b
        print(np.sum(np.sum(np.abs(res))))
        plt.subplot(1, 2, 1)
        plt.imshow(intensity_a, cmap='hot', aspect='auto')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(intensity_b, cmap='hot', aspect='auto')
        plt.colorbar()
        plt.show()

    def plot_beam_phase(self,beam):
        if beam.ndim != 2 or np.iscomplexobj(beam) is False:
            raise ValueError("El campo complejo debe ser una matriz 2D con valores complejos")
        
        phase = np.angle(beam)  # Calcula directamente la fase del campo complejo
        plt.imshow(phase, cmap='hot', aspect='auto')  # Usamos 'hsv' para un mejor contraste
        plt.colorbar()
        plt.title("Distribución de la Fase de un Campo Complejo")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()



    def plot_history(self, history):
        plt.plot(np.log(history.history["loss"]))
        plt.legend(["loss"])
        plt.xlabel('epochs',fontsize=15,)
        plt.ylabel(r'$ln(Loss)$',fontsize=15,)
        plt.show()
        

    def compose_beam(self, alpha, beta):
        N = self.problem_size
        xx, yy = self.create_cartesian_meshgrid()
        r, theta = self.cart2pol(xx, yy)
        phi = 2 * np.pi / N * np.array([mode for mode in range(1,N+1)]) 
        complex_pred_beam = np.zeros((self.resolution, self.resolution), dtype=complex)

        for a, b, p in zip(alpha, beta, phi):
            complex_pred_beam += a * np.exp(1j*b)* np.exp(1j * r * np.cos(p-theta))
        pred_beam = np.array([complex_pred_beam.real ** 2 + complex_pred_beam.imag ** 2]).reshape(self.resolution, self.resolution, 1)
        complex_pred_beam = np.array([complex_pred_beam.real + complex_pred_beam.imag * 1j]).reshape(self.resolution, self.resolution, 1)
        return pred_beam, complex_pred_beam
    
    
    def CustomGaussianLoss(self, y, y_pred):
        len_y_pred = y_pred.shape[1]
        alpha, beta = y_pred[:,:len_y_pred//2], y_pred[:,len_y_pred//2:]
        
        # 1. CORRECCIÓN FÍSICA: Forzar que la amplitud sea estrictamente positiva
        alpha = tf.math.abs(alpha)

        real_pred, imag_pred = self.get_pred_beam(alpha, beta)
        int_true = y[:,:,:,0] 

        int_pred = tf.math.abs((tf.math.square(real_pred) + tf.math.square(imag_pred)))

        # Pérdida principal de Intensidad (Ronda los ~4000 a ~5000)
        loss_intensity = K.sum(K.sum(self.gaussian_mask() * tf.math.abs(int_true - int_pred)))
        
        # 2. RE-ESCALAMIENTO DE CASTIGOS:
        # Si la intensidad es 4000, los castigos deben ser lo suficientemente grandes 
        # para que al gradiente le "duela" ignorarlos.
        
        penalty_factor = 100.0 # Mantiene la fase alrededor de ~180
        phase_collapse_penalty = K.sum(tf.math.exp(-tf.math.abs(beta)))
        phase_loss = penalty_factor * phase_collapse_penalty
        
        # Subimos el factor drásticamente de 0.1 a 5000.0 
        # L1_Alpha pasará de ~0.06 a ~300. Ahora la red SÍ apagará modos.
        sparsity_factor = 50000.0
        loss_sparsity = sparsity_factor * K.sum(tf.math.abs(alpha))
        
        # Subimos el factor de 0.005 a 200.0
        # TV pasará de ~1.5 a ~300. Obligará a que la imagen sea suave.
        tv_factor = 20000.0
        int_pred_expanded = tf.expand_dims(int_pred, axis=-1)
        loss_tv = tv_factor * K.sum(tf.image.total_variation(int_pred_expanded))
    
        # IMPRESIÓN DE DIAGNÓSTICO (Déjalo para vigilar la nueva pelea)
        # tf.print("\n[DEBUG] Int:", loss_intensity, " | Fase:", phase_loss, " | L1_Alpha:", loss_sparsity, " | TV:", loss_tv)
    
        # Sumamos todas las penalizaciones
        return loss_intensity + phase_loss 
    
    def CustomKLDLoss(self, y, y_pred):
        len_y_pred = y_pred.shape[1]
        alpha, beta = y_pred[:,:len_y_pred//2], y_pred[:,len_y_pred//2:]
        int_true = y[:,:,:,0] 

        real_pred, imag_pred = self.get_pred_beam(alpha, beta)
        int_pred = tf.math.abs((tf.math.square(real_pred) + tf.math.square(imag_pred)))
        loss = int_true * tf.math.log(int_true / int_pred) + (1 - int_true) * tf.math.log((1 - int_true) / (1 - int_pred))
        # make sure loss is not nan
        loss = tf.where(tf.math.is_nan(loss), tf.ones_like(loss), loss)
        loss = K.sum(K.sum(loss))
        return loss

    def get_pred_beam(self, alpha, beta):
        N = self.args["problem_size"]
        xx, yy = self.create_cartesian_meshgrid()
        r, theta = self.cart2pol(xx, yy)
        phi = 2 * np.pi / N * np.array(list(map(lambda x: x + 1, list(range(N)))))
        z = tf.cast(tf.complex(alpha * tf.math.cos(beta), alpha * tf.math.sin(beta)), tf.complex128)
        
        nb_matrix = tf.stack([tf.complex(tf.math.cos(r * tf.math.cos(phi[n] - theta)), tf.math.sin(r * tf.math.sin(phi[n] - theta))) for n in range(0,N)])
        pred_beam = tf.tensordot(z, nb_matrix, axes = ([1,0]))
        return tf.cast(tf.math.real(pred_beam), tf.float32), tf.cast(tf.math.imag(pred_beam), tf.float32)

    def gaussian_mask(self):
        xx, yy = self.create_cartesian_meshgrid()
        r, theta = self.cart2pol(xx, yy)
        sigma = self.args["loss_sigma"]
        zz = np.sqrt(2/np.pi)* np.exp(-(r**2/(2*sigma**2))) * r 
        zz = tf.constant(zz, dtype=np.float32)
        return zz
    
    
    def create_cartesian_meshgrid(self):
        window_size = self.args["window_size"]
        resolution = self.args["resolution"]
        x = np.linspace(-window_size, window_size, resolution)
        y = np.linspace(-window_size, window_size, resolution)

        xx, yy = np.meshgrid(x, y)
        return xx.astype("float64"), yy.astype("float64")

    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)
    
class BeamCompareAnimationCallback(tf.keras.callbacks.Callback):
    # Añadimos parámetros para el nombre del GIF y la frecuencia de actualización
    def __init__(self, nblab_instance, sample_x, sample_y, save_path="evolucion.gif", freq=2):
        super().__init__()
        self.nblab = nblab_instance
        self.sample_x = sample_x
        self.sample_y = sample_y  
        self.save_path = save_path
        self.freq = freq  # Solo dibuja cada 'freq' épocas
        self.frames = []  # Aquí guardaremos las imágenes para el GIF
        self.im1 = None
        self.im2 = None
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plt.pause(0.1)

    def on_epoch_end(self, epoch, logs=None):
        # 1. ACELERACIÓN: Solo renderizar la época 0 y luego cada 'freq' épocas
        if epoch != 0 and (epoch + 1) % self.freq != 0:
            return

        # Predecir
        y_pred = self.model.predict(np.array([self.sample_x]), verbose=0)
        
        len_y_pred = y_pred.shape[1]
        alpha = y_pred[0, :len_y_pred//2]
        beta = y_pred[0, len_y_pred//2:]
        
        abs_pred_beam, _ = self.nblab.compose_beam(alpha, beta)
        intensity_pred = np.sqrt(np.abs(abs_pred_beam[:, :, 0])).astype(float)

        if self.im1 is None:
            intensity_true = np.sqrt(np.abs(self.sample_y[:, :, 0])).astype(float)
            
            self.im1 = self.ax1.imshow(intensity_pred, cmap='hot', aspect='auto')
            self.ax1.set_title("Predicción", fontweight='bold')
            self.fig.colorbar(self.im1, ax=self.ax1)
            
            self.im2 = self.ax2.imshow(intensity_true, cmap='hot', aspect='auto')
            self.ax2.set_title("Objetivo Real", fontweight='bold')
            self.fig.colorbar(self.im2, ax=self.ax2)
        else:
            self.im1.set_data(intensity_pred)
            
            vmin = np.min(intensity_pred)
            vmax = np.max(intensity_pred)
            if vmin == vmax: 
                vmax = vmin + 1e-5
            self.im1.set_clim(vmin=vmin, vmax=vmax)
        
        loss_val = logs.get('loss', 0)
        self.fig.suptitle(f"Época: {epoch + 1} | Loss: {loss_val:.4f}", fontsize=14, fontweight='bold')
        
        # 2. Reducimos el tiempo de pausa al mínimo para que fluya rápido
        plt.pause(0.001) 
        
        # 3. CAPTURA DE FRAME: Convertimos la figura de Matplotlib a una imagen de PIL
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).copy()
        self.frames.append(img)
        buf.close()

    def on_train_end(self, logs=None):
        plt.ioff()
        # 4. GUARDAR EL GIF AL TERMINAR
        if len(self.frames) > 0:
            print(f"\nGuardando animación en {self.save_path}...")
            
            self.frames[0].save(
                self.save_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=20,  # <--- ¡AQUÍ ESTÁ LA CLAVE! Bájsalo a 20 o 30
                loop=0
            )
            print("¡GIF guardado con éxito!\n")
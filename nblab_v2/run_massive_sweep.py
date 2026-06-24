import os
import glob
import json
import time
import itertools
import subprocess
import pandas as pd
from datetime import datetime

# =====================================================================
# 1. DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS
# =====================================================================
beams = [f"beams/beam_v{i}_256.mat" for i in range(2, 11)]

# Rango de N: de 2 a 40. (Paso de 2 en 2: [2, 4, 6... 40]). 
# Si de verdad quieres los 39 enteros seguidos, cambia el '2' final por un '1'.
N_values = list(range(2, 10, 1)) 

loss_sigmas = [1.0, 10.0, 100.0, 100000.0]
learning_rates = [1e-2,  1e-4, 1e-6]

combinations = list(itertools.product(N_values, loss_sigmas, learning_rates))
total_runs = len(beams) * len(combinations)

print("="*60)
print(" INICIANDO ORQUESTADOR DE BARRIDO MASIVO - NBLAB")
print(f" Haces detectados         : {len(beams)}")
print(f" Combinaciones por haz    : {len(combinations)}")
print(f" TOTAL DE EXPERIMENTOS    : {total_runs}")
print("="*60 + "\n")

# =====================================================================
# 2. PREPARACIÓN DEL LOG MAESTRO
# =====================================================================
timestamp_sesion = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_maestro = f"master_sweep_summary_{timestamp_sesion}.csv"

log_acumulado = []
global_start_time = time.time()
run_id = 1

try:
    for beam_path in beams:
        if not os.path.exists(beam_path):
            print(f"[!] ADVERTENCIA: No se encontró el archivo {beam_path}. Saltando...")
            continue

        for N, sigma, lr in combinations:
            beam_name = os.path.basename(beam_path)
            print(f"[{run_id}/{total_runs}] Ejecutando -> Haz: {beam_name} | N={N} | sigma={sigma} | lr={lr}")

            # Construimos el comando de consola aislando el entorno
            cmd = [
                "python", "nblab_experiment.py",
                "-beam", str(beam_path),
                "-N", str(N),
                "-loss_sigma", str(sigma),
                "-learning_rate", str(lr),
                "-epochs", "10000",
                "-plot", "False",       # APAGADO: Si se abre una ventana de matplotlib, el código se congela
                "-live_anim", "False",  # APAGADO: No queremos ver 4,800 películas
                "-save_gif", "False"     # <-- OJO CON ESTO (Lee la advertencia abajo)
            ]

            t0 = time.time()
            try:
                # Ejecutamos el experimento en un sub-proceso hermético
                proceso = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                tiempo_ejecucion = time.time() - t0

                # Inspeccionamos la carpeta de outputs más reciente generada
                carpetas_generadas = glob.glob(os.path.join("outputs", "*"))
                carpeta_actual = max(carpetas_generadas, key=os.path.getmtime)
                json_result = os.path.join(carpeta_actual, "results.json")

                if os.path.exists(json_result):
                    with open(json_result, 'r') as f:
                        data_json = json.load(f)
                    
                    meta = data_json["metadata"]
                    final_loss = meta["final_apodized_loss"]
                    k_t_inf = meta["k_t_inferred"]
                    estado = "EXITOSO"
                else:
                    final_loss, k_t_inf = None, None
                    estado = "ERROR: No se generó results.json"

            except Exception as e:
                tiempo_ejecucion = time.time() - t0
                final_loss, k_t_inf = None, None
                carpeta_actual = "NINGUNA"
                estado = f"CRASH: {str(e)}"

            # Registramos el resultado en la memoria de pandas
            fila = {
                "run_id": run_id,
                "beam_file": beam_name,
                "N": N,
                "loss_sigma": sigma,
                "learning_rate": lr,
                "final_apodized_loss": final_loss,
                "k_t_inferred": k_t_inf,
                "exec_time_seconds": round(tiempo_ejecucion, 2),
                "status": estado,
                "output_directory": carpeta_actual
            }

            log_acumulado.append(fila)
            
            # SOBRESCRIBIMOS EL CSV EN CADA PASO (Protección anti-apagones)
            df_log = pd.DataFrame(log_acumulado)
            df_log.to_csv(csv_maestro, index=False)

            print(f"   ↳ Estado: {estado} | Pérdida: {final_loss} | k_t: {k_t_inf} ({tiempo_ejecucion:.1f}s)\n")
            run_id += 1

except KeyboardInterrupt:
    print("\n" + "="*60)
    print(" [!] SECUENCIA ABORTADA MANUALMENTE POR EL USUARIO (Ctrl+C)")
    print(f" El progreso hasta el run {run_id-1} está a salvo en: {csv_maestro}")
    print("="*60)
    exit()

horas_totales = (time.time() - global_start_time) / 3600.0
print("\n" + "="*60)
print(f" BARRIDO GIGANTE COMPLETADO EN {horas_totales:.2f} HORAS.")
print(f" Tabla de posiciones final guardada en: {csv_maestro}")
print("="*60)
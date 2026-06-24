import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUTS_DIR = "outputs"
ANALYTICS_DIR = "sweep_analytics"

def harvest_all_outputs():
    """Recorre recursivamente todas las carpetas buscando results.json"""
    json_files = glob.glob(os.path.join(OUTPUTS_DIR, "*", "results.json"))
    
    if not json_files:
        print(f"[!] No se encontraron archivos results.json dentro de '{OUTPUTS_DIR}/'.")
        return pd.DataFrame()

    records = []
    for j_path in json_files:
        parent_folder = os.path.basename(os.path.dirname(j_path))
        try:
            with open(j_path, 'r') as f:
                d = json.load(f)
            
            meta = d["metadata"]
            beam_clean = os.path.splitext(os.path.basename(meta["source_file"]))[0]
            
            records.append({
                "folder": parent_folder,
                "beam": beam_clean,
                "N": meta.get("N_modes", np.nan),
                "loss_sigma": meta.get("loss_sigma", np.nan),
                "learning_rate": meta.get("learning_rate", np.nan),
                "final_loss": meta.get("final_apodized_loss", np.nan),
                "k_t_inferred": meta.get("k_t_inferred", np.nan),
                "timestamp": meta.get("timestamp", "")
            })
        except Exception as e:
            print(f"  ↳ Error parseando {parent_folder}/results.json: {e}")
            continue

    df = pd.DataFrame(records)
    df = df.dropna(subset=["final_loss"]) # Descartar corridas corruptas
    return df


def generate_excel_report(df, output_excel_path):
    """Crea un Excel profesional con múltiples hojas de cálculo"""
    print(f"\n[+] Generando reporte Excel en: '{output_excel_path}'...")
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        
        # --- PESTAÑA 1: EL SALÓN DE LA FAMA (Mejor corrida por cada Haz) ---
        # Ordenamos por pérdida y nos quedamos con la primera aparición de cada haz
        top1_df = df.sort_values("final_loss").groupby("beam", as_index=False).first()
        top1_df.sort_values("beam", inplace=True)
        top1_df.to_excel(writer, sheet_name="TOP1_GANADORES", index=False)

        # --- PESTAÑAS INDIVIDUALES: Una hoja por cada Haz ---
        haces_unicos = sorted(df["beam"].unique())
        for haz in haces_unicos:
            df_haz = df[df["beam"] == haz].sort_values("final_loss")
            
            # Excel limita el nombre de las pestañas a 31 caracteres
            sheet_title = str(haz)[:31] 
            df_haz.to_excel(writer, sheet_name=sheet_title, index=False)


def plot_global_diagnostics(df):
    """Genera los 3 gráficos clave de interpretación del barrido"""
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    print(f"[+] Generando gráficos de diagnóstico en '{ANALYTICS_DIR}/'...")

    # =================================================================
    # GRÁFICO 1: El "Codo" de N (¿Dónde se satura la base de ondas?)
    # =================================================================
    plt.figure(figsize=(10, 6))
    haces = df["beam"].unique()
    
    for haz in haces:
        sub = df[df["beam"] == haz]
        # Para cada N, graficamos el que obtuvo la mínima pérdida
        best_n = sub.groupby("N")["final_loss"].min().reset_index()
        plt.plot(best_n["N"], np.log10(best_n["final_loss"]), marker='o', label=haz, alpha=0.8)

    plt.title("Estancamiento Espectral: log10(Pérdida) vs. N modos")
    plt.xlabel("Número de Ondas Planas (N)")
    plt.ylabel("log10(Mínima Pérdida Apodizada)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYTICS_DIR, "1_espectro_codo_N.png"), dpi=300)
    plt.close()

    # =================================================================
    # GRÁFICO 2: Sensibilidad a la Tasa de Aprendizaje (Boxplot)
    # =================================================================
    if not df["learning_rate"].isna().all():
        plt.figure(figsize=(8, 5))
        # Filtramos el 10% peor de los datos para que los outliers extremos no aplasten la gráfica
        techo = df["final_loss"].quantile(0.85)
        df_filtrado = df[df["final_loss"] <= techo]
        
        lrs = sorted(df_filtrado["learning_rate"].unique())
        data_to_plot = [df_filtrado[df_filtrado["learning_rate"] == lr]["final_loss"] for lr in lrs]
        
        plt.boxplot(data_to_plot, labels=[str(lr) for lr in lrs])
        plt.yscale("log")
        plt.title("Estabilidad de Convergencia según Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Pérdida Final (Escala Log)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYTICS_DIR, "2_sensibilidad_learning_rate.png"), dpi=300)
        plt.close()

    # =================================================================
    # GRÁFICO 3: Distribución del parámetro k_t inferido
    # =================================================================
    plt.figure(figsize=(8, 5))
    # Nos quedamos solo con las corridas "buenas" (el 25% superior) para ver dónde ancló la física
    buenos = df[df["final_loss"] <= df["final_loss"].quantile(0.25)]
    
    plt.hist(buenos["k_t_inferred"], bins=30, color='teal', edgecolor='black', alpha=0.7)
    plt.title("Frecuencia Transversal (k_t) inferida en las mejores reconstrucciones")
    plt.xlabel("Valor de k_t [rad / unidad espacial]")
    plt.ylabel("Cantidad de Experimentos")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYTICS_DIR, "3_distribucion_kt.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    df_master = harvest_all_outputs()
    
    if df_master.empty:
        exit()

    print("="*65)
    print(f" BASE DE DATOS CONSTRUIDA: {len(df_master)} experimentos analizados.")
    print("="*65)

    # 1. Imprimir resumen rápido en consola
    print("\n--- PODIO: LOS GANADORES ABSOLUTOS POR HAZ ---")
    podio = df_master.sort_values("final_loss").groupby("beam").first()
    
    for haz, fila in podio.iterrows():
        print(f" • {haz:<14} | Loss: {fila['final_loss']:.5f} | N: {fila['N']:<2} | lr: {fila['learning_rate']} | kt: {fila['k_t_inferred']:.2f}")

    # 2. Generar Excel
    excel_path = os.path.join(ANALYTICS_DIR, "Reporte_Fisica_Barrido.xlsx")
    generate_excel_report(df_master, excel_path)

    # 3. Generar Gráficas
    plot_global_diagnostics(df_master)

    print("\n[✓] ANÁLISIS COMPLETADO. Revisa la carpeta 'sweep_analytics/'.\n")
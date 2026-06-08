import os
import ast
import numpy as np 
import pandas as pd
import argparse
import scipy.io


def get_args():
    parsed_args = get_parsed_args()
    args = {arg: getattr(parsed_args, arg) for arg in vars(parsed_args)}
    for key, value in args.items():
        try: 
            args[key] = ast.literal_eval(value)
        except:
            continue
    return args

def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-beam", default="gaussian")
    parser.add_argument("-resolution", default=128)
    parser.add_argument("-window_size", default=5)
    parser.add_argument("-mu", default=0)
    parser.add_argument("-sigma", default=1)
    parser.add_argument("-n_order", default=2)
    parser.add_argument("-m_order", default=2)
    parser.add_argument("-loss_sigma", default=100)
    parser.add_argument("-num_plane_waves", default=360)
    parser.add_argument("-kt", default=1)
    parser.add_argument("-save", default=False)
    parser.add_argument("-plot", default=False)

    #Mathieu parameters
    parser.add_argument("-q", default=1)
    parser.add_argument("-a", default=1)
    parser.add_argument("-k", default=10)
    parser.add_argument("-alpha", default=1)
    parser.add_argument("-theta", default=np.pi/4)

    #Parabolic parameter
    parser.add_argument("-parabolic_type", default="even")
    parser.add_argument("-a_param", default=1)
    parser.add_argument("-r_anillo", default=1)
    parser.add_argument("-dr", default=1e-4)

    parser.add_argument("-problem_size", default=5)
    parser.add_argument("-num_samples", default=5)
    parser.add_argument("-alpha_upper_bound", default=1)
    parser.add_argument("-alpha_lower_bound", default=-1)
    parser.add_argument("-beta_upper_bound", default=1)
    parser.add_argument("-beta_lower_bound", default=-1)

    parser.add_argument("-epochs", default=30)
    parser.add_argument("-batch_size", default=32)
    parser.add_argument("-units", default=1)
    parser.add_argument("-learning_rate", default=0.01)

    parser.add_argument("-heuristic", default="greedy")
    parser.add_argument("-greedy_num_iterations", default=200)
    parser.add_argument("-ga_population_size", default=500)
    parser.add_argument("-ga_generations", default=25)
    parser.add_argument("-ga_comb_rate", default=0.9)
    parser.add_argument("-ga_mutation_rate", default=0.1)
    parsed_args = parser.parse_args()
    return parsed_args

def get_beam(beams, args):
    beam = beams[args["beam"]](args)
    return beam

def create_cartesian_meshgrid(window_size, resolution):
    x = np.linspace(-window_size, window_size, resolution)
    y = np.linspace(-window_size, window_size, resolution)

    xx, yy = np.meshgrid(x, y)
    return xx.astype("float64"), yy.astype("float64")

def save_experiment(args, alpha, beta, pred_beam, true_beam, execution_time):
    save_name = args["beam"] + "_" + str(args["resolution"]) + "_" + str(args["window_size"]) + "_" + str(args["problem_size"]) + "_" + str(args["loss_sigma"]) 
    
    # Es buena práctica usar os.path.join también aquí para evitar problemas con las barras "/"
    save_path = os.path.join(os.environ["NBLAB_PATH"], "Results", save_name)
    
    # Create save_path and assume it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    weights = pd.DataFrame()
    weights["alpha"] = alpha
    weights["beta"] = beta
    weights["execution_time"] = execution_time
    weights["error"] = compute_error(pred_beam, true_beam)
    
    # CORRECCIÓN: Guardar el CSV DENTRO de save_path con el nombre "weights.csv"
    csv_file = os.path.join(save_path, "weights.csv")
    weights.to_csv(csv_file, index=False)
    
    args_to_save = {key: value for key, value in args.items() if key not in ["save"]}

    # CORRECCIÓN: Guardar el TXT DENTRO de save_path con el nombre "args.txt"
    txt_file = os.path.join(save_path, "args.txt")
    with open(txt_file, "w") as f:
        for key, value in args_to_save.items():
            f.write(f"{key}: {value}\n")

    # CORRECCIÓN: Guardar el MAT DENTRO de save_path con el nombre "data.mat" (o el que prefieras)
    mat_file = os.path.join(save_path, "data.mat")
    scipy.io.savemat(mat_file, {"pred_beam": pred_beam, "true_beam": true_beam})

def compute_error(pred_beam, true_beam):
    return np.sum(np.abs(pred_beam - true_beam))
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def get_sample(nblab):
    sample, alpha, beta = nblab.generate_random_sample()
    df = pd.DataFrame.from_dict({"0" : sample}, orient="index")
    df = (df - df.mean())/(df.std())
    return df

def create_sample(alpha, beta):
    sample = {}
    for i in range(len(alpha)):
        sample[f"alpha_{i+1}"] = alpha[i]
    for i in range(len(beta)):
        sample[f"beta_{i+1}"] = beta[i]
        
    df = pd.DataFrame.from_dict({"0" : sample}, orient="index")
    df = (df - df.mean())/(df.std())
    return df

def get_fft(beam):
    beam = beam[:,:,0] + 1j*beam[:,:,1]
    fft_beam = np.fft.fft2(beam)
    return fft_beam



import numpy as np 
import pandas as pd


def create_cartesian_meshgrid(window_size, resolution):
    x = np.linspace(-window_size, window_size, resolution)
    y = np.linspace(-window_size, window_size, resolution)

    xx, yy = np.meshgrid(x, y)
    return xx, yy

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
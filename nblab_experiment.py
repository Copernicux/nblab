import os
from NBLab import NBLab
from Beams.beam_dict import beams
from utils import get_args, get_beam, save_experiment
import time
import matplotlib.pyplot as plt
import numpy as np 


def nblab_experiment(args):

    start_time = time.time()
    beam = get_beam(beams, args)
    lab = NBLab(beam, args)
    data = lab.get_data()

    if args["plot"]:
        lab.plot_beam(beam)
    
    model, history = lab.train_model(data)
    alpha, beta = lab.search(model)
    abs_pred_beam, complex_pred_beam  = lab.compose_beam(alpha, beta)
    execution_time = time.time() - start_time
    print(f"--- {execution_time}s seconds ---")

    if args["plot"]:
        lab.plot_compare_beams(abs_pred_beam, data["y"][0])
        lab.plot_history(history)
        lab.plot_beam_phase(complex_pred_beam[:,:,0])
        lab.plot_beam_phase(beam[-1][:,:,0])
        lab.plot_alpha_beta(alpha, beta)
    
    if args["save"]:
        save_experiment(args, alpha, beta, abs_pred_beam, data["y"][0], execution_time)
    


if __name__ == "__main__":
    args = get_args()
    nblab_experiment(args)
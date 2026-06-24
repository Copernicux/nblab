import pandas as pd
import numpy as np

def greedy(nblab, model):

    num_iterations = nblab.args["greedy_num_iterations"]
    sample, alpha, beta = nblab.generate_random_sample()
    # df = pd.DataFrame.from_dict({"0" : sample}, orient="index")
    # df = (df - df.mean())/(df.std())
    # val = model.predict(df)
    pred_beam = nblab.nb_superposition(alpha, beta)
    val = nblab.obj_fun(pred_beam)
    for i in range(num_iterations):
        print(i)
        sample, new_alpha, new_beta = nblab.generate_random_sample()
        # df = pd.DataFrame.from_dict({"0" : sample}, orient="index")
        # df = (df - df.mean())/(df.std())
        # new_val = model.predict(df)
        new_pred_beam = nblab.nb_superposition(new_alpha, new_beta)
        new_val = nblab.obj_fun([new_pred_beam])
        if new_val < val:
            val = new_val
            alpha, beta = new_alpha, new_beta
    
    return alpha, beta 
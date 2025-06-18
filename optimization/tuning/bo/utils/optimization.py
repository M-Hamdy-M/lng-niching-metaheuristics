"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy                                  ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""
import os
import pickle as pkl


module_path = os.path.abspath(os.path.dirname(__file__))
def get_hyperparams(problem_name, algorithm_name, id, base_directory=None):
    if base_directory is not None:
        output_loc = os.path.join(base_directory, f"{problem_name}/{algorithm_name}")
    else:
        output_loc = os.path.join(module_path, f"../hyperparams/{problem_name}/{algorithm_name}")
               
    save_path = os.path.join(output_loc, f"{id}.pk")

    
    with open(save_path , "rb") as file :
        best_dict = pkl.load(file)
    return best_dict
"""
╔════════════════════════════════════════════════════════════════╗
║  Implemented by Mohamed Hamdy & Shahd Gaben                    ║
║  Contact: mm1905748@qu.edu.qa                                  ║
╚════════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import win32com.client as win32
from optimization.problems.types.constraint_handlers import PenaltyConstraintHandler, StaticPenaltyConstraintHandler, RangeConstraint
from optimization.problems.types import ConstrainedOptimizationProblem


class SMROptimizationProblem(ConstrainedOptimizationProblem):
    def __init__(self,
                 file_name: str,
                 name: str = "SMR",
                 constraint_handler: PenaltyConstraintHandler = StaticPenaltyConstraintHandler(constraints=[
                     RangeConstraint(factor=1, low=3),                 
                     RangeConstraint(factor=1, low=3),                 
                 ], beta=1),
                 timeout=10_000,
                 simulation_is_visible=False,
                 sleep_duration=100,
                 max_trials=30, 
                 do_scale=False) -> None:
        self.simulation_file = file_name
        self.timeout = timeout
        self.sleep_duration = sleep_duration
        self.initiate_connection(simulation_is_visible)
        self.change_indices = {0: 8, 1: 0, 2: 1, 3: 2, 4: 3, 5: 5}
        self.max_trials = max_trials
        self.total_trials = 0
        bounds = np.array([
            [0.15, 0.28], 
            [0.3, 0.55], 
            [0.7, 1.15], 
            [0.8, 1.25], 
            [0.5, 0.8], 
            [0.6, 1], 
            [1.5, 3.5], 
            [35, 55], 
            ])
        super().__init__(name=name, evaluator=self._evaluate, bounds=bounds,
                         is_maximization=False, constraint_handler=constraint_handler, do_scale=do_scale, 
                         variable_names=["N2", "C1", "C2", "C3", "iC4", "iC5", "S17 Pressure", "S23 Pressure"],
                         constraint_names=["MITA1", "MITA2"])

    def initiate_connection(self, is_visible):
        try:
            hysys_file = os.path.abspath(self.simulation_file)
            hyapp = win32.Dispatch("HYSYS.Application")
            sim_case = hyapp.SimulationCases.Open(hysys_file)
            sim_case.Visible = is_visible
            self.simulation_case = sim_case
            self.solver = sim_case.solver
            self.sim_case = sim_case
            self.flow_sheet = sim_case.FlowSheet
            self.mr_stream = self.flow_sheet.MaterialStreams.Item("5")
            self.suction_stream = self.flow_sheet.MaterialStreams.Item("23")
            self.discharge_stream = self.flow_sheet.MaterialStreams.Item("17")
        except Exception as e:
            print("Error Connecting to simulation!")
            raise e
            # raise Exception(f"Error Connecting to simulation: {str(e)}")

    def get_dec_vars(self):
        dec_vars = np.empty(self.input_dimension)
        mfl = np.array(self.mr_stream.ComponentMassFlow.GetValues("kg/h"))
        dec_vars[list(self.change_indices.keys())] = mfl[list(self.change_indices.values())]
        dec_vars[6] = self.suction_stream.Pressure.GetValue("bar")
        dec_vars[7] = self.discharge_stream.Pressure.GetValue("bar")
        return dec_vars

    def get_constraints(self):
        return np.array([
            self.sim_case.Flowsheet.Operations.Item("LNG-100").Specifications.Item("ExchSpec").CurrentValue,
            self.sim_case.Flowsheet.Operations.Item("LNG-101").Specifications.Item("ExchSpec").CurrentValue,
            ])
    
    def get_objective(self):
        return self.sim_case.Flowsheet.Operations.Item(
                            "Objective Values").Cell("B8").CellValue
                            
    def _evaluate(self, individuals):
        obj_values = np.zeros(len(individuals))
        constraint_values = np.zeros((len(individuals), self.n_constraints))
        for idx, dec_vars in enumerate(individuals):
            if np.any(dec_vars > self.original_bounds[:, 1]) or np.any(dec_vars < self.original_bounds[:, 0]):
                print("WARNING: the given decision variables are out of bounds")
            try:
                self.solver.CanSolve = False
                # change the mass flows
                new_mfl = np.zeros(9)
                new_mfl[list(self.change_indices.values())
                        ] = dec_vars[list(self.change_indices.keys())].round(4)
                self.mr_stream.MassFlow.SetValue(np.sum(new_mfl), "kg/h")
                self.mr_stream.ComponentMassFlow.SetValues(new_mfl, "kg/h")

                # change the pressures
                self.suction_stream.Pressure.SetValue(dec_vars[6], "bar")
                self.discharge_stream.Pressure.SetValue(dec_vars[7], "bar")
                
                trial = 0
                start_time = time.time()
                while True:
                    try:
                        self.solver.CanSolve = True

                        while self.solver.IsSolving:
                            te = time.time() - start_time
                            if te >= self.timeout:
                                raise Exception(
                                    f"Timeout Exception: simulation took more than expected. time elapsed: {te} seconds")
                            time.sleep(self.sleep_duration)
                        obj_values[idx] = self.get_objective()
                        constraint_values[idx] = self.get_constraints()
                        break
                    except Exception:
                        trial += 1
                        if trial % 5 == 0:
                            print(f"Warning Reached Trial No. {trial} for dec_var: {dec_vars}")
                        if trial >= self.max_trials:
                            print(f"Error: Reached more than {self.max_trials} trial, Terminating...")
                            raise Exception("Error: Reached more than 1000 trial, Terminating...")
                        continue
            except Exception as e:
                raise Exception(
                    f"Error Evaluating {dec_vars}: {e}")
            self.total_trials += trial
        return obj_values, constraint_values

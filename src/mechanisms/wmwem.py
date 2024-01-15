import numpy as np
import itertools
from scipy.special import softmax
from scipy import sparse
from cdp2adp import cdp_rho
from mechanism import Mechanism
import pdb
import math
from mbi.inference import FactoredInference
from mbi.graphical_model import GraphicalModel
from utils import downward_closure
from mbi.evaluation import Evaluator
from mbi.advanced_sliced_inference import AdvancedSlicedInference
import torch
from mbi.particle_model import ParticleModel



"""
This file contains an implementation of MWEM+PGM that is designed specifically for marginal query workloads.
Unlike mwem.py, which selects a single query in each round, this implementation selects an entire marginal 
in each step.  It leverages parallel composition to answer many more queries using the same privacy budget.

This enhancement of MWEM was described in the original paper in section 3.3 (https://arxiv.org/pdf/1012.4763.pdf).

There are two additional improvements not described in the original Private-PGM paper:
- In each round we only consider candidate cliques to select if they result in sufficiently small model sizes
- At the end of the mechanism, we generate synthetic data (rather than query answers)
"""
class WMWEM(Mechanism):
    def __init__(self, prng=None, structural_zeros={}, hp={}):
        super(WMWEM, self).__init__(hp["epsilon"], hp["delta"])
        self.epsilon = hp["epsilon"]
        self.delta = hp["delta"]
        self.k = hp["degree"]
        self.hp = hp



    def worst_approximated(self, dists, workload, eps,N):
        """ Select a (noisy) worst-approximated marginal for measurement.
        
        :param workload_answers: a dictionary of true answers to the workload
            keys are cliques
            values are numpy arrays, corresponding to the counts in the marginal
        :param est: a GraphicalModel object that approximates the data distribution
        :param: workload: The list of candidates to consider in the exponential mechanism
        :param eps: the privacy budget to use for this step.
        """
        max_len = max([len(cl) for cl in workload])

        errors = np.array([])
        for cl in workload:

            errors = np.append(errors,dists[cl]/math.sqrt(len(cl)) *N)
        prob = softmax(0.5*eps*(errors - errors.max()))
        key = np.random.choice(len(errors), p=prob)
        return workload[key]





    def run(self, data, workload, engine, rounds=None, noise='gaussian', bounded=True, alpha=0.9):
        """
        Implementation of MWEM + PGM

        :param data: an mbi.Dataset object
        :param epsilon: privacy budget
        :param delta: privacy parameter (ignored)
        :param workload: A list of cliques (attribute tuples) to include in the workload (default: all pairs of attributes)
        :param rounds: The number of rounds of MWEM to run (default: number of attributes)
        :param maxsize_mb: [New] a limit on the size of the model (in megabytes), used to filter out candidate cliques from selection.
            Used to avoid MWEM+PGM failure modes (intractable model sizes).   
            Set to np.inf if you would like to run MWEM as originally described without this modification 
            (Note it may exceed resource limits if run for too many rounds)

        Implementation Notes:
        - During each round of MWEM, one clique will be selected for measurement, but only if measuring the clique does
            not increase size of the graphical model too much
        """ 
        if workload is None:
            workload = list(itertools.combinations(data.domain, 2))
        if rounds is None:
            rounds = len(data.domain)

        bounded = True
        rounds = min(len(workload), self.hp['wmwem_rounds'])

        evaluator = Evaluator(data,data,workload)


        rho = self.rho
        rho_per_round = rho / rounds
        sigma = np.sqrt(0.5 / (alpha*rho_per_round))
        exp_eps = np.sqrt(8*(1-alpha)*rho_per_round)
        marginal_sensitivity = np.sqrt(2) if bounded else 1.0

        domain = data.domain
        total = data.records if bounded else None

        def size(cliques):
            return GraphicalModel(domain, cliques).size * 8 / 2**20


        workload_answers = { cl : data.project(cl).datavector() for cl in workload }

        measurements = []
        if isinstance(engine, FactoredInference):
            est, loss = engine.estimate(measurements, total)
        else:
            est  = ParticleModel(
            data.domain,
            embedding=engine.embedding,
            n_particles=engine.n_particles,
            data_init=self.hp["data_init"])
            loss =0.0
        cliques = []
        for i in range(1, rounds+1):
            # [New] Only consider candidates that keep the model sufficiently small
            synth = est.synthetic_data()
            evaluator.update_synth(synth)
            if isinstance(engine, FactoredInference):
                candidates = [cl for cl in workload if size(cliques+[cl]) <= self.hp['max_model_size']*i/rounds]
                ax = self.worst_approximated( evaluator.evaluate(), candidates, exp_eps, data.df.shape[0])

                print('Round', i, 'Selected', ax, 'Model Size (MB)', est.size*8/2**20)
            else:
                ax = self.worst_approximated( evaluator.evaluate(), workload, exp_eps, data.df.shape[0])

            n = domain.size(ax)
            x = data.project(ax).datavector()

            y = x + np.random.normal(loc=0, scale=marginal_sensitivity*sigma, size=n)
            Q = sparse.eye(n)
            measurements.append((Q, y, 1.0, ax))
            est, loss= engine.estimate(measurements, total)
            cliques.append(ax)

        print('Generating Data...')
        return est.synthetic_data(), loss

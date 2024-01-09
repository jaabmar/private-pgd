import torch
import numpy as np
import os
from inference.embedding import Embedding
from ot import sliced_wasserstein_distance  
import pickle
from itertools import combinations
import pdb

class DatasetEvaluator:
    def __init__(self, X, Y, dataset_name, main_path):
        self.X = X
        self.Y = Y
        self.embedding = Embedding(X.domain)
        self.dataset = dataset_name

        self.Xemb = self.embedding.embedd(self.X.df) 
        self.Yemb = self.embedding.embedd(self.Y.df)
        self.Xembnp = self.Xemb.cpu().detach().numpy()        
        self.Yempnp = self.Yemb.cpu().detach().numpy()
        self.domain = self.X.domain.shape
        self.main_path = main_path



    def swd(self, workload, num_projections=50, p=1):
            result = {}
            
            # Loop over each key in workload dictionary
            max_swd_for_marginals = []
            avg_swd_for_marginals = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dims = self.embedding.get_dims(workload)
            for proj in workload:
                dim = dims[proj].to(self.Xemb.device)
                
                projected_X_full = self.Xemb[:,dim]
                projected_X, counts_X = torch.unique(projected_X_full, dim=0, return_counts=True)
                counts_X = counts_X / torch.sum(counts_X).float()

                project_Y_full = self.Yemb[:,dim]
                project_Y, counts_Y = torch.unique(project_Y_full, dim=0, return_counts=True)
                counts_Y = counts_Y / torch.sum(counts_Y).float()


                # Ensure that sliced_wasserstein_distance can work with GPU tensors or
                # is appropriately modified to do so
                swd = sliced_wasserstein_distance(
                    projected_X, project_Y, counts_X, counts_Y, n_projections=num_projections, p=p
                ) ** p



                max_swd_for_marginals.append(swd.item())  # move swd to CPU and convert to Python scalar
                avg_swd_for_marginals.append(swd.item())

                
                # Compute and store the maximum and average SWD with keys formed as required
            result[f"max"] = max(max_swd_for_marginals)
            result[f"avg"] = np.mean(avg_swd_for_marginals)
            
            return result


    def lp_distances(self, workload, p=1):
            result = {}
            
            # Loop over each key in workload dictionary
            lp_for_marginals = []
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for proj in workload:



                X = self.X.project(proj).datavector()
                Y = self.Y.project(proj).datavector()
                e = np.linalg.norm(X / X.sum() - Y / Y.sum(), p)**p

                lp_for_marginals.append(e)  # move swd to CPU and convert to Python scalar

                
                # Compute and store the maximum and average SWD with keys formed as required
            result[f"max"] = max(lp_for_marginals)
            result[f"avg"] = np.mean(lp_for_marginals)
            
            return result


    def correlation_difference(self):
        # Centering the rows of self.X and self.X_prime
        centered_X = self.Xembnp - np.mean(self.Xembnp, axis=1, keepdims=True)
        centered_X_prime = self.Yempnp - np.mean(self.Yempnp, axis=1, keepdims=True)

        # Normalizing the centered rows by self.domain
        normalized_X = centered_X 
        normalized_X_prime = centered_X_prime 

        # Compute the correlation matrices for the normalized matrices
        corr_X = np.cov(normalized_X, rowvar=False)
        corr_X_prime = np.cov(normalized_X_prime, rowvar=False)
        diff = corr_X - corr_X_prime
        
        frobenius_norm = np.linalg.norm(diff, 'fro')
        spectral_norm = np.linalg.norm(diff, 2)

        
        return {
            "frobenius_norm": frobenius_norm,
            "spectral_norm": spectral_norm,
            "spectral_base": np.linalg.norm(corr_X, 2), 
            "frobenius_base":np.linalg.norm(corr_X, 'fro')

        }

       


    def random_thresholding(self, num_hyperplanes=200, num_dimensions=3):
        if num_dimensions is None:
            num_dimensions = self.Xembnp.shape[1]

        chosen_dimensions, w_data, b_data = self._get_or_save_random_data('thresholding_data.pkl', 
                                                                          self.generate_thresholding_data,
                                                                          num_hyperplanes, num_dimensions)

        discrepancies = []
        orig = []

        for i in range(num_hyperplanes):
            w = w_data[i]
            w /= np.linalg.norm(w)

            min_b = np.min(np.dot(self.Xembnp[:, chosen_dimensions], w))
            max_b = np.max(np.dot(self.Xembnp[:, chosen_dimensions], w))

            b_range = max_b - min_b
            b = b_data[i] * b_range + min_b

            prop_X = np.mean(np.dot(self.Xembnp[:, chosen_dimensions], w) + b > 0)
            prop_X_prime = np.mean(np.dot(self.Yempnp[:, chosen_dimensions], w) + b > 0)

            discrepancies.append(np.abs(prop_X - prop_X_prime))
            orig.append(np.abs(prop_X))
        return {"mean_dist": np.mean(discrepancies), "orig": np.mean(orig) }



    def random_counting_queries(self, num_queries=200, s=3):
        if s > self.Xembnp.shape[1]:
            raise ValueError("s should be less than or equal to number of columns in X")

        chosen_columns_data, bounds_data = self._get_or_save_random_data('counting_query_data_new.pkl',
                                                                         self.generate_counting_query_data,
                                                                         num_queries, s)

        discrepancies = []
        orig_counts = []
        nx = self.Xembnp.shape[0]
        nxprime = self.Yempnp.shape[0]

        for i in range(num_queries):
            chosen_columns = chosen_columns_data[i]
            lower_bounds, upper_bounds = bounds_data[i]

            inside_X = np.all((self.Xembnp[:, chosen_columns] >= lower_bounds) & (self.Xembnp[:, chosen_columns] <= upper_bounds), axis=1)
            inside_X_prime = np.all((self.Yempnp[:, chosen_columns] >= lower_bounds) & (self.Yempnp[:, chosen_columns] <= upper_bounds), axis=1)
            
            count_X = np.sum(inside_X)/nx
            count_X_prime = np.sum(inside_X_prime)/nxprime
            
            discrepancies.append(np.abs(count_X - count_X_prime))
            orig_counts.append(count_X)
        return {"mean_dist": np.mean(discrepancies), "orig": np.mean(orig_counts) }





    def generate_counting_query_data(self, num_queries, s):
        columns_data = []
        bounds_data = []
        lower_bound = 0.05
        upper_bound = 0.95

        for _ in range(num_queries):
            
            while True:
                # Generate random columns and bounds
                chosen_columns = np.random.choice(self.Xembnp.shape[1], s, replace=False)
                lower_bounds =np.zeros(s)
                upper_bounds = np.zeros(s)
                for i,column in enumerate(chosen_columns):
                    #generate a random real number between 0 and 1 
                    lb = np.random.uniform(0,1)
                    ub = np.random.uniform(lb,1)
                        
                    lower_bounds[i] = lb
                    upper_bounds[i] = ub


                # Calculate count_X for the selected columns
                inside_X = np.all((self.Xembnp[:, chosen_columns] >= lower_bounds) & (self.Xembnp[:, chosen_columns] <= upper_bounds), axis=1)
                count_X = np.sum(inside_X) / self.Xembnp.shape[0]
                # Check if count_X is within the desired range
                if lower_bound <= count_X <= upper_bound:
                    columns_data.append(chosen_columns)
                    bounds_data.append((lower_bounds, upper_bounds))
                    print("found combination!")
                    break  # Accept the generated query
                else:
                    # Reject and regenerate if not within the desired range
                    continue
        
        return columns_data, bounds_data


    def generate_thresholding_data(self, num_hyperplanes, num_dimensions):
        dimensions = np.random.choice(self.Xembnp.shape[1], num_dimensions, replace=False)
        w_data = [np.random.randn(num_dimensions) for _ in range(num_hyperplanes)]
        b_data = [np.random.uniform() for _ in range(num_hyperplanes)]
        return dimensions, w_data, b_data

    def _get_or_save_random_data(self, filename, generator_func, *args):
        folder_path = os.path.join(self.main_path, 'evaluation', self.dataset)
        file_path = os.path.join(folder_path, filename)

        if os.path.exists(file_path):
            print(f"use existing data from the path {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else: 
            print(f"create new data under the path {file_path}")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            data = generator_func(*args)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)        
        return data


    def evaluate(self, metrics_to_evaluate):

        raw_stats ={}
        for metric in metrics_to_evaluate:
            if metric == "cov_fixed":
                raw_stats[metric] = self.correlation_difference()
            elif metric == "rand_thresholding_query":  
                raw_stats[metric] = self.random_thresholding()
            elif metric == "rand_counting_query":  
                raw_stats[metric] = self.random_counting_queries()  
             
            elif "Way" in metric:
                #first extract the workload....
                    # Extract k from the key
                k = int(metric[-4])
                # Generate all k-combinations of the column names of the DataFrame
                workload = list(combinations(self.X.df.columns, k))
                # Store the combinations in the new dictionary under the corresponding key
                if "wdist" in metric:
                    # the metric is always of the form ..._wdist_p_..., extract p!
                    # we take the number directly after wdist_
                    #find a smart way to do this
                    p = int(metric[metric.find("wdist_")+6])

                    raw_stats[metric] = self.swd(workload, p=p )
                elif "newl" in metric:
                 # the metric is always of the form newl1..., extract p!
                    # we take the number directly after newl
                    #find a smart way to do this
                    p = int(metric[metric.find("newl")+4])
                    raw_stats[metric] = self.lp_distances(workload, p=p )
            

        expanded_stats = {}
        for key, value in raw_stats.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    expanded_key = f"{key}_{subkey}"
                    expanded_stats[expanded_key] = subvalue
            else:
                expanded_stats[key] = value
                
        return expanded_stats














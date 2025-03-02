import torch
        

class SpikeCount():
    """
    Sampler that calculates the spike count from spike activity.
    """

    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return f"{type(self).__name__}()"
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X, y=None):
        """
        Calculates the spike count from spike activity.

        Parameters:
            X: Spike activity tensor or array-like object.

        Returns:
            State vectors (spike count for each sample).
        """
        return X.sum(axis=1)
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def get_params(self, deep=None):
        return self.__dict__
    
    def set_params(self, **params):
        pass

class DeSNN():
    """
    DeSNN Sampler to sample state vectors based on spike activity.
    """

    def __init__(self, alpha=1, mod=0.4, drift=0.25):
        """
        Initializes the DeSNN sampler.

        Parameters:
            alpha: Maximum initial weight (default: 1).
            mod: Modulation factor for importance of the order of spikes (default: 0.4).
            drift: Drift factor (default: 0.25).
        """
        self.alpha = alpha
        self.mod = mod
        self.drift = drift

    def __repr__(self):
        return f"{type(self).__name__}(alpha={self.alpha}, mod={self.mod}, drift={self.drift})"
    
    def __sklearn_clone__(self):
        return self
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X, y=None):
        """
        Calculates ranks based on spike activity.

        Parameters:
            X: Spike activity tensor or array-like object.

        Returns:
            State vectors (DeSNN).
        """
        # Find the index of the first non-zero element along the second axis
        first_spike = (X != 0).int().argmax(axis=1)

        # Calculate initial ranks based on the alpha and mod parameters
        initial_ranks = self.alpha * (self.mod ** first_spike)

        # Calculate drift_up by multiplying spike activity along the second axis
        up = self.drift * X.sum(axis=1)

        # Calculate drift_down by multiplying by no spikes
        down = self.drift * (X.shape[1] - X.sum(axis=1))

        # Return the result of the rank calculation
        return initial_ranks + (up - down)
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def get_params(self, deep=None):
        return {"alpha": self.alpha, "mod": self.mod, "drift": self.drift}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
class DeSNNt():
    """
    DeSNN Sampler implementation based on https://doi.org/10.1007/s11063-020-10322-8,
    to sample state vectors based on spike activity.
    """

    def __init__(self, mod=0.4, drift=0.25):
        """
        Initializes the DeSNN sampler.

        Parameters:
            mod: Modulation factor for importance of the order of spikes (default: 0.4).
            drift: Drift factor (default: 0.25).
        """
        self.mod = mod
        self.drift = drift

    def __repr__(self):
        return f"{type(self).__name__}(mod={self.mod}, drift={self.drift})"
    
    def __sklearn_clone__(self):
        return self
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X, y=None):
        """
        Calculates ranks based on spike activity.

        Parameters:
            X: Spike activity tensor or array-like object. (n_samples, n_time, n_neurons)

        Returns:
            State vectors (DeSNN).
        """
        
        state_vectors = torch.zeros((X.shape[0], X.shape[2]))
    
        for sample in range(X.shape[0]):
            fire_mat = torch.zeros(X.shape[2])
            w_mat = torch.zeros(X.shape[2])

            for time_step in range(X.shape[1]):
                activity = X[sample, time_step]
                
                inverse_fire_mat = 1. - fire_mat
                
                already_fired = torch.logical_and(fire_mat, activity).to(dtype=torch.float32)
                first_time_firing = torch.logical_and(inverse_fire_mat, activity).to(dtype=torch.float32)
                
                
                for idx in first_time_firing.nonzero().reshape(-1):
                    w_mat[idx] = self.mod ** fire_mat.sum()
                    fire_mat[idx] = 1.
                
                w_mat[already_fired.nonzero().reshape(-1)] += self.drift
                
                w_mat = torch.where(X[sample, time_step] == 0., w_mat - self.drift, w_mat)

            state_vectors[sample] = w_mat 
            
        return state_vectors
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def get_params(self, deep=None):
        return {"mod": self.mod, "drift": self.drift}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

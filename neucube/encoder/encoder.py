import torch
from abc import ABC, abstractmethod

class Encoder(ABC):
  def __init__(self):
    super().__init__()
    
  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    return self.encode_dataset(X)

  def fit_transform(self, X, y=None):
    return self.fit(X, y).transform(X, y)

  def encode_dataset(self, dataset):
    """
    Encodes a dataset using the implemented encoding technique.

    Args:
      dataset (torch.Tensor): Input dataset to be encoded.

    Returns:
      torch.Tensor: Encoded dataset with spike patterns.
    """
    encoded_data = torch.zeros_like(dataset)

    for i in range(dataset.shape[0]):
      for j in range(dataset.shape[2]):
        encoded_data[i][:, j] = self.encode(dataset[i][:, j])

    return encoded_data

    
  @abstractmethod
  def encode(self, X_in):
    """
    Encodes an input sample using the implemented encoding technique.

    Args:
      X_in (torch.Tensor): Input sample to be encoded.

    Returns:
      torch.Tensor: Encoded spike pattern for the input sample.
    """
    pass

class Delta(Encoder):
  def __init__(self, threshold=0.1, neg=False):
    """
    Initializes the Delta encoder with a threshold value.

    Args:
      threshold (float, optional): Threshold value for spike generation. Defaults to 0.1.
      neg (bool, optional): Flag to control the generate spike trains, excitatory only vs excitatory and inhibitory. Defaults to False.
    """
    self.threshold = threshold
    self.neg = neg
  
  def __repr__(self):
    return f"{type(self).__name__}(Threshold={self.threshold}, Negative Spikes={self.neg})"
    
  def __sklearn_clone__(self):
    return self

  def encode(self, sample):
    """
    Encodes an input sample using delta encoding.

    Delta encoding compares each element in the sample with its previous element,
    and if the difference exceeds the threshold, it generates a spike (1); otherwise, no spike (0).

    Args:
      sample (torch.Tensor): Input sample to be encoded.

    Returns:
      torch.Tensor: Encoded spike train for the input sample.
    """
    aux = torch.cat((sample[0].unsqueeze(0), sample))[:-1]
    
    diff_X = sample - aux
    
    exci_spikes = torch.ones_like(sample) * (diff_X > self.threshold)
    
    if self.neg:
      inhi_spikes = torch.ones_like(sample) * (-diff_X > self.threshold)
      spikes = exci_spikes - inhi_spikes
    else:
      spikes = exci_spikes
            
    return spikes

  def get_params(self, deep=None):
      return {"threshold": self.threshold, "neg": self.neg}
  
  def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self

class TBR():
  def __init__(self, threshold=0.5, neg=False):
    """
    Initializes the Address Event Representation (Threshold Based Representation) encoder with a threshold scale value.

    Args:
      threshold (float, optional): Threshold scale value for value threshold calculation. Defaults to 0.5.
      neg (bool, optional): Flag to control the generate spike trains, excitatory only vs excitatory and inhibitory. Defaults to False. 
    """
    self.threshold = threshold
    self.neg = neg
    self.tbr_array = None
  
  def __repr__(self):
    return f"{type(self).__name__}(threshold={self.threshold}, neg={self.neg})"
  
  def __sklearn_clone__(self):
    return self
  
  def value_threshold(self, diff_X):
    std, mean = torch.std_mean(diff_X)
    V_t = mean + (std * self.threshold)
    return V_t
  
  def diff_signal(self, sample, fit=False):
    
    aux = torch.cat((sample[0].unsqueeze(0), sample))[:-1]
    if fit:
      diff_X = torch.abs(sample - aux)
    else:
      diff_X = sample - aux
      
    return diff_X
  
  def fit(self, X, y=None):
    
    tbr_array = torch.zeros(X.shape[2], X.shape[0])
    for i in range(X.shape[0]):
      for k in range(X.shape[2]):
        sample = X[i, :, k]
        diff_X = self.diff_signal(sample, fit=True)
        V_t = self.value_threshold(diff_X)
        tbr_array[k, i] = V_t

    self.tbr_array = torch.mean(tbr_array, 1)
    return self
  
  def transform(self, X, y=None):
    
    if self.tbr_array is None:
      raise Exception("Please fit the encoder using the fit method first.")
    
    spike_trains = torch.zeros_like(X)
    
    for i in range(X.shape[0]):
      for k in range(X.shape[2]):
        sample = X[i, :, k]
        k_spike_train = self.encode(sample, self.tbr_array[k])
        spike_trains[i, :, k] = k_spike_train
    
    return spike_trains

  def fit_transform(self, X, y=None):
    return self.fit(X, y).transform(X, y)

  def encode(self, sample, V_t):
    """
    Encodes an input sample using delta encoding.

    TBR encoding compares each element in the sample with its previous element,
    and if the difference exceeds the value threshold, it generates a spike (1); otherwise, no spike (0).
    
    Args:
      sample (torch.Tensor): Input sample to be encoded.

    Returns:
      torch.Tensor: Encoded spike train for the input sample.
    """
    diff_X = self.diff_signal(sample)
    
    
    exci_spikes = torch.ones_like(diff_X) * (diff_X > V_t)
    
    if self.neg:
      inhi_spikes = torch.ones_like(diff_X) * (-diff_X > V_t)
      spike_train = exci_spikes - inhi_spikes
    else:
      spike_train = exci_spikes
    return spike_train
  
  def get_params(self, deep=None):
      return {"threshold": self.threshold, "neg": self.neg}
  
  def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      self.tbr_array = None
      return self
    

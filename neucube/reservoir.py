import torch
from tqdm import tqdm
import math
from .topology import small_world_connectivity, SWC, small_world_connectivity_test
from .utils import print_summary
from .training import STDP
from .training import NRDP

class Reservoir():
  def __init__(self, cube_shape=(10,10,10), inputs=14, coordinates=None, mapping=None, swc=0, swr=0.15, c=0.4, l=0.169, c_in=0.9, l_in=1.2, leak_rate=0.002, mem_thr=0.1, refractory_period=5, learning_rule=STDP(), n_train=1, device=None):
    """
    Initializes the reservoir object.

    Parameters:
        cube_shape (tuple): Dimensions of the reservoir as a 3D cube (default: (10,10,10)).
        inputs (int): Number of input features.
        coordinates (torch.Tensor): Coordinates of the neurons in the reservoir.
                                    If not provided, the coordinates are generated based on `cube_shape`.
        mapping (torch.Tensor): Coordinates of the input neurons, must be a subset of reservoir's neurons.
                                If not provided, random connectivity is used.
        swc (int): Parameter controlling which Small World Connectivity mode is used. (0 for probabilistic, random int for non-probabilistic)
        swr (float): Parameter controlling the Small World Radius (SWR) when using non-probabilistic SWC.
        c (float): Parameter controlling the connection probability of the reservoir when using probabilistic SWC.
        l (float): Parameter controlling the SWR of the reservoir when using probabilistic SWC.
        c_in (float): Parameter controlling the connection probability of the input neurons when using probabilistic SWC.
        l_in (float): Parameter controlling the SWR of the input neurons when using probabilistic SWC.
        leak_rate (float): Parameter that controls the leak rate of the LIF neuron.
        mem_thr (float): Action-potential threshold for spike generation.
        refractory_period (int): Refractory period after a spike.
        learning_rule (LearningRule): The learning rule implementation to use for training.
        n_train (int): Number of unsupervised STDP learning iterations.
        device (str): String to indicate which device to use ('cuda', 'cpu', 'mps', defaults to 'cpu' if None).
    """
    
    self.cube_shape = cube_shape
    self.inputs = inputs
    self.coordinates = coordinates
    self.mapping = mapping
    self.swc = swc
    self.swr = swr
    self.c = c
    self.l = l
    self.c_in = c_in
    self.l_in = l_in
    self.leak_rate = leak_rate
    self.mem_thr = mem_thr
    self.refractory_period = refractory_period
    self.learning_rule = learning_rule
    self.n_train = n_train
    self.device = device
    
    self.__init_model()

  def __repr__(self):
    return f"{type(self).__name__}(cube_shape={self.cube_shape}, inputs={self.inputs}, swc={self.swc}, swr={self.swr}, c={self.c}, l={self.l}, c_in={self.c_in}, l={self.l_in}, leak_rate={self.leak_rate}, mem_thr={self.mem_thr}, refractory_period={self.refractory_period}, learning_rule={self.learning_rule}, n_train={self.n_train})"
  
  def __init_model(self):
    self.trained_ = False
    self.device_ = torch.device(self.device if (self.device != None) else 'cpu')
    
    if self.coordinates is None:
      self.n_neurons_ = math.prod(self.cube_shape)
      x, y, z = torch.meshgrid(torch.linspace(0, 1, self.cube_shape[0]), torch.linspace(0, 1, self.cube_shape[1]), torch.linspace(0, 1, self.cube_shape[2]), indexing='xy')
      self.pos_ = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).to(self.device_)
    else:
      if self.coordinates.device != self.device_:
        self.coordinates = torch.tensor(self.coordinates).to(self.device_)
      self.n_neurons_ = self.coordinates.shape[0]
      self.pos_ = self.coordinates

    dist = torch.cdist(self.pos_, self.pos_)
    if self.swc == 0:
      conn_mat = small_world_connectivity_test(dist, self.c, self.l) / 100
    else:
      conn_mat = SWC(dist, swr=self.swr)
    inh_n = torch.randint(self.n_neurons_, size=(int(self.n_neurons_*0.2),))
    conn_mat[:, inh_n] = -conn_mat[:, inh_n]

    if self.mapping is None:
      input_conn = torch.where(torch.rand(self.n_neurons_, self.inputs) > 0.95, torch.ones_like(torch.rand(self.n_neurons_, self.inputs)), torch.zeros(self.n_neurons_, self.inputs)) / 50
    else:
      if self.mapping.device != self.device_:
        self.mapping = torch.tensor(self.mapping).to(self.device_)
      dist_in = torch.cdist(self.pos_, self.mapping, p=2)
      if self.swc == 0:
        input_conn = small_world_connectivity_test(dist_in, self.c_in, self.l_in) / 50
      else:
        input_conn = 2 * SWC(dist_in, self.swr, input_n=True)

    self.w_latent_ = conn_mat.to(self.device_)
    self.w_in_ = input_conn.to(self.device_)

  # we would like clones to maintain the state of the reservoir
  # this is so that cloning a reservoir does not change the current state of the unsupervised learning
  def __sklearn_clone__(self):
    return self
  
  def _fit(self, X, y=None, verbose=True):
    self.batch_size_, self.n_time_, self.n_features_ = X.shape

    self.learning_rule.setup(self.device_, self.n_neurons_)

    for sample in tqdm(range(X.shape[0]), disable = not verbose, desc="Fitting SNNr to input data..."):

      spike_latent = torch.zeros(self.n_neurons_).to(self.device_)
      mem_poten = torch.zeros(self.n_neurons_).to(self.device_)
      refrac = torch.ones(self.n_neurons_).to(self.device_)
      refrac_count = torch.zeros(self.n_neurons_).to(self.device_)
      spike_times = torch.zeros(self.n_neurons_).to(self.device_)


      self.learning_rule.per_sample(sample)

      for activity in range(self.n_time_):
        spike_in = X[sample, activity, :]
        spike_in = spike_in.to(self.device_)

        refrac[refrac_count < 1] = 1

        I = torch.sum(self.w_in_*spike_in, axis=1)+torch.sum(self.w_latent_*spike_latent, axis=1)
        mem_poten = mem_poten*torch.exp(torch.tensor(-(1/400)))*(1-spike_latent)+(refrac*I)

        spike_latent[mem_poten >= self.mem_thr] = 1
        spike_latent[mem_poten < self.mem_thr] = 0

        refrac[mem_poten >= self.mem_thr] = 0
        refrac_count[mem_poten >= self.mem_thr] = self.refractory_period
        refrac_count = refrac_count-1


        self.learning_rule.per_time_slice(sample, activity)
        pre_updates, pos_updates = self.learning_rule.train(activity-spike_times, self.w_latent_, spike_latent)
        self.w_latent_ += pre_updates
        self.w_latent_ += pos_updates
        self.learning_rule.reset()

        spike_times[mem_poten >= self.mem_thr] = activity
    
    self.trained_ = True
    
    return self
  
  def fit(self, X, y=None, verbose=True):
    """
      Fits the reservoir to the input data using a specified learning rule.

      Parameters:
          X (torch.Tensor): Input data of shape (batch_size, n_time, n_features).
          verbose (bool): Flag indicating whether to display progress during simulation.

      Returns:
          Reservoir: An instance of the fitted reservoir.
    """
    if self.trained_ == True:
      return self
    elif self.trained_ == False:
      for i in range(self.n_train):
        self._fit(X, y, verbose)
      self.trained_ = True
    return self
  
  def transform(self, X, y=None, verbose=True):
    """
    Generates a spike record for the given input data.

    Parameters:
        X (torch.Tensor): Input data of shape (batch_size, n_time, n_features).
        verbose (bool): Flag indicating whether to display progress during simulation.

    Returns:
        torch.Tensor: Spike activity of the reservoir neurons over time, of shape (batch_size, n_time, n_neurons).
    """
      

    self.batch_size_, self.n_time_, self.n_features_ = X.shape

    spike_rec = torch.zeros(self.batch_size_, self.n_time_, self.n_neurons_)

    for sample in tqdm(range(X.shape[0]), disable = not verbose, desc="Simulating data in SNNr..."):

      spike_latent = torch.zeros(self.n_neurons_).to(self.device_)
      mem_poten = torch.zeros(self.n_neurons_).to(self.device_)
      refrac = torch.ones(self.n_neurons_).to(self.device_)
      refrac_count = torch.zeros(self.n_neurons_).to(self.device_)
      spike_times = torch.zeros(self.n_neurons_).to(self.device_)

      for activity in range(self.n_time_):
        spike_in = X[sample, activity, :]
        spike_in = spike_in.to(self.device_)

        refrac[refrac_count < 1] = 1

        I = torch.sum(self.w_in_*spike_in, axis=1)+torch.sum(self.w_latent_*spike_latent, axis=1)
        mem_poten = mem_poten*torch.exp(torch.tensor(-(self.leak_rate)))*(1-spike_latent)+(refrac*I)

        spike_latent[mem_poten >= self.mem_thr] = 1
        spike_latent[mem_poten < self.mem_thr] = 0

        refrac[mem_poten >= self.mem_thr] = 0
        refrac_count[mem_poten >= self.mem_thr] = self.refractory_period
        refrac_count = refrac_count-1

        spike_times[mem_poten >= self.mem_thr] = activity
        
        spike_rec[sample,activity,:] = spike_latent

    return spike_rec
  
  def fit_transform(self, X, y=None):
    """
    Fits data to the reservoir and returns spike record.

    Parameters:
        X : {torch.Tensor} Input data of shape (batch_size, n_time, n_features)
        y : array-like labels for supervised learning of shape (batch_size,), default=None
    Returns:
        spike_rec : {torch.Tensor} Spike activity of the reservoir neurons over time, of shape (batch_size, n_time, n_neurons).
    """
    
    spike_rec = self.fit(X, y).transform(X, y)
    return spike_rec
  
  def summary(self):
    """
    Prints a summary of the reservoir.
    """
    res_info = [["Neurons", str(self.n_neurons_)],
                ["Reservoir connections", str(sum(sum(self.w_latent_ != 0)).item())],
                ["Input connections", str(self.inputs)],
                ["Device", str(self.device_)]]

    print_summary(res_info)
    
  def get_params(self, deep=None):
    return {"cube_shape": self.cube_shape, "inputs": self.inputs, "coordinates": self.coordinates, "mapping": self.mapping, "swc": self.swc, "swr": self.swr, "c": self.c, "l": self.l, "c_in": self.c_in, "l_in": self.l_in, "leak_rate": self.leak_rate, "mem_thr": self.mem_thr, "refractory_period": self.refractory_period, "learning_rule": self.learning_rule, "n_train": self.n_train}
  
  def set_params(self, **params):
    for parameter, value in params.items():
        setattr(self, parameter, value)
    self.__init_model()
    return self

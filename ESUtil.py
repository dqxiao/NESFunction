import numpy as np 
from sklearn.neighbors import NearestNeighbors


def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)



def compute_novelty_obs(obs,K=10):
  """
  Compute KNN distance function 
  """
  nbrs=NearestNeighbors(n_neighbors=K, algorithm='auto',metric='cosine').fit(obs) #
  distances,_=nbrs.kneighbors(obs)
  spareness = np.sum(distances,axis=1)

  #print(spareness)

  return spareness 
from copy import deepcopy
import numpy as np
import scipy.stats as ss
from scipy.special import logit

from turbo_1 import Turbo1
from utils import from_unit_cube, latin_hypercube, to_unit_cube

from skopt.space import Categorical, Integer, Real

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


class TurboOptimizer(AbstractOptimizer):
    primary_import = "Turbo"

    def __init__(self, api_config, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.dimensions, self.vars_types, self.param_list = TurboOptimizer.get_sk_dimensions(api_config)
        print("dimensions: {}".format(self.dimensions))
        print("vars_types: {}".format(self.vars_types))
        # names of variables
        print("param_list: {}".format(self.param_list))

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.dim = len(self.bounds)
        print("lb: {}".format(self.lb))
        print("ub: {}".format(self.ub))
        print("dim: {}".format(self.dim))

        if "max_depth" in self.param_list:
            print("DT or RF")
            # max_depth
            att = "max_depth"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 10
            self.ub[att_idx] = 15
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))

            # max_features
            att = "max_features"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = logit(0.9)
            self.ub[att_idx] = logit(0.99)
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))

            # min_impurity_decrease
            att = "min_impurity_decrease"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 1e-5
            self.ub[att_idx] = 1e-4
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
        if "beta_1" in self.param_list and "hidden_layer_sizes" in self.param_list:
            print("MLP-adam")
            # batch_size
            att = "batch_size"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 16
            self.ub[att_idx] = 128
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            # hidden_layer_sizes
            att = "hidden_layer_sizes"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 64
            self.ub[att_idx] = 200
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            # validation_fraction
            att = "validation_fraction"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = logit(0.1)
            self.ub[att_idx] = logit(0.2)
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
        if "momentum" in self.param_list and "hidden_layer_sizes" in self.param_list:
            print("MLP-sgd")
            # batch_size
            att = "batch_size"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 16
            self.ub[att_idx] = 128
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            # hidden_layer_sizes
            att = "hidden_layer_sizes"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 64
            self.ub[att_idx] = 200
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            # validation_fraction
            att = "validation_fraction"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = logit(0.1)
            self.ub[att_idx] = logit(0.2)
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
        if "C" in self.param_list and "gamma" in self.param_list:
            print("SVM")
            # C
            att = "C"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = np.log(1e0)
            self.ub[att_idx] = np.log(1e3)
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            # tol
            att = "tol"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = np.log(1e-3)
            self.ub[att_idx] = np.log(1e-1)
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
        if "learning_rate" in self.param_list and "n_estimators" in self.param_list:
            print("ada")
            # n_estimators
            att = "n_estimators"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 30
            self.ub[att_idx] = 100
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
        if "n_neighbors" in self.param_list:
            print("kNN")
            # n_neighbors
            att = "n_neighbors"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 1
            self.ub[att_idx] = 15
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            # p
            att = "p"
            print("att: {}".format(att))
            att_idx = self.param_list.index(att)
            print("old lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))
            self.lb[att_idx] = 1
            self.ub[att_idx] = 2
            print("new lb: {}, ub: {}".format(self.lb[att_idx], self.ub[att_idx]))

        print("new_lb: {}".format(self.lb))
        print("new_ub: {}".format(self.ub))

        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []

        self.turbo = Turbo1(
            f=None,
            lb=self.lb,
            ub=self.ub,
            n_init=2 * self.dim + 1,
            max_evals=self.max_evals,
            batch_size=1,  # We need to update this later
            verbose=False,
        )

        # count restart
        self.cnt_restart = 0
        # use smaller length_min
        self.turbo.length_min = 0.5 ** 4
        # use distance between batch elements
        self.turbo.ele_distance = 1e-2

    # Obtain the search space configurations
    @staticmethod
    def get_sk_dimensions(api_config, transform="normalize"):
        """Help routine to setup skopt search space in constructor.

        Take api_config as argument so this can be static.
        """
        # The ordering of iteration prob makes no difference, but just to be
        # safe and consistnent with space.py, I will make sorted.
        param_list = sorted(api_config.keys())

        sk_types = []
        sk_dims = []
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]
            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # Some setup for case that whitelist of values is provided:
            values_only_type = param_type in ("cat", "ordinal")
            if (param_values is not None) and (not values_only_type):
                assert param_range is None
                param_values = np.unique(param_values)
                param_range = (param_values[0], param_values[-1])
            if param_type == "int":
                # Integer space in sklearn does not support any warping => Need
                # to leave the warping as linear in skopt.
                sk_dims.append(Integer(param_range[0], param_range[-1], transform=transform, name=param_name))
            elif param_type == "bool":
                assert param_range is None
                assert param_values is None
                sk_dims.append(Integer(0, 1, transform=transform, name=param_name))
            elif param_type in ("cat", "ordinal"):
                assert param_range is None
                # Leave x-form to one-hot as per skopt default
                sk_dims.append(Categorical(param_values, name=param_name))
            elif param_type == "real":
                # Skopt doesn't support all our warpings, so need to pick
                # closest substitute it does support.
                # prior = "log-uniform" if param_space in ("log", "logit") else "uniform"
                if param_space == "log":
                    prior = "log-uniform"
                elif param_space == "logit":
                    prior = "logit-uniform"
                else:
                    prior = "uniform"
                sk_dims.append(
                    Real(param_range[0], param_range[-1], prior=prior, transform=transform, name=param_name))
            else:
                assert False, "type %s not handled in API" % param_type
            sk_types.append(param_type)
        return sk_dims, sk_types, param_list

    def restart(self):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        X_init = latin_hypercube(self.turbo.n_init, self.dim)
        self.X_init = from_unit_cube(X_init, self.lb, self.ub)

    def suggest(self, n_suggestions=1):
        if self.batch_size is None:  # Remember the batch size on the first call to suggest
            self.batch_size = n_suggestions
            self.turbo.batch_size = n_suggestions
            self.turbo.failtol = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))
            self.turbo.n_init = max([self.turbo.n_init, self.batch_size])
            self.cnt_restart = self.cnt_restart + 1
            self.restart()

        X_next = np.zeros((n_suggestions, self.dim))

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        print("n_adapt: {}, n_suggestions: {}, n_init: {}".format(n_adapt, n_suggestions, n_init))
        if n_adapt > 0:
            if len(self.turbo._X) > 0:  # Use random points if we can't fit a GP
                print("running Turbo...")
                X = to_unit_cube(deepcopy(self.turbo._X), self.lb, self.ub)
                fX = copula_standardize(deepcopy(self.turbo._fX).ravel())  # Use Copula
                X_cand, y_cand, _ = self.turbo._create_candidates(
                    X, fX, length=self.turbo.length, n_training_steps=100, hypers={}
                )
                X_next[-n_adapt:, :] = self.turbo._select_candidates(X_cand, y_cand)[:n_adapt, :]
                X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.lb, self.ub)

        # Unwarp the suggestions
        suggestions = self.space_x.unwarp(X_next)
        return suggestions

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y)
        XX, yy = self.space_x.warp(X), np.array(y)[:, None]

        if len(self.turbo._fX) >= self.turbo.n_init:
            print("adjust region length")
            print("original region length: {}".format(self.turbo.length))
            self.turbo._adjust_length(yy)
            print("adjusted region length: {}".format(self.turbo.length))

        self.turbo.n_evals += self.batch_size

        self.turbo._X = np.vstack((self.turbo._X, deepcopy(XX)))
        self.turbo._fX = np.vstack((self.turbo._fX, deepcopy(yy)))
        self.turbo.X = np.vstack((self.turbo.X, deepcopy(XX)))
        self.turbo.fX = np.vstack((self.turbo.fX, deepcopy(yy)))

        ind_best = np.argmin(self.turbo.fX)
        f_best, x_best = self.turbo.fX[ind_best], self.turbo.X[ind_best, :]
        print("best f(x): {}, at x: {}".format(round(f_best[0], 2), np.around(x_best, 2)))
        print("x_best: {}".format(self.space_x.unwarp([x_best])))

        # Check for a restart
        print("turbo.length: {}, turbo.length_min: {}".format(self.turbo.length, self.turbo.length_min))
        if self.turbo.length < self.turbo.length_min:
            self.cnt_restart = self.cnt_restart + 1
            self.restart()
            print("original new region length: {}".format(self.turbo.length))
            # already exploit current region (current_length < length_min)
            # try new region but smaller one
            self.turbo.length = round(self.turbo.length / self.cnt_restart, 1)
            print("reduced new region length: {}".format(self.turbo.length))


if __name__ == "__main__":
    experiment_main(TurboOptimizer)

import json
import os
import re
from importlib import import_module
from io import open as ioopen
from warnings import warn

import numpy as np

from .analytical import DeltaFunction
from .base import Prior, Constraint
from .joint import JointPrior, BaseJointPriorDist
from ..utils import (
    logger,
    check_directory_exists_and_if_not_mkdir,
    BilbyJsonEncoder,
    decode_bilby_json,
)


class PriorDict(dict):
    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """A dictionary of priors

        Parameters
        ==========
        dictionary: Union[dict, str, None]
            If given, a dictionary to generate the prior set.
        filename: Union[str, None]
            If given, a file containing the prior to generate the prior set.
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            Default is no conversion.
        """
        super(PriorDict, self).__init__()
        if isinstance(dictionary, dict):
            self.from_dictionary(dictionary)
        elif type(dictionary) is str:
            logger.debug(
                'Argument "dictionary" is a string.'
                + " Assuming it is intended as a file name."
            )
            self.from_file(dictionary)
        elif type(filename) is str:
            self.from_file(filename)
        elif dictionary is not None:
            raise ValueError("PriorDict input dictionary not understood")
        self._cached_normalizations = {}

        self.convert_floats_to_delta_functions()

        if conversion_function is not None:
            self.conversion_function = conversion_function
        else:
            self.conversion_function = self.default_conversion_function

    def evaluate_constraints(self, sample):
        out_sample = self.conversion_function(sample)
        prob = 1
        for key in self:
            if isinstance(self[key], Constraint) and key in out_sample:
                prob *= self[key].prob(out_sample[key])
        return prob

    def default_conversion_function(self, sample):
        """
        Placeholder parameter conversion function.

        Parameters
        ==========
        sample: dict
            Dictionary to convert

        Returns
        =======
        sample: dict
            Same as input
        """
        return sample

    def to_file(self, outdir, label):
        """Write the prior distribution to file.

        Parameters
        ==========
        outdir: str
            output directory name
        label: str
            Output file naming scheme
        """

        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}.prior".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        joint_dists = []
        with open(prior_file, "w") as outfile:
            for key in self.keys():
                if JointPrior in self[key].__class__.__mro__:
                    distname = "_".join(self[key].dist.names) + "_{}".format(
                        self[key].dist.distname
                    )
                    if distname not in joint_dists:
                        joint_dists.append(distname)
                        outfile.write("{} = {}\n".format(distname, self[key].dist))
                    diststr = repr(self[key].dist)
                    priorstr = repr(self[key])
                    outfile.write(
                        "{} = {}\n".format(key, priorstr.replace(diststr, distname))
                    )
                else:
                    outfile.write("{} = {}\n".format(key, self[key]))

    def _get_json_dict(self):
        self.convert_floats_to_delta_functions()
        total_dict = {key: json.loads(self[key].to_json()) for key in self}
        total_dict["__prior_dict__"] = True
        total_dict["__module__"] = self.__module__
        total_dict["__name__"] = self.__class__.__name__
        return total_dict

    def to_json(self, outdir, label):
        check_directory_exists_and_if_not_mkdir(outdir)
        prior_file = os.path.join(outdir, "{}_prior.json".format(label))
        logger.debug("Writing priors to {}".format(prior_file))
        with open(prior_file, "w") as outfile:
            json.dump(self._get_json_dict(), outfile, cls=BilbyJsonEncoder, indent=2)

    def from_file(self, filename):
        """Reads in a prior from a file specification

        Parameters
        ==========
        filename: str
            Name of the file to be read in

        Notes
        =====
        Lines beginning with '#' or empty lines will be ignored.
        Priors can be loaded from:

        - bilby.core.prior as, e.g.,    :code:`foo = Uniform(minimum=0, maximum=1)`
        - floats, e.g.,                 :code:`foo = 1`
        - bilby.gw.prior as, e.g.,      :code:`foo = bilby.gw.prior.AlignedSpin()`
        - other external modules, e.g., :code:`foo = my.module.CustomPrior(...)`

        """

        comments = ["#", "\n"]
        prior = dict()
        with ioopen(filename, "r", encoding="unicode_escape") as f:
            for line in f:
                if line[0] in comments:
                    continue
                line.replace(" ", "")
                elements = line.split("=")
                key = elements[0].replace(" ", "")
                val = "=".join(elements[1:]).strip()
                prior[key] = val
        self.from_dictionary(prior)

    @classmethod
    def _get_from_json_dict(cls, prior_dict):
        try:
            class_ = getattr(
                import_module(prior_dict["__module__"]), prior_dict["__name__"]
            )
        except ImportError:
            logger.debug(
                "Cannot import prior module {}.{}".format(
                    prior_dict["__module__"], prior_dict["__name__"]
                )
            )
            class_ = cls
        except KeyError:
            logger.debug("Cannot find module name to load")
            class_ = cls
        for key in ["__module__", "__name__", "__prior_dict__"]:
            if key in prior_dict:
                del prior_dict[key]
        obj = class_(prior_dict)
        return obj

    @classmethod
    def from_json(cls, filename):
        """Reads in a prior from a json file

        Parameters
        ==========
        filename: str
            Name of the file to be read in
        """
        with open(filename, "r") as ff:
            obj = json.load(ff, object_hook=decode_bilby_json)

        # make sure priors containing JointDists are properly handled and point
        # to the same object when required
        jointdists = {}
        for key in obj:
            if isinstance(obj[key], JointPrior):
                for name in obj[key].dist.names:
                    jointdists[name] = obj[key].dist
        # set dist for joint values so that they point to the same object
        for key in obj:
            if isinstance(obj[key], JointPrior):
                obj[key].dist = jointdists[key]

        return obj

    def from_dictionary(self, dictionary):
        jpdkwargs = {}
        for key in list(dictionary.keys()):
            val = dictionary[key]
            if isinstance(val, Prior):
                continue
            elif isinstance(val, (int, float)):
                dictionary[key] = DeltaFunction(peak=val)
            elif isinstance(val, str):
                cls = val.split("(")[0]
                args = "(".join(val.split("(")[1:])[:-1]
                try:
                    dictionary[key] = DeltaFunction(peak=float(cls))
                    logger.debug("{} converted to DeltaFunction prior".format(key))
                    continue
                except ValueError:
                    pass
                if "." in cls:
                    module = ".".join(cls.split(".")[:-1])
                    cls = cls.split(".")[-1]
                else:
                    module = __name__.replace(
                        "." + os.path.basename(__file__).replace(".py", ""), ""
                    )
                try:
                    cls = getattr(import_module(module), cls, cls)
                except ModuleNotFoundError:
                    logger.error(
                        "Cannot import prior class {} for entry: {}={}".format(
                            cls, key, val
                        )
                    )
                    raise
                if key.lower() in ["conversion_function", "condition_func"]:
                    setattr(self, key, cls)
                elif isinstance(cls, str):
                    if "(" in val:
                        raise TypeError("Unable to parse prior class {}".format(cls))
                    else:
                        continue
                elif issubclass(cls, BaseJointPriorDist):
                    dictionary.pop(key)
                    if key not in jpdkwargs:
                        jpdkwargs[key] = cls.from_repr(args)
                elif issubclass(cls, JointPrior):
                    jpkwargs = {
                        item[0].strip(): cls._parse_argument_string(item[1])
                        for item in cls._split_repr(
                            ", ".join(
                                [arg for arg in args.split(",") if "dist=" not in arg]
                            )
                        ).items()
                    }
                    keymatch = re.match(r"dist=(?P<distkey>\S+),", args)
                    if keymatch is None:
                        raise ValueError(
                            "'dist' argument for JointPrior is not specified"
                        )

                    if keymatch["distkey"] not in jpdkwargs:
                        raise ValueError(
                            f"BaseJointPriorDist {keymatch['distkey']} must be defined before {cls.__name__}"
                        )

                    jpkwargs["dist"] = jpdkwargs[keymatch["distkey"]]
                    dictionary[key] = cls(**jpkwargs)
                else:
                    try:
                        dictionary[key] = cls.from_repr(args)
                    except TypeError as e:
                        raise TypeError(
                            "Unable to parse prior, bad entry: {} "
                            "= {}. Error message {}".format(key, val, e)
                        )
            elif isinstance(val, dict):
                try:
                    _class = getattr(
                        import_module(val.get("__module__", "none")),
                        val.get("__name__", "none"),
                    )
                    dictionary[key] = _class(**val.get("kwargs", dict()))
                except ImportError:
                    logger.debug(
                        "Cannot import prior module {}.{}".format(
                            val.get("__module__", "none"), val.get("__name__", "none")
                        )
                    )
                    logger.warning(
                        "Cannot convert {} into a prior object. "
                        "Leaving as dictionary.".format(key)
                    )
                    continue
            else:
                raise TypeError(
                    "Unable to parse prior, bad entry: {} "
                    "= {} of type {}".format(key, val, type(val))
                )
        self.update(dictionary)

    def convert_floats_to_delta_functions(self):
        """Convert all float parameters to delta functions"""
        for key in self:
            if isinstance(self[key], Prior):
                continue
            elif isinstance(self[key], float) or isinstance(self[key], int):
                self[key] = DeltaFunction(self[key])
                logger.debug("{} converted to delta function prior.".format(key))
            else:
                logger.debug(
                    "{} cannot be converted to delta function prior.".format(key)
                )

    def fill_priors(self, likelihood=None, default_priors_file=None):
        """
        Any floats in prior will be converted to delta function prior.

        Parameters
        ==========
        likelihood: bilby.likelihood.GravitationalWaveTransient instance
            Used to infer the set of parameters to fill the prior with
        default_priors_file: str, optional
            If given, a file containing the default priors.


        Returns
        =======
        prior: dict
            The filled prior dictionary

        """
        if likelihood is not None:
            warn("Filling priors from likelihood parameters is deprecated", FutureWarning)
        if default_priors_file is not None:
            warn("Setting default priors from a defaults file is deprecated", FutureWarning)

        self.convert_floats_to_delta_functions()

        for key in self:
            self.test_redundancy(key)

    def sample(self, size=None):
        """Draw samples from the prior set

        Parameters
        ==========
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        dict: Dictionary of the samples
        """
        return self.sample_subset_constrained(keys=list(self.keys()), size=size)

    def sample_subset_constrained_as_array(self, keys=iter([]), size=None):
        """Return an array of samples

        Parameters
        ==========
        keys: list
            A list of keys to sample in
        size: int
            The number of samples to draw

        Returns
        =======
        array: array_like
            An array of shape (len(key), size) of the samples (ordered by keys)
        """
        samples_dict = self.sample_subset_constrained(keys=keys, size=size)
        samples_dict = {key: np.atleast_1d(val) for key, val in samples_dict.items()}
        samples_list = [samples_dict[key] for key in keys]
        return np.array(samples_list)

    def sample_subset(self, keys=iter([]), size=None):
        """Draw samples from the prior set for parameters which are not a DeltaFunction

        Parameters
        ==========
        keys: list
            List of prior keys to draw samples from
        size: int or tuple of ints, optional
            See numpy.random.uniform docs

        Returns
        =======
        dict: Dictionary of the drawn samples
        """
        self.convert_floats_to_delta_functions()
        samples = dict()
        for key in keys:
            if isinstance(self[key], Constraint):
                continue
            elif isinstance(self[key], Prior):
                samples[key] = self[key].sample(size=size)
            else:
                logger.debug("{} not a known prior.".format(key))
        return samples

    @property
    def non_fixed_keys(self):
        keys = self.keys()
        keys = [k for k in keys if isinstance(self[k], Prior)]
        keys = [k for k in keys if self[k].is_fixed is False]
        keys = [k for k in keys if k not in self.constraint_keys]
        return keys

    @property
    def fixed_keys(self):
        return [
            k for k, p in self.items() if (p.is_fixed and k not in self.constraint_keys)
        ]

    @property
    def constraint_keys(self):
        return [k for k, p in self.items() if isinstance(p, Constraint)]

    def sample_subset_constrained(self, keys=iter([]), size=None):
        """
        Sample a subset of priors while ensuring constraints are satisfied.

        Parameters
        ==========
        keys: list
            List of prior keys to sample from.
        size: int
            The number of samples to draw.

        Returns
        =======
        dict: Dictionary of valid samples.
        """
        if not any(isinstance(self[key], Constraint) for key in self):
            return self.sample_subset(keys=keys, size=size)

        efficiency_warning_was_issued = False

        def check_efficiency(n_tested, n_valid):
            nonlocal efficiency_warning_was_issued
            if efficiency_warning_was_issued:
                return
            efficiency = n_valid / float(n_tested)
            if n_tested >= 1e3 and efficiency < 1e-3:
                logger.warning("Prior sampling efficiency is very low, please verify its validity.")
                efficiency_warning_was_issued = True

        n_tested_samples, n_valid_samples = 0, 0
        if size is None or size == 1:
            while True:
                sample = self.sample_subset(keys=keys, size=size)
                is_valid = self.evaluate_constraints(sample)
                n_tested_samples += 1
                n_valid_samples += int(is_valid)
                check_efficiency(n_tested_samples, n_valid_samples)
                if is_valid:
                    return sample
        else:
            needed = np.prod(size)
            for key in keys.copy():
                if isinstance(self[key], Constraint):
                    del keys[keys.index(key)]
            all_samples = {key: np.array([]) for key in keys}
            _first_key = list(all_samples.keys())[0]
            while len(all_samples[_first_key]) < needed:
                samples = self.sample_subset(keys=keys, size=needed)
                keep = np.array(self.evaluate_constraints(samples), dtype=bool)
                for key in keys:
                    all_samples[key] = np.hstack(
                        [all_samples[key], samples[key][keep].flatten()]
                    )
                n_tested_samples += needed
                n_valid_samples += np.sum(keep)
                check_efficiency(n_tested_samples, n_valid_samples)
            all_samples = {
                key: np.reshape(all_samples[key][:needed], size) for key in keys
            }
            return all_samples

    def normalize_constraint_factor(
        self, keys, min_accept=10000, sampling_chunk=50000, nrepeats=10
    ):
        if len(self.constraint_keys) == 0:
            return 1
        elif keys in self._cached_normalizations.keys():
            return self._cached_normalizations[keys]
        else:
            factor_estimates = [
                self._estimate_normalization(keys, min_accept, sampling_chunk)
                for _ in range(nrepeats)
            ]
            factor = np.mean(factor_estimates)
            if np.std(factor_estimates) > 0:
                decimals = int(-np.floor(np.log10(3 * np.std(factor_estimates))))
                factor_rounded = np.round(factor, decimals)
            else:
                factor_rounded = factor
            self._cached_normalizations[keys] = factor_rounded
            return factor_rounded

    def _estimate_normalization(self, keys, min_accept, sampling_chunk):
        samples = self.sample_subset(keys=keys, size=sampling_chunk)
        keep = np.atleast_1d(self.evaluate_constraints(samples))
        if len(keep) == 1:
            self._cached_normalizations[keys] = 1
            return 1
        all_samples = {key: np.array([]) for key in keys}
        while np.count_nonzero(keep) < min_accept:
            samples = self.sample_subset(keys=keys, size=sampling_chunk)
            for key in samples:
                all_samples[key] = np.hstack([all_samples[key], samples[key].flatten()])
            keep = np.array(self.evaluate_constraints(all_samples), dtype=bool)
        factor = len(keep) / np.count_nonzero(keep)
        return factor

    def prob(self, sample, **kwargs):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.prod`

        Returns
        =======
        float: Joint probability of all individual sample probabilities

        """
        prob = np.prod([self[key].prob(sample[key]) for key in sample], **kwargs)

        return self.check_prob(sample, prob)

    def check_prob(self, sample, prob):
        ratio = self.normalize_constraint_factor(tuple(sample.keys()))
        if np.all(prob == 0.0):
            return prob * ratio
        else:
            if isinstance(prob, float):
                if self.evaluate_constraints(sample):
                    return prob * ratio
                else:
                    return 0.0
            else:
                constrained_prob = np.zeros_like(prob)
                in_bounds = np.isfinite(prob)
                subsample = {key: sample[key][in_bounds] for key in sample}
                keep = np.array(self.evaluate_constraints(subsample), dtype=bool)
                constrained_prob[in_bounds] = prob[in_bounds] * keep * ratio
                return constrained_prob

    def ln_prob(self, sample, axis=None, normalized=True):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which to calculate the log probability
        axis: None or int
            Axis along which the summation is performed
        normalized: bool
            When False, disables calculation of constraint normalization factor
            during prior probability computation. Default value is True.

        Returns
        =======
        float or ndarray:
            Joint log probability of all the individual sample probabilities

        """
        ln_prob = np.sum([self[key].ln_prob(sample[key]) for key in sample], axis=axis)
        return self.check_ln_prob(sample, ln_prob,
                                  normalized=normalized)

    def check_ln_prob(self, sample, ln_prob, normalized=True):
        if normalized:
            ratio = self.normalize_constraint_factor(tuple(sample.keys()))
        else:
            ratio = 1
        if np.all(np.isinf(ln_prob)):
            return ln_prob
        else:
            if isinstance(ln_prob, float):
                if self.evaluate_constraints(sample):
                    return ln_prob + np.log(ratio)
                else:
                    return -np.inf
            else:
                constrained_ln_prob = -np.inf * np.ones_like(ln_prob)
                in_bounds = np.isfinite(ln_prob)
                subsample = {key: sample[key][in_bounds] for key in sample}
                keep = np.log(np.array(self.evaluate_constraints(subsample), dtype=bool))
                constrained_ln_prob[in_bounds] = ln_prob[in_bounds] + keep + np.log(ratio)
                return constrained_ln_prob

    def cdf(self, sample):
        """Evaluate the cumulative distribution function at the provided points

        Parameters
        ----------
        sample: dict, pandas.DataFrame
            Dictionary of the samples of which to calculate the CDF

        Returns
        -------
        dict, pandas.DataFrame: Dictionary containing the CDF values

        """
        return sample.__class__(
            {key: self[key].cdf(sample) for key, sample in sample.items()}
        )

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ==========
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        =======
        list: List of floats containing the rescaled sample
        """
        return list(
            [self[key].rescale(sample) for key, sample in zip(keys, theta)]
        )

    def test_redundancy(self, key, disable_logging=False):
        """Empty redundancy test, should be overwritten in subclasses"""
        return False

    def test_has_redundant_keys(self):
        """
        Test whether there are redundant keys in self.

        Returns
        =======
        bool: Whether there are redundancies or not
        """
        redundant = False
        for key in self:
            if isinstance(self[key], Constraint):
                continue
            temp = self.copy()
            del temp[key]
            if temp.test_redundancy(key, disable_logging=True):
                logger.warning(
                    f"{key} is a redundant key in this {self.__class__.__name__}."
                )
                redundant = True
        return redundant

    def copy(self):
        """
        We have to overwrite the copy method as it fails due to the presence of
        defaults.
        """
        return self.__class__(dictionary=dict(self))


class PriorDictException(Exception):
    """General base class for all prior dict exceptions"""


class ConditionalPriorDict(PriorDict):
    def __init__(self, dictionary=None, filename=None, conversion_function=None):
        """

        Parameters
        ==========
        dictionary: dict
            See parent class
        filename: str
            See parent class
        """
        self._conditional_keys = []
        self._unconditional_keys = []
        self._rescale_keys = []
        self._rescale_indexes = []
        self._least_recently_rescaled_keys = []
        super(ConditionalPriorDict, self).__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=conversion_function,
        )
        self._resolved = False
        self._resolve_conditions()

    def _resolve_conditions(self):
        """
        Resolves how priors depend on each other and automatically
        sorts them into the right order.
        1. All unconditional priors are put in front in arbitrary order
        2. We loop through all the unsorted conditional priors to find
        which one can go next
        3. We repeat step 2 len(self) number of times to make sure that
        all conditional priors will be sorted in order
        4. We set the `self._resolved` flag to True if all conditional
        priors were added in the right order
        """
        self._unconditional_keys = [
            key for key in self.keys() if not hasattr(self[key], "condition_func")
        ]
        conditional_keys_unsorted = [
            key for key in self.keys() if hasattr(self[key], "condition_func")
        ]
        self._conditional_keys = []
        for _ in range(len(self)):
            for key in conditional_keys_unsorted[:]:
                if self._check_conditions_resolved(key, self.sorted_keys):
                    self._conditional_keys.append(key)
                    conditional_keys_unsorted.remove(key)

        self._resolved = True
        if len(conditional_keys_unsorted) != 0:
            self._resolved = False

    def _check_conditions_resolved(self, key, sampled_keys):
        """Checks if all required variables have already been sampled so we can sample this key"""
        conditions_resolved = True
        for k in self[key].required_variables:
            if k not in sampled_keys:
                conditions_resolved = False
        return conditions_resolved

    def sample_subset(self, keys=iter([]), size=None):
        self.convert_floats_to_delta_functions()
        add_delta_keys = [
            key
            for key in self.keys()
            if key not in keys and isinstance(self[key], DeltaFunction)
        ]
        use_keys = add_delta_keys + list(keys)
        subset_dict = ConditionalPriorDict({key: self[key] for key in use_keys})
        if not subset_dict._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolvable conditions."
            )
        samples = dict()
        for key in subset_dict.sorted_keys:
            if key not in keys or isinstance(self[key], Constraint):
                continue
            if isinstance(self[key], Prior):
                try:
                    samples[key] = subset_dict[key].sample(
                        size=size, **subset_dict.get_required_variables(key)
                    )
                except ValueError:
                    # Some prior classes can not handle an array of conditional parameters (e.g. alpha for PowerLaw)
                    # If that is the case, we sample each sample individually.
                    required_variables = subset_dict.get_required_variables(key)
                    samples[key] = np.zeros(size)
                    for i in range(size):
                        rvars = {
                            key: value[i] for key, value in required_variables.items()
                        }
                        samples[key][i] = subset_dict[key].sample(**rvars)
            else:
                logger.debug("{} not a known prior.".format(key))
        return samples

    def get_required_variables(self, key):
        """Returns the required variables to sample a given conditional key.

        Parameters
        ==========
        key : str
            Name of the key that we want to know the required variables for

        Returns
        =======
        dict: key/value pairs of the required variables
        """
        return {
            k: self[k].least_recently_sampled
            for k in getattr(self[key], "required_variables", [])
        }

    def prob(self, sample, **kwargs):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the probability of
        kwargs:
            The keyword arguments are passed directly to `np.prod`

        Returns
        =======
        float: Joint probability of all individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        prob = np.prod(res, **kwargs)
        return self.check_prob(sample, prob)

    def ln_prob(self, sample, axis=None, normalized=True):
        """

        Parameters
        ==========
        sample: dict
            Dictionary of the samples of which we want to have the log probability of
        axis: Union[None, int]
            Axis along which the summation is performed
        normalized: bool
            When False, disables calculation of constraint normalization factor
            during prior probability computation. Default value is True.

        Returns
        =======
        float: Joint log probability of all the individual sample probabilities

        """
        self._prepare_evaluation(*zip(*sample.items()))
        res = [
            self[key].ln_prob(sample[key], **self.get_required_variables(key))
            for key in sample
        ]
        ln_prob = np.sum(res, axis=axis)
        return self.check_ln_prob(sample, ln_prob,
                                  normalized=normalized)

    def cdf(self, sample):
        self._prepare_evaluation(*zip(*sample.items()))
        res = {
            key: self[key].cdf(sample[key], **self.get_required_variables(key))
            for key in sample
        }
        return sample.__class__(res)

    def rescale(self, keys, theta):
        """Rescale samples from unit cube to prior

        Parameters
        ==========
        keys: list
            List of prior keys to be rescaled
        theta: list
            List of randomly drawn values on a unit cube associated with the prior keys

        Returns
        =======
        list: List of floats containing the rescaled sample
        """
        keys = list(keys)
        theta = list(theta)
        self._check_resolved()
        self._update_rescale_keys(keys)
        result = dict()
        joint = dict()
        for key, index in zip(
            self.sorted_keys_without_fixed_parameters, self._rescale_indexes
        ):
            result[key] = self[key].rescale(
                theta[index], **self.get_required_variables(key)
            )
            self[key].least_recently_sampled = result[key]
            if isinstance(self[key], JointPrior) and self[key].dist.distname not in joint:
                joint[self[key].dist.distname] = [key]
            elif isinstance(self[key], JointPrior):
                joint[self[key].dist.distname].append(key)
        for names in joint.values():
            # this is needed to unpack how joint prior rescaling works
            # as an example of a joint prior over {a, b, c, d} we might
            # get the following based on the order within the joint prior
            # {a: [], b: [], c: [1, 2, 3, 4], d: []}
            # -> [1, 2, 3, 4]
            # -> {a: 1, b: 2, c: 3, d: 4}
            values = list()
            for key in names:
                values = np.concatenate([values, result[key]])
            for key, value in zip(names, values):
                result[key] = value

        def safe_flatten(value):
            """
            this is gross but can be removed whenever we switch to returning
            arrays, flatten converts 0-d arrays to 1-d so has to be special
            cased
            """
            if isinstance(value, (float, int)):
                return value
            else:
                return result[key].flatten()

        return [safe_flatten(result[key]) for key in keys]

    def _update_rescale_keys(self, keys):
        if not keys == self._least_recently_rescaled_keys:
            self._rescale_indexes = [
                keys.index(element)
                for element in self.sorted_keys_without_fixed_parameters
            ]
            self._least_recently_rescaled_keys = keys

    def _prepare_evaluation(self, keys, theta):
        self._check_resolved()
        for key, value in zip(keys, theta):
            self[key].least_recently_sampled = value

    def _check_resolved(self):
        if not self._resolved:
            raise IllegalConditionsException(
                "The current set of priors contains unresolveable conditions."
            )

    @property
    def conditional_keys(self):
        return self._conditional_keys

    @property
    def unconditional_keys(self):
        return self._unconditional_keys

    @property
    def sorted_keys(self):
        return self.unconditional_keys + self.conditional_keys

    @property
    def sorted_keys_without_fixed_parameters(self):
        return [
            key
            for key in self.sorted_keys
            if not isinstance(self[key], (DeltaFunction, Constraint))
        ]

    def __setitem__(self, key, value):
        super(ConditionalPriorDict, self).__setitem__(key, value)
        self._resolve_conditions()

    def __delitem__(self, key):
        super(ConditionalPriorDict, self).__delitem__(key)
        self._resolve_conditions()


class TransformedPriorView(Prior):
    """View of a transformed parameter prior.

    This proxy exposes a transformed parameter as a :class:`Prior` instance
    while delegating probability and sampling calculations back to the parent
    :class:`TransformedConditionalPriorDict`.
    """

    def __init__(
        self,
        parent,
        group_id,
        transformed_key,
        base_prior,
        index,
        definition,
        *,
        minimum=None,
        maximum=None,
        latex_label=None,
        unit=None,
    ):
        self._parent = parent
        self._group_id = group_id
        self._definition = definition
        self._index = index
        self._base_prior = base_prior
        self._native_keys = tuple(definition["native_keys"])
        self._transformed_keys = tuple(definition["transformed_keys"])
        inferred_minimum, inferred_maximum = self._infer_bounds(minimum, maximum)
        base_label = getattr(base_prior, "latex_label", transformed_key)
        base_unit = getattr(base_prior, "unit", None)
        super(TransformedPriorView, self).__init__(
            name=transformed_key,
            latex_label=latex_label or base_label,
            unit=unit if unit is not None else base_unit,
            minimum=inferred_minimum,
            maximum=inferred_maximum,
            check_range_nonzero=False,
            boundary=base_prior.boundary,
        )
        self._is_fixed = base_prior.is_fixed
        required = getattr(base_prior, "required_variables", [])
        transformed_required = []
        for var in required:
            transformed_required.extend(parent._transformed_keys_for_native(var))
        # Preserve order while removing duplicates
        seen = set()
        ordered = []
        for var in transformed_required:
            if var not in seen:
                ordered.append(var)
                seen.add(var)
        self.required_variables = ordered

    def _infer_bounds(self, minimum, maximum):
        if minimum is not None and maximum is not None:
            return minimum, maximum
        if len(self._native_keys) == 1:
            native_key = self._native_keys[0]
            try:
                result = self._parent._evaluate_forward(
                    self._definition, {native_key: self._base_prior.minimum}
                )
                lower = np.asarray(result[self.name])
                result = self._parent._evaluate_forward(
                    self._definition, {native_key: self._base_prior.maximum}
                )
                upper = np.asarray(result[self.name])
                return float(lower), float(upper)
            except Exception:
                pass
        return self._base_prior.minimum, self._base_prior.maximum

    def sample(self, size=None, **kwargs):
        samples = self._parent.sample_subset(keys=[self.name], size=size)
        return samples[self.name]

    def rescale(self, val, **kwargs):
        result = self._parent.rescale([self.name], [val])
        return result[0]

    def prob(self, val, **kwargs):
        sample = {self.name: val}
        sample.update(kwargs)
        self._require_group_values(sample)
        return self._parent.prob(sample)

    def ln_prob(self, val, axis=None, normalized=True, **kwargs):
        sample = {self.name: val}
        sample.update(kwargs)
        self._require_group_values(sample)
        return self._parent.ln_prob(sample, axis=axis, normalized=normalized)

    def cdf(self, val):
        if len(self._transformed_keys) != 1:
            raise NotImplementedError(
                "CDF evaluation for transformed parameter '{}' requires all {} values".format(
                    self.name, self._transformed_keys
                )
            )
        native = self._parent._evaluate_inverse(
            self._definition, {self.name: val}
        )[self._native_keys[0]]
        return self._base_prior.cdf(native)

    def _require_group_values(self, sample):
        missing = [
            key
            for key in self._transformed_keys
            if key not in sample and key != self.name
        ]
        if missing:
            raise KeyError(
                "Values for transformed keys {} are required to evaluate '{}'".format(
                    missing, self.name
                )
            )

    @property
    def least_recently_sampled(self):
        return self._parent._transformed_least_recently_sampled.get(self.name)

    @least_recently_sampled.setter
    def least_recently_sampled(self, value):
        self._parent._transformed_least_recently_sampled[self.name] = value


class TransformedConditionalPriorDict(ConditionalPriorDict):
    """A conditional prior dictionary that supports parameter transformations.

    This class augments :class:`ConditionalPriorDict` by allowing users to
    specify forward and inverse transformations between a native parameter
    space and a transformed parameter space. Sampling and probability
    evaluations are performed in the native space, while public facing APIs can
    operate in the transformed space. Jacobian corrections associated with the
    transformations are accounted for when computing probabilities.

    Parameters
    ----------
    dictionary : dict, optional
        Mapping from native parameter names to prior objects.
    filename : str, optional
        Filename containing a serialized prior dictionary.
    conversion_function : callable, optional
        Conversion function passed through to :class:`PriorDict`.
    transformations : dict, optional
        Dictionary describing transformations. Each entry can either be keyed
        by the native parameter name or by the transformed name. A definition
        must contain the following fields:

        ``native_key`` (optional)
            Name of the native parameter. Required when the dictionary key is
            the transformed name.
        ``transformed_key`` (optional)
            Name of the transformed parameter. Defaults to the native key.
        ``forward`` (callable, optional)
            Function mapping native values to transformed values. Defaults to
            the identity transformation.
        ``inverse`` (callable, optional)
            Function mapping transformed values to native values. Defaults to
            the identity transformation.
        ``jacobian`` (callable, optional)
            Function returning the determinant of the Jacobian matrix of the
            forward transformation evaluated in the native space. Defaults to a
            function returning ones with the appropriate shape.
    """

    def __init__(
        self,
        dictionary=None,
        filename=None,
        conversion_function=None,
        transformations=None,
    ):
        self._forward_transforms = dict()
        self._inverse_transforms = dict()
        self._jacobian_transforms = dict()
        self._transformed_least_recently_sampled = dict()
        self._transform_definitions = dict()
        self._pending_transformations = transformations or dict()
        self._transformed_priors = dict()
        self._user_conversion_function = conversion_function
        self._transformation_groups = dict()
        self._group_by_native = dict()
        self._group_by_transformed = dict()
        super(TransformedConditionalPriorDict, self).__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=None,
        )
        self.conversion_function = self._compose_conversion_function(
            self._user_conversion_function
        )
        self._initialize_transformations(self._pending_transformations)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _identity(value):
        return value

    @staticmethod
    def _unity_jacobian(*values):
        if not values:
            return 1.0
        arrays = [np.asarray(val) for val in values]
        broadcast = np.broadcast_arrays(*arrays)
        target = broadcast[0]
        return np.ones_like(target, dtype=float)

    @staticmethod
    def _as_tuple(value, default=None):
        if value is None:
            return tuple(default or [])
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    def _initialize_transformations(self, transformations):
        self._register_default_transforms()
        for key, definition in (transformations or {}).items():
            if not isinstance(definition, dict):
                raise TypeError(
                    "Transformation definition for '{}' must be a dictionary".format(key)
                )
            definition = definition.copy()
            normalized = self._normalize_transformation_definition(key, definition)
            if normalized is None:
                continue
            self._register_transformation_group(**normalized)

    def _register_default_transforms(self):
        for native_key in list(dict.keys(self)):
            self._register_identity_group(native_key)

    def _register_identity_group(self, native_key):
        group_id = self._group_by_native.get(native_key)
        if group_id is not None and self._transformation_groups.get(group_id, {}).get(
            "native_keys"
        ) == (native_key,):
            return
        definition = dict(
            native_keys=(native_key,),
            transformed_keys=(native_key,),
            forward=self._identity,
            inverse=self._identity,
            jacobian=self._unity_jacobian,
            raw_definition=dict(native_key=native_key, transformed_key=native_key),
        )
        self._unregister_group((native_key,))
        self._store_transformation_group(definition)

    def _normalize_transformation_definition(self, key, definition):
        native_keys = None
        if "native_keys" in definition:
            native_keys = self._as_tuple(definition.pop("native_keys"))
        elif "native_key" in definition:
            native_keys = self._as_tuple(definition.pop("native_key"))
        elif key in self:
            native_keys = (key,)
        if not native_keys:
            logger.debug(
                "Ignoring transformation definition for %s as native keys are unknown", key
            )
            return None
        for native_key in native_keys:
            if native_key not in self:
                logger.debug(
                    "Ignoring transformation for %s as native key %s is unknown",
                    key,
                    native_key,
                )
                return None
        transformed_keys = None
        if "transformed_keys" in definition:
            transformed_keys = self._as_tuple(definition.pop("transformed_keys"))
        elif "transformed_key" in definition:
            transformed_keys = self._as_tuple(definition.pop("transformed_key"))
        elif key not in native_keys:
            transformed_keys = (key,)
        if not transformed_keys:
            transformed_keys = native_keys
        if len(transformed_keys) != len(native_keys):
            raise ValueError(
                "Number of transformed keys ({}) must match native keys ({})".format(
                    len(transformed_keys), len(native_keys)
                )
            )
        forward = definition.pop("forward", self._identity)
        inverse = definition.pop("inverse", self._identity)
        jacobian = definition.pop("jacobian", self._unity_jacobian)
        raw_definition = dict(definition)
        raw_definition.update(
            native_keys=native_keys,
            transformed_keys=transformed_keys,
            forward=forward,
            inverse=inverse,
            jacobian=jacobian,
        )
        return dict(
            native_keys=native_keys,
            transformed_keys=transformed_keys,
            forward=forward,
            inverse=inverse,
            jacobian=jacobian,
            raw_definition=raw_definition,
        )

    def _register_transformation_group(
        self, native_keys, transformed_keys, forward, inverse, jacobian, raw_definition
    ):
        for native_key in native_keys:
            group_id = self._group_by_native.get(native_key)
            if group_id is not None:
                self._unregister_group(group_id)
        definition = dict(
            native_keys=tuple(native_keys),
            transformed_keys=tuple(transformed_keys),
            forward=forward,
            inverse=inverse,
            jacobian=jacobian,
            raw_definition=raw_definition,
        )
        self._store_transformation_group(definition)

    def _store_transformation_group(self, definition):
        native_keys = definition["native_keys"]
        transformed_keys = definition["transformed_keys"]
        group_id = native_keys
        self._transformation_groups[group_id] = definition
        for native_key in native_keys:
            self._group_by_native[native_key] = group_id
        for index, transformed_key in enumerate(transformed_keys):
            self._group_by_transformed[transformed_key] = (group_id, index)
            self._forward_transforms[transformed_key] = definition["forward"]
            self._inverse_transforms[transformed_key] = definition["inverse"]
            self._jacobian_transforms[transformed_key] = definition["jacobian"]
            self._transformed_least_recently_sampled.setdefault(transformed_key, None)
            if transformed_key != native_keys[min(index, len(native_keys) - 1)] or len(native_keys) > 1:
                self._transformed_priors[transformed_key] = self._create_transformed_prior(
                    native_keys=native_keys,
                    transformed_keys=transformed_keys,
                    index=index,
                    definition=definition,
                )
        self._transform_definitions[group_id] = definition
        self._transformed_keys_cache = None

    def _unregister_group(self, group_id):
        definition = self._transformation_groups.pop(group_id, None)
        if definition is None:
            return
        for native_key in definition["native_keys"]:
            self._group_by_native.pop(native_key, None)
        for transformed_key in definition["transformed_keys"]:
            self._group_by_transformed.pop(transformed_key, None)
            self._forward_transforms.pop(transformed_key, None)
            self._inverse_transforms.pop(transformed_key, None)
            self._jacobian_transforms.pop(transformed_key, None)
            self._transformed_priors.pop(transformed_key, None)
            self._transformed_least_recently_sampled.pop(transformed_key, None)
        self._transform_definitions.pop(group_id, None)
        self._transformed_keys_cache = None

    def register_transformation(
        self,
        native_key=None,
        native_keys=None,
        transformed_key=None,
        transformed_keys=None,
        forward=None,
        inverse=None,
        jacobian=None,
        **metadata,
    ):
        """Register or update a transformation for one or more native parameters."""

        if native_keys is None:
            native_keys = ()
        native_keys = list(native_keys)
        if native_key is not None:
            native_keys.append(native_key)
        if not native_keys:
            raise ValueError("At least one native key must be supplied")
        for key in native_keys:
            if key not in self:
                raise KeyError("Unknown native key '{}'".format(key))
        if transformed_keys is None:
            transformed_keys = []
        transformed_keys = list(transformed_keys)
        if transformed_key is not None:
            transformed_keys.append(transformed_key)
        if not transformed_keys:
            transformed_keys = list(native_keys)
        if len(native_keys) != len(transformed_keys):
            raise ValueError(
                "Number of transformed keys ({}) must match native keys ({})".format(
                    len(transformed_keys), len(native_keys)
                )
            )
        definition = dict(metadata)
        definition.update(
            native_keys=tuple(native_keys),
            transformed_keys=tuple(transformed_keys),
            forward=forward or self._identity,
            inverse=inverse or self._identity,
            jacobian=jacobian or self._unity_jacobian,
        )
        normalized = self._normalize_transformation_definition(
            native_keys[0], definition
        )
        if normalized is None:
            raise ValueError("Could not normalize transformation definition")
        self._register_transformation_group(**normalized)

    # ------------------------------------------------------------------
    # Key utilities
    # ------------------------------------------------------------------
    def _transformed_keys_for_native(self, native_key):
        group_id = self._group_by_native.get(native_key)
        if group_id is None:
            return [native_key]
        definition = self._transformation_groups.get(group_id)
        if definition is None:
            return [native_key]
        return list(definition["transformed_keys"])

    def _native_keys_for(self, key):
        if dict.__contains__(self, key):
            return [key]
        mapping = self._group_by_transformed.get(key)
        if mapping is None:
            raise KeyError("Unknown key '{}'".format(key))
        group_id, _ = mapping
        definition = self._transformation_groups.get(group_id)
        if definition is None:
            raise KeyError("Unknown key '{}'".format(key))
        return list(definition["native_keys"])

    def _convert_keys_to_native(self, keys):
        native_keys = []
        for key in keys:
            for native_key in self._native_keys_for(key):
                if native_key not in native_keys:
                    native_keys.append(native_key)
        return native_keys

    # ------------------------------------------------------------------
    # Transformation helpers
    # ------------------------------------------------------------------
    def _transform_native_samples(self, samples, transformed_keys):
        transformed_samples = dict()
        group_cache = dict()
        for key in transformed_keys:
            if dict.__contains__(self, key):
                if key in samples:
                    transformed_samples[key] = samples[key]
                    self._transformed_least_recently_sampled[key] = samples[key]
                continue
            mapping = self._group_by_transformed.get(key)
            if mapping is None:
                continue
            group_id, _ = mapping
            if group_id not in group_cache:
                definition = self._transformation_groups.get(group_id)
                if definition is None:
                    continue
                native_values = {
                    native_key: samples[native_key]
                    for native_key in definition["native_keys"]
                    if native_key in samples
                }
                if len(native_values) != len(definition["native_keys"]):
                    continue
                group_cache[group_id] = self._evaluate_forward(definition, native_values)
            group_result = group_cache.get(group_id, {})
            if key in group_result:
                transformed_samples[key] = group_result[key]
                self._transformed_least_recently_sampled[key] = group_result[key]
        return transformed_samples

    def _transform_to_native(self, sample, update_least_recently_sampled=False):
        native_sample = dict()
        log_abs_det_jacobian = None
        for key, value in sample.items():
            if dict.__contains__(self, key):
                native_sample[key] = value
        for group_id, definition in self._transformation_groups.items():
            transformed_keys = definition["transformed_keys"]
            if all(key in sample for key in transformed_keys):
                transformed_values = {key: sample[key] for key in transformed_keys}
                inverse_result = self._evaluate_inverse(definition, transformed_values)
                native_sample.update(inverse_result)
                jacobian_value = self._evaluate_jacobian(definition, inverse_result)
                term = np.log(np.abs(jacobian_value))
                if log_abs_det_jacobian is None:
                    log_abs_det_jacobian = term
                else:
                    log_abs_det_jacobian = log_abs_det_jacobian + term
                if update_least_recently_sampled:
                    for native_key, native_value in inverse_result.items():
                        if native_key in self:
                            self[native_key].least_recently_sampled = native_value
                    for transformed_key in transformed_keys:
                        self._transformed_least_recently_sampled[
                            transformed_key
                        ] = sample[transformed_key]
            elif any(key in sample for key in transformed_keys):
                missing = [key for key in transformed_keys if key not in sample]
                raise KeyError(
                    "Sample is missing transformed keys {} required to invert transformation for {}".format(
                        missing, definition["native_keys"]
                    )
                )
        if log_abs_det_jacobian is None:
            log_abs_det_jacobian = 0.0
        else:
            log_abs_det_jacobian = np.asarray(log_abs_det_jacobian)
        return native_sample, log_abs_det_jacobian

    def _evaluate_forward(self, definition, native_values):
        ordered_values = [native_values[key] for key in definition["native_keys"]]
        result = definition["forward"](*ordered_values)
        return self._format_group_result(result, definition["transformed_keys"])

    def _evaluate_inverse(self, definition, transformed_values):
        ordered_values = [transformed_values[key] for key in definition["transformed_keys"]]
        result = definition["inverse"](*ordered_values)
        return self._format_group_result(result, definition["native_keys"])

    def _evaluate_jacobian(self, definition, native_values):
        ordered_values = [native_values[key] for key in definition["native_keys"]]
        result = definition["jacobian"](*ordered_values)
        array = np.asarray(result)
        dimension = len(definition["native_keys"])
        if array.ndim >= 2 and array.shape[-2:] == (dimension, dimension):
            return np.linalg.det(array)
        return array

    def _format_group_result(self, result, keys):
        if isinstance(result, dict):
            missing = [key for key in keys if key not in result]
            if missing:
                raise KeyError(
                    "Transformation result missing entries for keys {}".format(missing)
                )
            return {key: result[key] for key in keys}
        if len(keys) == 1:
            return {keys[0]: result}
        if isinstance(result, (list, tuple)):
            if len(result) != len(keys):
                raise ValueError(
                    "Transformation returned {} values but {} were expected".format(
                        len(result), len(keys)
                    )
                )
            return {key: result[index] for index, key in enumerate(keys)}
        array = np.asarray(result)
        if array.shape[0] == len(keys):
            return {key: array[index] for index, key in enumerate(keys)}
        if array.shape[-1] == len(keys):
            return {
                key: array[..., index]
                for index, key in enumerate(keys)
            }
        raise ValueError(
            "Transformation output has incompatible shape {} for keys {}".format(
                array.shape, keys
            )
        )

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------
    def sample(self, size=None):
        return self.sample_subset_constrained(keys=self.transformed_keys, size=size)

    def sample_subset(self, keys=iter([]), size=None):
        keys = list(keys)
        if len(keys) == 0:
            keys = self.transformed_keys
        native_keys = self._convert_keys_to_native(keys)
        native_samples = super(TransformedConditionalPriorDict, self).sample_subset(
            keys=native_keys, size=size
        )
        return self._transform_native_samples(native_samples, keys)

    def sample_subset_constrained(self, keys=iter([]), size=None):
        keys = list(keys)
        if len(keys) == 0:
            keys = self.transformed_keys
        native_keys = self._convert_keys_to_native(keys)
        native_samples = super(
            TransformedConditionalPriorDict, self
        ).sample_subset_constrained(keys=native_keys, size=size)
        return self._transform_native_samples(native_samples, keys)

    def sample_subset_constrained_as_array(self, keys=iter([]), size=None):
        keys = list(keys)
        use_keys = keys if len(keys) > 0 else self.transformed_keys
        samples_dict = self.sample_subset_constrained(keys=use_keys, size=size)
        samples_dict = {key: np.atleast_1d(val) for key, val in samples_dict.items()}
        samples_list = [samples_dict[key] for key in use_keys]
        return np.array(samples_list)

    # ------------------------------------------------------------------
    # Probability interface
    # ------------------------------------------------------------------
    def prob(self, sample, **kwargs):
        native_sample, log_abs_det_jacobian = self._transform_to_native(sample)
        native_prob = super(TransformedConditionalPriorDict, self).prob(
            native_sample, **kwargs
        )
        transformed_prob = native_prob / np.exp(log_abs_det_jacobian)
        for key in sample:
            if key in self._transformed_least_recently_sampled:
                self._transformed_least_recently_sampled[key] = sample[key]
        return transformed_prob

    def ln_prob(self, sample, axis=None, normalized=True):
        native_sample, log_abs_det_jacobian = self._transform_to_native(sample)
        native_ln_prob = super(TransformedConditionalPriorDict, self).ln_prob(
            native_sample, axis=axis, normalized=normalized
        )
        transformed_ln_prob = native_ln_prob - log_abs_det_jacobian
        for key in sample:
            if key in self._transformed_least_recently_sampled:
                self._transformed_least_recently_sampled[key] = sample[key]
        return transformed_ln_prob

    # ------------------------------------------------------------------
    # Rescaling utilities
    # ------------------------------------------------------------------
    def rescale(self, keys, theta):
        keys = list(keys)
        native_keys = self._convert_keys_to_native(keys)
        native_values = super(TransformedConditionalPriorDict, self).rescale(
            native_keys, theta
        )
        native_samples = dict(zip(native_keys, native_values))
        transformed_samples = self._transform_native_samples(native_samples, keys)
        return [
            transformed_samples[key]
            if key in transformed_samples
            else native_samples.get(key)
            for key in keys
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def native_keys(self):
        return list(self.keys())

    @property
    def transformed_keys(self):
        if getattr(self, "_transformed_keys_cache", None) is None:
            ordered = []
            for native_key in self.sorted_keys:
                for transformed_key in self._transformed_keys_for_native(native_key):
                    if transformed_key not in ordered:
                        ordered.append(transformed_key)
            self._transformed_keys_cache = ordered
        return list(self._transformed_keys_cache)

    @property
    def transformed_sorted_keys_without_fixed_parameters(self):
        keys = []
        for native_key in self.sorted_keys_without_fixed_parameters:
            for transformed_key in self._transformed_keys_for_native(native_key):
                if transformed_key not in keys:
                    keys.append(transformed_key)
        return keys

    @property
    def transformed_least_recently_sampled(self):
        return dict(self._transformed_least_recently_sampled)

    @property
    def non_fixed_keys(self):
        return self.transformed_sorted_keys_without_fixed_parameters

    # ------------------------------------------------------------------
    # Dictionary interface overrides
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        if key in self._transformed_priors:
            return self._transformed_priors[key]
        return super(TransformedConditionalPriorDict, self).__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __contains__(self, key):
        if key in self._transformed_priors:
            return True
        if key in self._group_by_transformed:
            return True
        return dict.__contains__(self, key)

    def __setitem__(self, key, value):
        super(TransformedConditionalPriorDict, self).__setitem__(key, value)
        self._register_identity_group(key)

    def __delitem__(self, key):
        super(TransformedConditionalPriorDict, self).__delitem__(key)
        group_id = self._group_by_native.get(key)
        remaining_natives = []
        if group_id is not None:
            definition = self._transformation_groups.get(group_id)
            if definition is not None:
                remaining_natives = [
                    native for native in definition["native_keys"] if native != key
                ]
            self._unregister_group(group_id)
        self._transformed_least_recently_sampled.pop(key, None)
        for native in remaining_natives:
            if native in self:
                self._register_identity_group(native)
        self._transformed_keys_cache = None

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _augment_sample_with_native(self, sample):
        if sample is None:
            return sample
        augmented = dict(sample)
        for group_id, definition in self._transformation_groups.items():
            transformed_keys = definition["transformed_keys"]
            if not all(key in sample for key in transformed_keys):
                continue
            transformed_values = {key: sample[key] for key in transformed_keys}
            inverse_result = self._evaluate_inverse(definition, transformed_values)
            for native_key, native_value in inverse_result.items():
                augmented.setdefault(native_key, native_value)
        return augmented

    def _compose_conversion_function(self, user_conversion):
        def conversion(sample, *args, **kwargs):
            augmented = self._augment_sample_with_native(sample)
            if user_conversion is None:
                return augmented
            return user_conversion(augmented, *args, **kwargs)

        return conversion

    def _create_transformed_prior(
        self,
        *,
        native_keys,
        transformed_keys,
        index,
        definition,
    ):
        base_native_index = min(index, len(native_keys) - 1)
        base_native_key = native_keys[base_native_index]
        base_prior = dict.__getitem__(self, base_native_key)
        transformed_key = transformed_keys[index]
        return TransformedPriorView(
            parent=self,
            group_id=tuple(native_keys),
            transformed_key=transformed_key,
            base_prior=base_prior,
            index=index,
            definition=definition,
            minimum=self._definition_value_for_key(
                definition, "minimum", transformed_key, index
            ),
            maximum=self._definition_value_for_key(
                definition, "maximum", transformed_key, index
            ),
            latex_label=self._definition_value_for_key(
                definition, "latex_label", transformed_key, index
            ),
            unit=self._definition_value_for_key(definition, "unit", transformed_key, index),
        )

    def _definition_value_for_key(self, definition, field, transformed_key, index):
        raw = definition.get("raw_definition", {})
        value = raw.get(field)
        if isinstance(value, dict):
            return value.get(transformed_key)
        if isinstance(value, (list, tuple)):
            if len(value) == len(definition["transformed_keys"]):
                return value[index]
        return value


class DirichletPriorDict(ConditionalPriorDict):
    def __init__(self, n_dim=None, label="dirichlet_"):
        from .conditional import DirichletElement

        self.n_dim = n_dim
        self.label = label
        super(DirichletPriorDict, self).__init__(dictionary=dict())
        for ii in range(n_dim - 1):
            self[label + "{}".format(ii)] = DirichletElement(
                order=ii, n_dimensions=n_dim, label=label
            )

    def copy(self, **kwargs):
        return self.__class__(n_dim=self.n_dim, label=self.label)

    def _get_json_dict(self):
        total_dict = dict()
        total_dict["__prior_dict__"] = True
        total_dict["__module__"] = self.__module__
        total_dict["__name__"] = self.__class__.__name__
        total_dict["n_dim"] = self.n_dim
        total_dict["label"] = self.label
        return total_dict

    @classmethod
    def _get_from_json_dict(cls, prior_dict):
        try:
            cls == getattr(
                import_module(prior_dict["__module__"]), prior_dict["__name__"]
            )
        except ImportError:
            logger.debug(
                "Cannot import prior module {}.{}".format(
                    prior_dict["__module__"], prior_dict["__name__"]
                )
            )
        except KeyError:
            logger.debug("Cannot find module name to load")
        for key in ["__module__", "__name__", "__prior_dict__"]:
            if key in prior_dict:
                del prior_dict[key]
        obj = cls(**prior_dict)
        return obj


class ConditionalPriorDictException(PriorDictException):
    """General base class for all conditional prior dict exceptions"""


def create_default_prior(name, default_priors_file=None):
    """Make a default prior for a parameter with a known name.

    Parameters
    ==========
    name: str
        Parameter name
    default_priors_file: str, optional
        If given, a file containing the default priors.

    Returns
    =======
    prior: Prior
        Default prior distribution for that parameter, if unknown None is
        returned.
    """

    if default_priors_file is None:
        logger.debug("No prior file given.")
        prior = None
    else:
        default_priors = PriorDict(filename=default_priors_file)
        if name in default_priors.keys():
            prior = default_priors[name]
        else:
            logger.debug("No default prior found for variable {}.".format(name))
            prior = None
    return prior


class IllegalConditionsException(ConditionalPriorDictException):
    """Exception class to handle prior dicts that contain unresolvable conditions."""

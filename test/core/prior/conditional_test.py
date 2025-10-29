import os
import shutil
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pickle

import bilby


class TestConditionalPrior(unittest.TestCase):
    def setUp(self):
        self.condition_func_call_counter = 0

        def condition_func(reference_parameters, test_variable_1, test_variable_2):
            self.condition_func_call_counter += 1
            return {key: value + 1 for key, value in reference_parameters.items()}

        self.condition_func = condition_func
        self.minimum = 0
        self.maximum = 5
        self.test_variable_1 = 0
        self.test_variable_2 = 1
        self.prior = bilby.core.prior.ConditionalBasePrior(
            condition_func=condition_func, minimum=self.minimum, maximum=self.maximum
        )

    def tearDown(self):
        del self.condition_func
        del self.condition_func_call_counter
        del self.minimum
        del self.maximum
        del self.test_variable_1
        del self.test_variable_2
        del self.prior

    def test_reference_params(self):
        self.assertDictEqual(
            dict(minimum=self.minimum, maximum=self.maximum),
            self.prior.reference_params,
        )

    def test_required_variables(self):
        self.assertListEqual(
            ["test_variable_1", "test_variable_2"],
            sorted(self.prior.required_variables),
        )

    def test_required_variables_no_condition_func(self):
        self.prior = bilby.core.prior.ConditionalBasePrior(
            condition_func=None, minimum=self.minimum, maximum=self.maximum
        )
        self.assertListEqual([], self.prior.required_variables)

    def test_get_instantiation_dict(self):
        expected = dict(
            minimum=0,
            maximum=5,
            name=None,
            latex_label=None,
            unit=None,
            boundary=None,
            condition_func=self.condition_func,
        )
        actual = self.prior.get_instantiation_dict()
        for key, value in expected.items():
            if key == "condition_func":
                continue
            self.assertEqual(value, actual[key])

    def test_update_conditions_correct_variables(self):
        self.prior.update_conditions(
            test_variable_1=self.test_variable_1, test_variable_2=self.test_variable_2
        )
        self.assertEqual(1, self.condition_func_call_counter)
        self.assertEqual(self.minimum + 1, self.prior.minimum)
        self.assertEqual(self.maximum + 1, self.prior.maximum)

    def test_update_conditions_no_variables(self):
        self.prior.update_conditions(
            test_variable_1=self.test_variable_1, test_variable_2=self.test_variable_2
        )
        self.prior.update_conditions()
        self.assertEqual(1, self.condition_func_call_counter)
        self.assertEqual(self.minimum + 1, self.prior.minimum)
        self.assertEqual(self.maximum + 1, self.prior.maximum)

    def test_update_conditions_illegal_variables(self):
        with self.assertRaises(bilby.core.prior.IllegalRequiredVariablesException):
            self.prior.update_conditions(test_parameter_1=self.test_variable_1)

    def test_sample_calls_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.sample(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_rescale_calls_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.rescale(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_prob_calls_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.prob(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_rescale_ln_prob_update_conditions(self):
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.ln_prob(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            calls = [
                mock.call(
                    test_parameter_1=self.test_variable_1,
                    test_parameter_2=self.test_variable_2,
                ),
                mock.call(),
            ]
            m.assert_has_calls(calls)

    def test_cdf_calls_update_conditions(self):
        self.prior = bilby.core.prior.ConditionalUniform(
            condition_func=self.condition_func, minimum=self.minimum, maximum=self.maximum
        )
        with mock.patch.object(self.prior, "update_conditions") as m:
            self.prior.cdf(
                1,
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )
            m.assert_called_with(
                test_parameter_1=self.test_variable_1,
                test_parameter_2=self.test_variable_2,
            )

    def test_reset_to_reference_parameters(self):
        self.prior.minimum = 10
        self.prior.maximum = 20
        self.prior.reset_to_reference_parameters()
        self.assertEqual(self.prior.reference_params["minimum"], self.prior.minimum)
        self.assertEqual(self.prior.reference_params["maximum"], self.prior.maximum)

    def test_cond_prior_instantiation_no_boundary_prior(self):
        prior = bilby.core.prior.ConditionalFermiDirac(
            condition_func=None, sigma=1, mu=1
        )
        self.assertIsNone(prior.boundary)


class TestConditionalPriorDict(unittest.TestCase):
    def setUp(self):
        def condition_func_1(reference_parameters, var_0):
            return dict(minimum=reference_parameters["minimum"], maximum=var_0)

        def condition_func_2(reference_parameters, var_0, var_1):
            return dict(minimum=reference_parameters["minimum"], maximum=var_1)

        def condition_func_3(reference_parameters, var_1, var_2):
            return dict(minimum=reference_parameters["minimum"], maximum=var_2)

        self.minimum = 0
        self.maximum = 1
        self.prior_0 = bilby.core.prior.Uniform(
            minimum=self.minimum, maximum=self.maximum
        )
        self.prior_1 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_1, minimum=self.minimum, maximum=self.maximum
        )
        self.prior_2 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_2, minimum=self.minimum, maximum=self.maximum
        )
        self.prior_3 = bilby.core.prior.ConditionalUniform(
            condition_func=condition_func_3, minimum=self.minimum, maximum=self.maximum
        )
        self.conditional_priors = bilby.core.prior.ConditionalPriorDict(
            dict(
                var_3=self.prior_3,
                var_2=self.prior_2,
                var_0=self.prior_0,
                var_1=self.prior_1,
            )
        )
        self.conditional_priors_manually_set_items = (
            bilby.core.prior.ConditionalPriorDict()
        )
        self.test_sample = dict(var_0=0.7, var_1=0.6, var_2=0.5, var_3=0.4)
        self.test_value = 1 / np.prod([self.test_sample[f"var_{ii}"] for ii in range(3)])
        for key, value in dict(
            var_0=self.prior_0,
            var_1=self.prior_1,
            var_2=self.prior_2,
            var_3=self.prior_3,
        ).items():
            self.conditional_priors_manually_set_items[key] = value

    def tearDown(self):
        del self.minimum
        del self.maximum
        del self.prior_0
        del self.prior_1
        del self.prior_2
        del self.prior_3
        del self.conditional_priors
        del self.conditional_priors_manually_set_items
        del self.test_sample

    def test_conditions_resolved_upon_instantiation(self):
        self.assertListEqual(
            ["var_0", "var_1", "var_2", "var_3"], self.conditional_priors.sorted_keys
        )

    def test_conditions_resolved_setting_items(self):
        self.assertListEqual(
            ["var_0", "var_1", "var_2", "var_3"],
            self.conditional_priors_manually_set_items.sorted_keys,
        )

    def test_unconditional_keys_upon_instantiation(self):
        self.assertListEqual(["var_0"], self.conditional_priors.unconditional_keys)

    def test_unconditional_keys_setting_items(self):
        self.assertListEqual(
            ["var_0"], self.conditional_priors_manually_set_items.unconditional_keys
        )

    def test_conditional_keys_upon_instantiation(self):
        self.assertListEqual(
            ["var_1", "var_2", "var_3"], self.conditional_priors.conditional_keys
        )

    def test_conditional_keys_setting_items(self):
        self.assertListEqual(
            ["var_1", "var_2", "var_3"],
            self.conditional_priors_manually_set_items.conditional_keys,
        )

    def test_prob(self):
        self.assertEqual(self.test_value, self.conditional_priors.prob(sample=self.test_sample))

    def test_prob_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.prob(sample=self.test_sample)

    def test_ln_prob(self):
        self.assertEqual(np.log(self.test_value), self.conditional_priors.ln_prob(sample=self.test_sample))

    def test_ln_prob_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.ln_prob(sample=self.test_sample)

    def test_sample_subset_all_keys(self):
        bilby.core.utils.random.seed(5)
        self.assertDictEqual(
            dict(
                var_0=0.8050029237453802,
                var_1=0.6503946979510289,
                var_2=0.33516501262044845,
                var_3=0.09579062316418356,
            ),
            self.conditional_priors.sample_subset(
                keys=["var_0", "var_1", "var_2", "var_3"]
            ),
        )

    def test_sample_illegal_subset(self):
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.sample_subset(keys=["var_1"])

    def test_sample_multiple(self):
        def condition_func(reference_params, a):
            return dict(
                minimum=reference_params["minimum"],
                maximum=reference_params["maximum"],
                alpha=reference_params["alpha"] * a,
            )

        priors = bilby.core.prior.ConditionalPriorDict()
        priors["a"] = bilby.core.prior.Uniform(minimum=0, maximum=1)
        priors["b"] = bilby.core.prior.ConditionalPowerLaw(
            condition_func=condition_func, minimum=1, maximum=2, alpha=-2
        )
        print(priors.sample(2))

    def test_rescale(self):
        self.conditional_priors = bilby.core.prior.ConditionalPriorDict(
            dict(
                var_3=self.prior_3,
                var_2=self.prior_2,
                var_0=self.prior_0,
                var_1=self.prior_1,
            )
        )
        ref_variables = self.test_sample.values()
        res = self.conditional_priors.rescale(
            keys=self.test_sample.keys(), theta=ref_variables
        )
        expected = [self.test_sample["var_0"]]
        for ii in range(1, 4):
            expected.append(expected[-1] * self.test_sample[f"var_{ii}"])
        self.assertListEqual(expected, res)

    def test_rescale_with_joint_prior(self):
        """
        Add a joint prior into the conditional prior dictionary and check that
        the returned list is flat.
        """

        # set multivariate Gaussian distribution
        names = ["mvgvar_0", "mvgvar_1"]
        mu = [[0.79, -0.83]]
        cov = [[[0.03, 0.], [0., 0.04]]]
        mvg = bilby.core.prior.MultivariateGaussianDist(names, mus=mu, covs=cov)

        priordict = bilby.core.prior.ConditionalPriorDict(
            dict(
                var_3=self.prior_3,
                var_2=self.prior_2,
                var_0=self.prior_0,
                var_1=self.prior_1,
                mvgvar_0=bilby.core.prior.MultivariateGaussian(mvg, "mvgvar_0"),
                mvgvar_1=bilby.core.prior.MultivariateGaussian(mvg, "mvgvar_1"),
            )
        )

        ref_variables = list(self.test_sample.values()) + [0.4, 0.1]
        keys = list(self.test_sample.keys()) + names
        res = priordict.rescale(keys=keys, theta=ref_variables)

        self.assertIsInstance(res, list)
        self.assertEqual(np.shape(res), (6,))
        self.assertListEqual([isinstance(r, float) for r in res], 6 * [True])

        # check conditional values are still as expected
        expected = [self.test_sample["var_0"]]
        for ii in range(1, 4):
            expected.append(expected[-1] * self.test_sample[f"var_{ii}"])
        self.assertListEqual(expected, res[0:4])

    def test_cdf(self):
        """
        Test that the CDF method is the inverse of the rescale method.

        Note that the format of inputs/outputs is different between the two methods.
        """
        sample = self.conditional_priors.sample()
        self.assertEqual(
            self.conditional_priors.rescale(
                sample.keys(),
                self.conditional_priors.cdf(sample=sample).values()
            ), list(sample.values())
        )

    def test_rescale_illegal_conditions(self):
        del self.conditional_priors["var_0"]
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            self.conditional_priors.rescale(
                keys=list(self.test_sample.keys()),
                theta=list(self.test_sample.values()),
            )

    def test_combined_conditions(self):
        def d_condition_func(reference_params, a, b, c):
            return dict(
                minimum=reference_params["minimum"], maximum=reference_params["maximum"]
            )

        def a_condition_func(reference_params, b, c):
            return dict(
                minimum=reference_params["minimum"], maximum=reference_params["maximum"]
            )

        priors = bilby.core.prior.ConditionalPriorDict()

        priors["a"] = bilby.core.prior.ConditionalUniform(
            condition_func=a_condition_func, minimum=0, maximum=1
        )

        priors["b"] = bilby.core.prior.LogUniform(minimum=1, maximum=10)

        priors["d"] = bilby.core.prior.ConditionalUniform(
            condition_func=d_condition_func, minimum=0.0, maximum=1.0
        )

        priors["c"] = bilby.core.prior.LogUniform(minimum=1, maximum=10)
        priors.sample()
        res = priors.rescale(["a", "b", "d", "c"], [0.5, 0.5, 0.5, 0.5])
        print(res)

    def test_subset_sampling(self):
        def _tp_conditional_uniform(ref_params, period):
            min_ref, max_ref = ref_params["minimum"], ref_params["maximum"]
            max_ref = np.minimum(max_ref, min_ref + period)
            return {"minimum": min_ref, "maximum": max_ref}

        p0 = 68400.0
        prior = bilby.core.prior.ConditionalPriorDict(
            {
                "tp": bilby.core.prior.ConditionalUniform(
                    condition_func=_tp_conditional_uniform, minimum=0, maximum=2 * p0
                )
            }
        )

        # ---------- 0. Sanity check: sample full prior
        prior["period"] = p0
        samples2d = prior.sample(1000)
        assert samples2d["tp"].max() < p0

        # ---------- 1. Subset sampling with external delta-prior
        print("Test 1: Subset-sampling conditionals for fixed 'externals':")
        prior["period"] = p0
        samples1d = prior.sample_subset(["tp"], 1000)
        self.assertLess(samples1d["tp"].max(), p0)

        # ---------- 2. Subset sampling with external uniform prior
        prior["period"] = bilby.core.prior.Uniform(minimum=p0, maximum=2 * p0)
        print("Test 2: Subset-sampling conditionals for 'external' uncertainties:")
        with self.assertRaises(bilby.core.prior.IllegalConditionsException):
            prior.sample_subset(["tp"], 1000)


class TestTransformedConditionalPriorDict(unittest.TestCase):

    def setUp(self):
        self.native_prior = bilby.core.prior.Uniform(minimum=0.5, maximum=2.0, name="x")
        self.other_prior = bilby.core.prior.Uniform(minimum=-1.0, maximum=1.0, name="y")

        def conversion(sample):
            converted = dict(sample)
            if "log_x" in converted and "log_x_constraint" not in converted:
                converted["log_x_constraint"] = converted["log_x"]
            if "x" in converted and "log_x_constraint" not in converted:
                converted["log_x_constraint"] = np.log(converted["x"])
            return converted

        transformations = {
            "x": dict(
                transformed_key="log_x",
                forward=np.log,
                inverse=np.exp,
                jacobian=lambda x: 1.0 / np.asarray(x),
            )
        }

        self.priors = bilby.core.prior.TransformedConditionalPriorDict(
            dictionary={"x": self.native_prior, "y": self.other_prior},
            transformations=transformations,
            conversion_function=conversion,
        )
        self.priors["log_x_constraint"] = bilby.core.prior.Constraint(
            minimum=-0.5, maximum=0.5
        )

    def test_transformed_keys(self):
        self.assertListEqual(
            ["log_x", "y", "log_x_constraint"], self.priors.transformed_keys
        )

    def test_sample_subset_uses_forward_transformation(self):
        samples = self.priors.sample_subset(keys=["log_x"], size=5)
        self.assertIn("log_x", samples)
        native_samples = np.exp(samples["log_x"])
        self.assertTrue(
            np.all(
                (native_samples >= self.native_prior.minimum)
                & (native_samples <= self.native_prior.maximum)
            )
        )

    def test_rescale_applies_transformation(self):
        theta = [0.3, 0.8]
        transformed = self.priors.rescale(["log_x", "y"], theta)
        expected_native = self.native_prior.rescale(theta[0])
        self.assertAlmostEqual(transformed[0], np.log(expected_native))

    def test_probability_includes_jacobian(self):
        sample = {"log_x": np.log(1.1), "y": 0.0}
        native_sample, log_abs_jac = self.priors._transform_to_native(sample)
        native_prob = super(
            bilby.core.prior.TransformedConditionalPriorDict, self.priors
        ).prob(native_sample)
        expected = native_prob / np.exp(log_abs_jac)
        self.assertAlmostEqual(expected, self.priors.prob(sample))

    def test_ln_probability_matches_log(self):
        sample = {"log_x": np.log(1.4), "y": 0.2}
        prob = self.priors.prob(sample)
        self.assertAlmostEqual(np.log(prob), self.priors.ln_prob(sample))

    def test_transformed_constraint_is_applied(self):
        valid_sample = {"log_x": np.log(1.2), "y": 0.0}
        invalid_sample = {"log_x": np.log(1.7), "y": 0.0}
        self.assertGreater(self.priors.prob(valid_sample), 0.0)
        self.assertEqual(0.0, self.priors.prob(invalid_sample))

    def test_rescale_updates_transformed_tracking(self):
        theta = [0.4, 0.5]
        values = self.priors.rescale(["log_x", "y"], theta)
        tracked = self.priors.transformed_least_recently_sampled
        self.assertAlmostEqual(tracked["log_x"], values[0])
        self.assertAlmostEqual(tracked["y"], values[1])

    def test_sample_subset_constrained_as_array_shape(self):
        samples = self.priors.sample_subset_constrained_as_array(
            keys=["log_x", "y"], size=3
        )
        self.assertEqual((2, 3), samples.shape)

    def test_sample_returns_transformed_keys(self):
        sample = self.priors.sample()
        self.assertIn("log_x", sample)
        self.assertIn("y", sample)
        self.assertNotIn("x", sample)

    def test_non_fixed_keys_match_transformed(self):
        self.assertListEqual(
            ["log_x", "y"], self.priors.non_fixed_keys
        )

    def test_conversion_function_includes_native_parameters(self):
        converted = self.priors.conversion_function({"log_x": np.log(1.3), "y": 0.1})
        self.assertIn("x", converted)
        self.assertAlmostEqual(converted["x"], np.exp(np.log(1.3)))

    def test_multi_parameter_transformation_probability(self):
        x_prior = bilby.core.prior.Uniform(minimum=0.5, maximum=1.5, name="x")
        y_prior = bilby.core.prior.Uniform(minimum=0.25, maximum=1.0, name="y")

        def forward(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            return r, theta

        def inverse(r, theta):
            return r * np.cos(theta), r * np.sin(theta)

        def jacobian(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            return 1.0 / r

        priors = bilby.core.prior.TransformedConditionalPriorDict(
            dictionary={"x": x_prior, "y": y_prior},
            transformations={
                "polar": dict(
                    native_keys=["x", "y"],
                    transformed_keys=["r", "theta"],
                    forward=forward,
                    inverse=inverse,
                    jacobian=jacobian,
                    minimum={"r": np.sqrt(0.5 ** 2 + 0.25 ** 2), "theta": -np.pi},
                    maximum={"r": np.sqrt(1.5 ** 2 + 1.0 ** 2), "theta": np.pi},
                )
            },
        )

        sample = {"r": 0.9, "theta": 0.4}
        native_sample, log_abs_jacobian = priors._transform_to_native(sample)
        expected_x, expected_y = inverse(sample["r"], sample["theta"])
        self.assertAlmostEqual(native_sample["x"], expected_x)
        self.assertAlmostEqual(native_sample["y"], expected_y)
        native_prob = super(
            bilby.core.prior.TransformedConditionalPriorDict, priors
        ).prob(native_sample)
        expected = native_prob / np.exp(log_abs_jacobian)
        self.assertAlmostEqual(expected, priors.prob(sample))

        transformed_samples = priors.sample_subset(keys=["r", "theta"], size=5)
        self.assertIn("r", transformed_samples)
        self.assertIn("theta", transformed_samples)

        rescaled = priors.rescale(["r", "theta"], [0.1, 0.6])
        self.assertEqual(2, len(rescaled))

    def test_transformed_view_requires_group_values(self):
        x_prior = bilby.core.prior.Uniform(minimum=0.3, maximum=1.2, name="x")
        y_prior = bilby.core.prior.Uniform(minimum=0.4, maximum=1.0, name="y")

        def forward(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            return r, phi

        def inverse(r, phi):
            return r * np.cos(phi), r * np.sin(phi)

        def jacobian(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            return 1.0 / r

        priors = bilby.core.prior.TransformedConditionalPriorDict(
            dictionary={"x": x_prior, "y": y_prior},
            transformations={
                "polar": dict(
                    native_keys=["x", "y"],
                    transformed_keys=["radius", "phi"],
                    forward=forward,
                    inverse=inverse,
                    jacobian=jacobian,
                )
            },
        )

        radius_prior = priors["radius"]
        with self.assertRaises(KeyError):
            radius_prior.prob(0.8)
        prob = radius_prior.prob(0.8, phi=0.3)
        self.assertIsInstance(prob, float)


class TestDirichletPrior(unittest.TestCase):

    def setUp(self):
        self.priors = bilby.core.prior.DirichletPriorDict(5)

    def tearDown(self):
        if os.path.isdir("priors"):
            shutil.rmtree("priors")

    def test_samples_sum_to_less_than_one(self):
        """
        Test that the samples sum to less than one as required for the
        Dirichlet distribution.
        """
        samples = pd.DataFrame(self.priors.sample(10000)).values
        self.assertLess(max(np.sum(samples, axis=1)), 1)

    def test_read_write_file(self):
        self.priors.to_file(outdir="priors", label="test")
        test = bilby.core.prior.PriorDict(filename="priors/test.prior")
        self.assertEqual(self.priors, test)

    def test_read_write_json(self):
        self.priors.to_json(outdir="priors", label="test")
        test = bilby.core.prior.PriorDict.from_json(filename="priors/test_prior.json")
        self.assertEqual(self.priors, test)

    def test_pickle(self):
        """Assert can be pickled (needed for use with bilby_pipe)"""
        pickle.dumps(self.priors)


if __name__ == "__main__":
    unittest.main()

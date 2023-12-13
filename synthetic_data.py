import numpy as np
import pandas as pd
import itertools
import json


def random_dist(size):
    dist = np.random.exponential(scale=1.0, size=size)  # This is to make sure that distribution is uniform on simplex
    # dist = np.random.random(size=size)
    return dist / dist.sum()


class SyntheticDataGenerator:
    """
    Class for generating synthetic categorical data, such that different subpopulations behave differently but are
    still similar for each feature specifically.

    """

    def __init__(self, name, num_features=1000, num_labels=2, num_samples=100000, num_subpops=5, seed=42):
        np.random.seed(seed)
        self.name = name
        self.consts = self.load_json()

        self.num_features = num_features
        self.num_labels = num_labels
        self.num_samples = num_samples
        self.num_subpops = num_subpops
        self.desc = f"Synthetic data with {num_features} features. Random numbers based on seed {seed}.\n"

        self.feature_names = [f'f{i}' for i in range(num_features)]
        self.subpops = [f's{i}' for i in range(num_subpops)]
        self.subpops_sizes = self.generate_subpop_sizes(num_samples, num_subpops)

        self.values = {subpop: dict() for subpop in self.subpops}
        self.pointers = {}
        self.num_categories = {name: np.random.randint(2, 5) for name in self.feature_names}
        self.generate_labels(num_subpops, num_labels)
        self.create_base_features()

    def load_json(self):
        try:
            self.consts = json.load(open('syntheticParametersConfigs.json'))[self.name]
        except KeyError:
            self.consts = json.load(open('syntheticParametersConfigs.json'))['default']
        return self.consts

    def generate_subpop_sizes(self, num_samples, num_subpops):
        subpops_sizes = np.random.normal(loc=num_samples / num_subpops, scale=self.consts['subpop_size_std'],
                                         size=num_subpops).round().astype(int)

        while not np.all(subpops_sizes > 0):  # Just in case, very unlikely
            subpops_sizes = np.random.normal(loc=num_samples / num_subpops, scale=self.consts['subpop_size_std'],
                                             size=num_subpops).round().astype(int)
        return subpops_sizes

    def generate_labels(self, num_subpops, num_labels):
        base = random_dist(size=num_labels)  # Base is the base distribution of the labels. We assume that the labels
        # are similarly distributed across all subpopulations.

        for subpop in range(num_subpops):
            rand_weight = self.consts['random_dist_weight1']
            dist = rand_weight * random_dist(size=num_labels) + (1 - rand_weight) * base
            self.values[self.subpops[subpop]]['label'] = np.random.choice(num_labels, size=self.subpops_sizes[subpop],
                                                                          p=dist)
        self.num_categories['label'] = num_labels

    def create_base_features(self):
        """
        Create ten random base features
        """
        for feature in self.feature_names[:10]:
            self.random_feature(feature)

    def random_feature(self, feature):
        """
        Generates feature values following completely random distribution.
        """
        self.desc += f"Feature {feature} is random. It has {self.num_categories[feature]} categories.\n"
        self.pointers[feature] = feature
        base_dist = random_dist(size=self.num_categories[feature])
        self.generate_random(feature, base_dist, bool(np.random.randint(0, 2)))

    def generate_random(self, feature, base_dist, subpop_specific):
        bias = np.random.random() * 0.5 if subpop_specific else 0
        for subpop in range(self.num_subpops):
            dist = random_dist(size=self.num_categories[feature])
            dist = dist * bias + (1 - bias) * base_dist
            self.values[self.subpops[subpop]][feature] = np.random.choice(self.num_categories[feature],
                                                                          size=self.subpops_sizes[subpop], p=dist)

    def feature_based_feature(self, feature, *other_features):
        """
        Generates a feature that is based on other features. There is a 50/50 chance that the feature value will be
        independent of subpopulation / subpopulation specific.
        """
        self.desc += f'Feature {feature} is based on {other_features}. ' \
                     f'It has {self.num_categories[feature]} categories.\n'
        self.pointers[feature] = other_features[0]
        very_base = random_dist(size=self.num_categories[feature])
        important = np.random.choice([0, 1], size=len(other_features), p=(1 - self.consts['feature_important_prob'],
                                                                          self.consts['feature_important_prob']))
        important[0] = 1  # Assume first feature is always important
        subpop_specific = bool(np.random.randint(0, 2))

        # Categories of the important features
        categories = [list(range(self.num_categories[other])) for flag, other in zip(important, other_features) if flag]
        base_dists = self.create_distributions(subpop_specific, categories, very_base, self.num_categories[feature])
        for subpop in range(self.num_subpops):
            dist = base_dists[np.random.randint(0, len(base_dists))]  # Choose a base distribution for the subpopulation
            self._values_for_feature_based_feature(feature, subpop, dist, important, other_features)

    def create_distributions(self, subpop_specific, categories, base_dist, num_categories):
        bias = self.consts['random_dist_weight2']  # Number between 0 and 1 depicting how similar the distributions are
        base_dists = [{} for _ in range(np.random.randint(2, self.num_subpops + 3) if subpop_specific else 1)]
        # There will be a random number of base distributions if subpoop specific, otherwise just one.
        for combination in itertools.product(*categories):  # Defining different distributions for each combination
            for i in range(len(base_dists)):
                base_dists[i][combination] = random_dist(size=num_categories) * bias + \
                                             base_dist * (1 - bias)
        return base_dists

    def _values_for_feature_based_feature(self, feature, subpop, distribution, important, other_features):
        """
        Generates values for a feature that is based on other features.

        :param feature: The feature that is being generated
        :param subpop: The subpopulation that the feature is being generated for
        :param distribution: The distribution that the feature is based on, for each combination of values of other
        features that are deemed important.
        :param important: A list of booleans indicating which of the other features the feature is dependent on.
        :param other_features: A list of the other features that the feature might be dependent on.
        """
        self.values[self.subpops[subpop]][feature] = np.zeros(self.subpops_sizes[subpop], dtype=int)
        for i in range(self.subpops_sizes[subpop]):
            combination = tuple(self.values[self.subpops[subpop]][other_features[j]][i] for j in
                                range(len(other_features)) if important[j])
            self.values[self.subpops[subpop]][feature][i] = np.random.choice(self.num_categories[feature],
                                                                             p=distribution[combination])

    def generate_values(self):
        """
        Generates values for each feature in each subpopulation.
        """
        for i in range(10, len(self.feature_names)):
            feature = self.feature_names[i]
            choice = np.random.random()
            if choice < self.consts['prob_option'][0]:
                self.random_feature(feature)
            elif choice < self.consts['prob_option'][1]:
                self.feature_based_feature(feature, 'label')
            elif choice < self.consts['prob_option'][2]:
                self.feature_based_feature(feature, np.random.choice(self.feature_names[:i]))
            elif choice < self.consts['prob_option'][3]:
                earlier_feature = np.random.choice(self.feature_names[:i])
                self.feature_based_feature(feature, self.pointers[earlier_feature])
            elif choice < self.consts['prob_option'][4]:
                self.feature_based_feature(feature, *np.random.choice(self.feature_names[:i],
                                                                      size=np.random.randint(2, 5), replace=False))
            else:
                self.feature_based_feature(feature, *np.random.choice(self.feature_names[:i],
                                                                      size=np.random.randint(1, 4),
                                                                      replace=False), 'label')

    def create_file(self):
        df = pd.DataFrame(self.values['s0'])
        df['subpop'] = 's0'
        for subpop in self.subpops[1:]:
            new_df = pd.DataFrame(self.values[subpop])
            new_df['subpop'] = subpop
            df = pd.concat([df, new_df])
        df.to_csv(f'SyntheticData/synthetic_data_{self.name}.csv', index=False)
        with open(f'SyntheticData/synthetic_data_{self.name}_desc.txt', 'w') as txt:  # Create description file
            txt.write(self.desc)


def make():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=1, help='Name of the synthetic dataset')
    parser.add_argument('--num_features', type=int, default=1000, help='Number of features')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of possible labels')
    parser.add_argument('--num_samples', type=int, default=100000, help='Mean size of each subpopulation')
    parser.add_argument('--num_subpops', type=int, default=5, help='Number of subpopulations')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')

    args = parser.parse_args()

    x = SyntheticDataGenerator(args.name, args.num_features, args.num_labels, args.num_samples,
                               args.num_subpops, args.seed)
    x.generate_values()
    x.create_file()


if __name__ == '__main__':
    make()

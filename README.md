Synthetic Parameters Configs:
1. Subpop_size_std: Standard deviation of the subpopulation sizes.
2. Random_dist_weight1: Weight of the random distribution when creating a new distribution based on 
   base_distribution. This is regarding the distribution of the labels.
3. Random_dist_weight2: Weight of the random distribution when creating a new distribution for a feature that is 
   based on base_distribution. This is used for features, not labels. This is how much the feature distribution is 
   to be different in different subpopulations.
4. Max_bias: Determines the maximum "similarity" that a label-based feature can have to the distribution that the 
   label dictated for the feature. Number between 0 and 1, such that 0 would mean that the feature would have 
   nothing to do with the label, and 1 would mean that the bias could be any random number between 0 and 1.
5. Feature_important_prob: Probability of a feature being important. When we have a feature that is important, its 
   value will affect the label. This will be used when we have a feature that is based on more than one other 
   previous feature.
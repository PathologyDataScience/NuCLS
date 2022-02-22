```
|
└─── interrater : Python methods and scripts used for the interrater and intra-rater analysis.
|  |  
│  └─── scripts : scripts used to perform the interrater and intra-rater analysis and plots
|  |    └─── i1_get_all_nucleus_anchors.py : Obtain nucleus anchors using constrained agglomerative clustering and infer classification truth using expectation-maximization.
|  |    └─── i1b_get_krippendorph_summary.py : Ontaing Krippendorph alpha interrater agreement statistics
|  |    └─── i1c_get_accuracy_stats.py : Obtain detection accuracy statistics for individual participants and vaarious participant groups.
|  |    └─── i1d_get_interrater_and_intrarater_stats.py :  Obtain interrater and intrarater agreement statistics for detection and classification.
|  |    └─── i1e_run_NPs_accuracy_simulations.py : Run simulations to determine minimal number of participants needed to achieve desired accuracy of inferred truth.
|  |    └─── i1f_parse_anchors_dataset.py : Parse a ground truth dataset using consensus anchor locations and inferred true classifications.
|  |    └─── i2_show_effect_of_constrained_clustering.py : Investigate the impact of clustering constraint in the constrained agglomerative clustering approach used.
|  |    └─── i3_get_anchor_composition_summary.py : Get and plot a summary of the composition of the distribution and composition of inferred labels and classifications.
|  |    └─── i4_get_detection_and_classification_tally.py : An extension of i3_get_anchor_composition_summary.py
|  |    └─── i5_plot_participant_accuracy_stats.py : Plot participant accuracy results. 
|  |    └─── i6_plot_segmentation_accuracy_stats.py : Plot segmentation accuracy of nuclei that were determined to have accurate algorithmically-suggested segmentation boundary.
|  |    └─── i7_plot_participant_confusion.py : Plot confusion matrix of participant classifications.
|  |    └─── i8_plot_intrarater_stats.py : Plot intrarater statistics (self-agreement).
|  |    └─── i9_plot_interrater_stats.py : Plot interrater statistics.
|  |    └─── i10_plot_krippendorph_summary.py : Plot Krippendorph alpha values.
|  |    └─── i11_plot_NPs_accuracy_simulations.py : Plot the results of the simulations from i1e_run_NPs_accuracy_simulations.py
|  |    └─── i12_statistical_tests.py : Run statistical tests to compare various results and obtain p-values.
│  |
|  └─── DawidAndSkene1979_EMGtruthInference.py : This is code by Zheng et al, implementing the Expectation-Maximization based method for ground truth inference from multi-observer datasets, as proposed by Dawid and Skene in 1979.
|  └─── constrained_agglomerative_clustering.py : Our constrained agglomerative clustering implementation. Please refer to the paper and function documentation for details.
|  └─── interrater_utils.py : Various utilities to support the interrater analysis.
|  └─── krippendorff.py : Modified from Samtiago Castro, based on Thomas Grill implementation. Works on Python 3.5+.
│
```

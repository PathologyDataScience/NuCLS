# NuCLS crowdsourcing approach, dataset, and explainable deep-learning method

<img src="https://user-images.githubusercontent.com/22067552/140637808-3a827cc5-ff9e-44fe-973e-e4b7cf36a21c.png" width="100" />

This repository contains the codebase corresponding to the following two papers:

_Amgad M, Atteya LA, Hussein H, Mohammed KH, Hafiz E, Elsebaie MA, Alhusseiny AM, AlMoslemany MA, Elmatboly AM, Pappalardo PA, Sakr RA. **NuCLS: A scalable crowdsourcing approach & dataset for nucleus classification and segmentation in breast cancer.** (Under reveiw). (Access [ArXiv preprint](https://arxiv.org/abs/2102.09099))_

_Amgad M, Atteya LA, Hussein H, Mohammed KH, Hafiz E, Elsebaie MA, Mobadersany P, Manthey D, Gutman DA, Elfandy H, Cooper LA. **Explainable nucleus classification using Decision Tree Approximation of Learned Embeddings.** Bioinformatics. 2021 Sep 29._

Please consult these papers for details on the context in which these scripts and methods were used in our research to generate and analyse the NuCLS dataset.

__________________________________________________________________________

## Repository structure

Structure and high-level documentation of the repository is included below:

```
│
└─── Mask_RCNN @ 403afaf : This is our fork from the Matterport Mask-RCNN implementation. We used this implementation as-is to refine the algorithmic suggestions that the participants saw.
│
└─── algorithmic_suggestions : Python methods and scripts used to generate the algorithmic suggestions. 
|  |  
│  └─── jupyter_notebooks : Walk-through examples of the steps used to suggestion generation and refinement.
|  |    └─── jn1_inspect_bootstrapping_workflow_TCGA.ipynb : Sample walk-through from the generation of algorithmic suggestions using image processing.
|  |    └─── jn2_inspect_maskrcnn_training.ipynb : Training the Matterport Mask R-CNN implementation.
|  |    └─── jn4_inspect_maskrcnn_inference.ipynb : Inference using the Matterport Mask R-CNN model to produce refined algorithmic suggestions.
|  |    └─── jn5_inspect_integrate_maskrcnn_prediction_with_region_priors.ipynb : Integrating region priors from the BCSS region annotation dataset to produce more sensible algorithmic suggestions.
|  |    
│  └─── scripts : python scripts used to generate bootstrapped suggestions using classical image processing, refine those suggestions using Mask R-CNN, further improve refinement by integrating region prior knowledge from the BCSS region annotation dataset, and saving the refined suggestions to a database to be shown to participants.
|  |    └─── m1_bootstrap_nuclei_from_regions_TCGA.py : Obtaining nuclear boundaries and preliminary classification by bootstrapping using classical image processing.
|  |    └─── m2_train_TCGA_maskrcnn.py : Training Matterport Mask R-CNN model.
|  |    └─── m3_save_extra_TCGA_tiles_for_inference.py : Saving dataset for inference.
|  |    └─── m4_inference_TCGA_maskrcnn.py : Inference using the Matterport Mask R-CNN model to produce refined algorithmic suggestions.
|  |    └─── m5_integrate_maskrcnn_prediction_with_region_priors.py : Integrating region priors from the BCSS region annotation dataset to produce more sensible algorithmic suggestions.
|  |    └─── m6_save_bootstrap_to_db.py : Saving predictions in coordinate-form into an SQLite database to be visualized through the HistomicsUI interface.
│  |
|  └─── SQLite_Methods.py : Methods for SQLite databse parsing.
|  └─── bootstrapping_utils.py : Methods for generating algorithmic suggestions using classical image processing.
|  └─── configs_for_AlgorithmicSuggestions_MaskRCNN.py : Configurations used for the Matterport Mask R-CNN implementation.
|  └─── data_management.py : utilities used for data management and wrangling.
|  └─── maskrcnn_region_integration_utils.py : Methods used for integrating region priors to improve refined suggestions.
|  └─── maskrcnn_utils_local.py : Other utilities to facilitate Mas R-CNN training.
│
└─── configs : configurations used, including color and ground truth codes
|  └─── nucleus_GTcodes.csv : Ground truth codes for forming nucleus masks.
|  └─── nucleus_model_configs.py : Configurations used for the NuCLS model (our modified Mask R-CNN implementation, as described in the Bioinformatics paper).
|  └─── nucleus_style_defaults.py : Color coding scheme and other configurations used.
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
└─── nucls_model : Python methods used in our paper: "Amgad M, Atteya LA, Hussein H, Mohammed KH, Hafiz E, Elsebaie MA, Mobadersany P, Manthey D, Gutman DA, Elfandy H, Cooper LA. Explainable nucleus classification using Decision Tree Approximation of Learned Embeddings. Bioinformatics. 2021 Sep 29."
|  |  
|  └─── torchvision_detection_utils
|  |    └─── ... Minimally-modified methods from the official torchvision implimentation 
|  └─── BackboneSwitcher.py
|  └─── DTALE.py
|  └─── DataFormattingUtils.py
|  └─── DataLoadingUtils.py
|  └─── FasterRCNN.py
|  └─── FeatureExtractor.py
|  └─── GeneralizedRCNN.py
|  └─── MaskRCNN.py
|  └─── MiscUtils.py
|  └─── ModelRunner.py
|  └─── NucleusWorkflows.py
|  └─── PartialMaskRCNN.py
|  └─── PlottingUtils.py
|  └─── ROIHeads.py
|  
└─── GeneralUtils.py
└─── TorchUtils.py
└─── wsi-conda-env-specs.txt

```
__________________________________________________________________________

## Requirements

Conda specs are exported in the file `wsi-conda-env-specs.txt`, which can be used to generate a conda environment.

Minimal requirements to run the interrater analysis: `numpy`, `scipy`, `pandas`, `sqlite`, `matplotlib`, and `seaborn`.
Other requirements are needed for the algorithmic suggestion generation, including `HistomicTK`, `tensorflow`, and the Matterport Mask R-CNN implementation, included here as a fork.
To run the NuCLS model (described out Bioinformatics paper), you also need `torch`.

__________________________________________________________________________

## Summary of contributions

We describe the following contributions:

### 1. [NuCLS datasets](https://sites.google.com/view/nucls)  

<img src="https://user-images.githubusercontent.com/22067552/140637897-87adddc5-b9e3-4151-8937-844202b56530.png" width="400" />

Over 220,000 labeled nuclei from breast cancer images from TCGA; one of the largest datasets for nucleus detection, classification and segmentation of hematoxylin and eosin-stained digital slides of breast cancer. These nuclei were annotated through the collaborative effort of pathologists, pathology residents, and medical students using the Digital Slide Archive. These data can be used in several ways to develop and validate algorithms for nuclear detection, classification, and segmentation, or as a resource to develop and evaluate methods for interrater analysis. Data from both single-rater and multi-rater studies are provided. For single-rater data we provide both pathologist-reviewed and uncorrected annotations. For multi-rater datasets we provide annotations generated with and without suggestions from weak segmentation and classification algorithms.

### 2. A novel crowdsourcing framework

<img src="https://user-images.githubusercontent.com/22067552/140638162-c57c78f6-8b7e-4736-ba52-a468cf315895.png" width="600" />

This paper describes a novel collaborative framework for engaging crowds of medical students and pathologists to produce quality labels for cell nuclei. This builds on [prior work](https://academic.oup.com/bioinformatics/article/35/18/3461/5307750) labeling tissue regions to produce an integrated tissue region- and cell-level annotation dataset for training that is the largest such resource for multi-scale analysis of breast cancer histology. This paper presents data and analysis results for single and multi-rater annotations from both non-experts and pathologists. We present a novel method for suggesting annotations that
allows us to collect accurate segmentation data without the need for laborious manual tracing of cells. Our results indicate that
even noisy algorithmic suggestions do not adversely affect pathologist accuracy, and can help non-experts improve annotation
quality. We also present a new approach for inferring truth from multiple raters, and show that non-experts can produce accurate
annotations for visually distinctive classes.

### 3. Mask R-CNN improvements

<img src="https://user-images.githubusercontent.com/22067552/140638550-76f88308-bcd2-4f56-a5ea-792fbb45ba30.png" width="600" />

We show how modifications to the widely used Mask R-CNN architecture, including decoupling the detection and classification tasks, improves accuracy and enables learning from hybrid annotation datasets like NuCLS, which contain mixtures of bounding boxes and segmentation boundaries. 

### 4. Decision Tree Approximation of Learned Embeddings

<img src="https://user-images.githubusercontent.com/22067552/140638638-1c3a3a14-c61d-43b7-ae9c-f0fabda981a7.png" width="600" />

We introduce an explainability method called Decision Tree Approximation of Learned Embeddings (DTALE), which provides explanations for classification model behavior globally, as well as for individual nuclear predictions. DTALE explanations are simple, quantitative, and can flexibly use any measurable morphological features that make sense to practicing pathologists, without sacrificing model accuracy.


## Dataset Licensing
This dataset itself is licensed under a [CC0 1.0 Universal (CC0 1.0) license](https://creativecommons.org/publicdomain/zero/1.0/). 
We would appreciate it if you cite our paper if you use the data.

## Code licensing
Thise codebase is licensed with an MIT license.

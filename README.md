# NuCLS crowdsourcing and explainable deep-learning approach

<img src="https://user-images.githubusercontent.com/22067552/140637808-3a827cc5-ff9e-44fe-973e-e4b7cf36a21c.png" width="100" />

This repository contains the codebase corresponding to the following two papers:

_Amgad M, Atteya LA, Hussein H, Mohammed KH, Hafiz E, Elsebaie MA, Alhusseiny AM, AlMoslemany MA, Elmatboly AM, Pappalardo PA, Sakr RA. **NuCLS: A scalable crowdsourcing approach & dataset for nucleus classification and segmentation in breast cancer.** (Under reveiw). (Access [ArXiv preprint](https://arxiv.org/abs/2102.09099))

_Amgad M, Atteya LA, Hussein H, Mohammed KH, Hafiz E, Elsebaie MA, Mobadersany P, Manthey D, Gutman DA, Elfandy H, Cooper LA. **Explainable nucleus classification using Decision Tree Approximation of Learned Embeddings.** Bioinformatics. 2021 Sep 29._

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

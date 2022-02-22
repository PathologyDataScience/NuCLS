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

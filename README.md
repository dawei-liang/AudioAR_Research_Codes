# Audioset_research

Source codes of the Audio Set activity recognition research

Source training data can be accessed from on line. Details can be seen on our paper.

We welcome you to adopt our work for your own research. If you would like to refer it, we appreciate your citation to: 

@article{liang2019audio,
  title={Audio-Based Activities of Daily Living (ADL) Recognition with Large-Scale Acoustic Embeddings from Online Videos},
  author={Liang, Dawei and Thomaz, Edison},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={3},
  number={1},
  pages={17},
  year={2019},
  publisher={ACM}
}

Thank you!


                      # ============================================================================== #

Instruction:

Some of the codes are written in the form of IPython notebook and some are in Python script. This is mainly because this project has been developed for several different stages and versions and on various systems. The final version of the project was run on the Texas Advanced Computing Center (TACC) Maverick (Linux) system, based on NVIDIA K40 GPU and Tensorflow-gpu 1.0.0.

Oversampling using SMOTE and random method: _main_resample.py (loading_data_and_labels.py, resampling.py)_

Loading data and training the classification network/CNN dedicated study: _cnn.ipynb_

Loading data and training the baseline (Random Forest)/baseline dedicated study: _rf.ipynb_

Show the distribution of predicted confidence: _predict_class_confidence.py_

Cross validation on user data: _validate_user_data_resample.py_, _validate_user_data_trainCNN.py_

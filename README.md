# Visual Map Embedding with Neural Networks
This project was made for CS230 at Stanford University. This project aims to create a neural network model capable of creating a dense embedding of a scene from visual images that can be used for robotics tasks such as mapping and localization.

## Requirements
This project uses `pybullet` for data generation, `tensorflow 2` for the neural network models, and `matplotlib` for visualizations.

## Data Generation
To generate training data, first download the "IKEA 3D models" from http://ikea.csail.mit.edu/. Next, unzip the archive in this directory and there should be a folder named `IKEA`. Next, run `organize_models.py`, to extract only the usable models into a more convienient file scheme. Now you can run `scene_builder.py` to generate 20000 scenes split into chunks of 200. The resulting filesystem will look like `data/datasets/chunkX/sceneY/` ofr each scene. Each scene folder contains 64 PNG images from the scene as well as a NumPy archive containing the poses from which each image was taken and a binary map of the scene.

Creating a dev set must be done manually by moving any number of `chunk` floders from `data/datasets/` to `data/dev/`. Thus you will have `data/dev/chunkX/sceneY` for as many chunks as desired. `dataloader.py` can load data from either set using `create_dataset(folder)`, with `folder` as in `data/folder/chunkX/.../`.

## Training Models
Use `train.py` to train models. The models are defined in files in the `architectures` directory. They can be imported and trained, and weight checkpoints will be saved every epoch. Running and training the baseline model is a little different because it eas defined more simply.

## Seeing Results
The `evaluate.py` and `visualize.py` help with evaluation of the models. `evaluate.py` runs a model on some data from the dev set and puts the results as well as the imput data used in a folder called `visuals`. `visualize.py` can be modified slightly to veiw different outputs and labels using `matplotlib`.
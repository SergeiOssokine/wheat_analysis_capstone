# MLZoomcamp 2025 final project: Classifying grain states

## Introduction and problem description

Cereal and cereal-based foods, such as bread (from wheat) or rice, has immense importance as staple foods in many regions. Although individual grains are miniscule, it is nevertheless of vital importance that inspection is carried out to a high degree of accuracy: we need to sort out flaws in grains (sprouted grains, moldy grains) and we may also need to sort out different kinds of grains. This is a challenging task well-suited for automated classification by computer vision, both to facilitate the detection of subtle flaws and to distinguish between groups of variants.

We use the [GrainSet dataset ](https://www.nature.com/articles/s41597-023-02660-8), a publicly-available, high resolution, highly-curated dataset with [350k] images of wheat, sorghum, rice, and maize collected from over 20 regions in 5 countries. We train 4 CNNs on the wheat dataset (comprised of 200k images) to classify states of grain into classes (such as e.g. normal, broken, moldy) and achieve state-of-the-art accuracy >99.8% F1 score on validation set. From these 4 models we select a winning candidate, assess its accuracy and also present an example prediction service for using the selected candidate model.

See [below](exploring-the-dataset) for instructions on downloading the data set.

### Requirements

In order to run everything in this project you will need the following:

- Docker >= 28.5.0
- GNU `make` >=4.3
- `bash` >=5.0
- git-lfs


To simplify various stages we use `make` commands. Run

```bash
make help
```

to see the available commands. In case `make` is unavailable, you can manually run the commands specified in the `Makefile`.

Note that this project has been tested in a Linux environment and should also run fine on MacOS. For Windows, we recommend running it in [WSL](https://learn.microsoft.com/en-us/windows/wsl/about).

## Setting up for development

To set up the `python` environment and all dependencies run

```bash
make setup_env
```

This will install `uv` (if not present) and install all dependencies needed

**Important**: The development environment by default is setup to use the CPU, to be agnostic. The precise development environment needed for GPU training and inference depends on the GPU accelerator being used.

- For instructions for nVidia CUDA, see e.g. [here](https://hpc.llnl.gov/documentation/user-guides/using-pytorch-lc/pytorch-nvidia-gpu-systems-quickstart-guide)
- For instructions for AMD ROCm, see [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html).  

For this project we used an AMD RX9060XT GPU, using `ROCm` version: 7.1.1 and  `amdgpu` version: 6.16.6 with the following exact Python packages:

```
torch==2.11.0.dev20260115+rocm7.1
torchvision==0.25.0.dev20260116+rocm7.1
```


## Exploring the dataset
**Note that the dataset used for this project is large (~20 GB) so it can take a long time to download.**

To download the dataset, navigate a level _above_ this repo and do

```bash
mkdir  GrainSetData
```

Then download 2 files from [here](https://figshare.com/articles/figure/wheat_zip/22992317/2). You will need to download both `wheat.zip` and
`wheat.xml` and place them inside `GrainSetData`. Finally do

```bash
cd GrainSetData
unzip -q wheat.zip
```


You can find the EDA notebook [here](./notebook.ipynb). This notebook contains:

- details about the dataset, its characteristics, 
- training and tuning of several deep learning classification models and assessing their performance
- picking a final model to use. We find that `resnet10` is the best model.
See the notebook for more details.

## Training different models

Training deep learning models can take a long time. As such training all the models is done via scripts that checkpoint and can be resume.
For example to train the `regnety_008` model do
```bash
make train_resnet10
```
This will produce something like the following:



To save time, we have added checkpoint files with models we have pre-trained in the repository, under `checkpoints`. These checkpoints are explicitly used in the EDA notbeook. To obtain them, you will need to use `git-lfs` to pull the data:

```bash
git lfs pull
```


## Building and deploying a prediction service

We create a FastAPI-based prediction service that runs inside a Docker container. To build the container run

```bash
make build_prediction_service
```

This will create an image `grain_prediction_service:` which will contain the trained model and the web-based prediction service

To launch the prediction service run

```bash
make serve_predictions
```

You may need to give the service a few seconds to start up. To test with a sample payload, run

```bash
make test_prediction_service
```

This command runs the script `test_prediction_service.py` with a payload which is an image from the `sample_images` folder (which comes from the test set), and returns the predicted class. You can change the image and run the script by hand with a different one, e.g.

```bash
uv run python test_prediction_service.py --image-path ./sample_images/7_IM/Grainset_wheat_2021-05-13-10-50-06_22_p600s.png 
```



## Cleaning up

To stop the prediction service run

```bash
make shutdown_prediction_service
```

To remove the service Docker container entirely, do

```bash
make remove_prediction_container
```

Finally to get rid of the Python virtual environment, run

```bash
make cleanup_venv
```

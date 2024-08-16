## A pipeline for generating acoustic pressure data

This is the codes for the data generation of PAWS-Dataset, which is a  dataset for room acoustic simulation based on wave acoustics simulation. We provide a sample pipeline in pipeline.py, you may use it to give it a try on making your own data



### 1. Install dependencies

```bash
# Install Python environment.
conda create --name PAWS python=3.9

# Activate environment.
conda activate PAWS

# Install Python packages dependencies.
pip install -r requirement.txt
```





### 2. Give it a try

You may directly run our pipeline.py file, this would generate a 256*256 shoebox data with 10000 frames.  By commenting and uncommenting code for the scene generation part, you can generate polygon room data, too. Please note that the first run will take relevantly lone time because the code will download the executable file of the solver.

The generated file looks like this, you can check the image file for environment definition and mp4 file for the outcome.

```
generated_data
├── image_file
├── mp4_file
└── metadata
```


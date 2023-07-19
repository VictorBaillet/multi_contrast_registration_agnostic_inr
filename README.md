# Using Implicit Neural Representation for super-resolution and registration simultaneously
<img src="https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/assets/105466709/e4f2e117-3f6c-4e2a-b8db-27b62d67c246" width="800"> 

![reg_ball](https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/assets/105466709/99bbfaab-85b8-4ce8-9406-1f5ef8071b09)

## Installation

Clone the repository:
~~~
git clone https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr.git
cd multi_contrast_registration_agnostic_inr
~~~

Then install the required libraries:
~~~
pip install -r requirements.txt
~~~

## Project description

## Data

Put the images in data/raw_data file.
For a simple proof of concept you can test the code with the "balls" : [data](https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/releases/tag/large_files). 

## Launch a training

In the command line : (replace "config_balls" by the name of your project's configuration file)

`python main.py --experiment_name parallel_registration --config config_balls.yaml --logging` 

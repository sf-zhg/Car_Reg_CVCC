# Multi-Task Regression

# 1. Download this Repository:
```
git clone https://github.com/sf-zhg/Car_Reg_CVCC.git
cd APPLIED_DL
```
# 2. Create a virtual environment:
create a virtual environment for to run this repository on and install dependencies. 
```
python -m venv working_environment
source working_environment/bin/activate
pip install -r requirements.txt
```
Alternatively, one can use conda to create a virtual environment:
```
conda create -n working_environment python=3.9
source activate working_environment
conda install --file requirements.txt
```
Note that ```source activate``` does not work on WindowsOS and has to be substituted by ```conda activate```

Then install the Persp_estimator with:
```pip install -r requirements.txt
!python setup.py install ```

# 3. Run Experiments via:
``` python run_experiment/run_experiment.py ```

make sure to set the working directory appropriately!



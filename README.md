# MLOps Pipeline Demo

## Project Description


## Data Description
- **Employee_ID**: Unique identifier for each employee.
- **Age**: Age of the employee.
- **Gender**: Gender of the employee.
- **Marital_Status**: Marital status of the employee (Single, Married, Divorced).
- **Department**: Department the employee works in (e.g., HR, IT, Sales, Marketing).
- **Job_Role**: Specific role within the department (e.g., Manager, Analyst).
- **Job_Level**: Level in the organizational hierarchy.
- **Monthly_Income**: Monthly salary of the employee.
- **Hourly_Rate**: Rate per hour for hourly employees.
- **Years_at_Company**: Number of years the employee has been with the company.
- ****Years_in_Current_Role**: Number of years the employee has been in their current role.
- **Years_Since_Last_Promotion**: Time since the employee’s last promotion.
- **Work_Life_Balance**: Rating of work-life balance.
- **Job_Satisfaction**: Rating of job satisfaction (1-5 scale).
- **Performance_Rating**: Performance rating (1-5 scale).
- **Training_Hours_Last_Year**: Number of training hours completed in the past year.
- **Overtime**: Whether the employee works overtime (Yes/No).
- **Project_Count**: Number of projects managed by the employee.
- **Average_Hours_Worked_Per_Week**: Average working hours per week.
- **Absenteeism**: Number of days the employee was absent in the past year.
- **Work_Environment_Satisfaction**: Rating of work environment satisfaction.
- **Relationship_with_Manager**: Rating of the relationship with the manager.
- **Job_Involvement**: Rating of job involvement.
- **Distance_From_Home**: Distance from home to the workplace (in kilometers).
- **Number_of_Companies_Worked**: Total number of companies the employee has worked for.
- **Attrition**: The target column (Yes/No) indicating whether the employee left the company.

## Project Structure
- `configs`: YAML configuration for training models.
- `scripts`: Scripts to run from command line to run and compare experiments.
- `src`: Python modules to preprocess data, build, train, evaluate model and moniter drift
- `test`: Pytest test suite for data, model and preprocessing.
- `data`: Dataset files - Not tracked by git.
- `notebooks`: Data analysis notebook.
- `reports`: Drift monitoring reports.
- `mlruns`: model artifacts from experiments runs.

## Set up

Requires Python 3.12 (matches CI). If you don't have it installed, [uv](https://github.com/astral-sh/uv) can install it for you:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
```

Clone the repo and create a virtual environment:
```
git clone <repo-url>
cd mlops-pipeline-demo
uv venv --python 3.12 venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```


### Run Experiments

Train the model and run multiple experiments
```
python scripts/run_experiments.py 
```

### Compare Experiments

Compare experiments and get the best run
```
python scripts/compare_experiments.py 
```

### Run Test Suite
```
pytest tests/ -v
```
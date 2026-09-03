import sys
sys.path.insert(0, "src")

from src.preprocess_data import load_data

DATA_PATH = 'data/processed/employee_attrition_dataset.csv'

def test_expected_features() -> None:
    
    df = load_data(DATA_PATH)
    expected = {
        'Age',
        'Gender',
        'Marital_Status',
        'Department', 
        'Job_Role',
        'Job_Level',
        'Monthly_Income',
        'Hourly_Rate',
        'Years_at_Company',
        'Years_in_Current_Role',
        'Years_Since_Last_Promotion',
        'Work_Life_Balance',
        'Job_Satisfaction',
        'Performance_Rating',
        'Training_Hours_Last_Year',
        'Overtime',
        'Project_Count',
        'Average_Hours_Worked_Per_Week',
        'Absenteeism',
        'Work_Environment_Satisfaction',
        'Relationship_with_Manager',
        'Job_Involvement',
        'Distance_From_Home',
        'Number_of_Companies_Worked',
        'Attrition'
    }
    assert expected.issubset(set(df.columns))

def test_exptected_target_values() -> None:
    df = load_data(DATA_PATH)

    assert set(df['Attrition'].dropna().unique()) == {'No', 'Yes'}


def test_numeric_features_range() -> None:
    df = load_data(DATA_PATH)

    assert(df['Age']).dropna().between(15, 65).all()
    assert(df['Years_at_Company']).dropna().between(1, 50).all()

def test_ordinal_features_range() -> None:
    df = load_data(DATA_PATH)

    assert(df['Job_Level']).dropna().between(1, 5).all()
    assert(df['Work_Life_Balance']).dropna().between(1, 5).all()

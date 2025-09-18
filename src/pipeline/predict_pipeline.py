import sys,os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 gender:str,
                 lunch:str,
                 reading_score: int,
                 writing_score:int,
                 parental_level_of_education:str,
                 race_ethnicity:str,
                 test_preparation_course:str):
    
        self.gender=gender
        self.lunch=lunch
        self.reading_score=reading_score
        self.writing_score=writing_score
        self.parental_level_of_education=parental_level_of_education
        self.race_ethnicity=race_ethnicity
        self.test_preparation_course=test_preparation_course
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "lunch":[self.lunch],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score],
                "parental_level_of_education":[self.parental_level_of_education],
                "race_ethnicity":[self.race_ethnicity],
                "test_preparation_course":[self.test_preparation_course]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
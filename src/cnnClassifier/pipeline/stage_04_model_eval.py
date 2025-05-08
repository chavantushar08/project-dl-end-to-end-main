import os 
import sys
sys.path.append(os.getcwd())   


from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_eval import Evaluation
from src.cnnClassifier import logger

import mlflow 


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/chavantushar08/project-dl-end-to-end-main.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"]="chavantushar08"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="1e9dc24a1031ae97b82b275e4a7b06a6fc4af344"

        #set mlflow tracking url 
        mlflow.set_tracking_uri("https://dagshub.com/chavantushar08/project-dl-end-to-end-main.mlflow") 




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
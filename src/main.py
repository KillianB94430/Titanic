import titanic_pred
import titanic_train

def main():
    print("Running Titanic Training ")
    titanic_train.run_pipeline_train()
    
    print("Running Titanic Prediction ")
    titanic_pred.run_pipeline_pred()

if __name__ == '__main__':
    main()

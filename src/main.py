import titanic_pred
import titanic_train

def main():
    print("Running Titanic Training ")
    titanic_train.main()
    
    print("Running Titanic Prediction ")
    titanic_pred.main()

if __name__ == '__main__':
    main()

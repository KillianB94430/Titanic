import titanic_pred
import titanic_train


# P2 : Renommer les fonctions et modules main avec des noms explicites et significatifs
def main():
    print("Running Titanic Training ")
    titanic_train.main()

    print("Running Titanic Prediction ")
    titanic_pred.main()


if __name__ == "__main__":
    main()

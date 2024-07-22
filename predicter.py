from clearml import Task, Model, Dataset
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

models = {
    "Decision Tree (J48)": DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, ccp_alpha=0.25),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt', random_state=1),
    "Support Vector Machine": SVC(C=1.0, kernel='rbf', tol=0.001, max_iter=100),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=1, weights='uniform'),
    "Logistic Regression": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.3, momentum=0.2, max_iter=200),
}

# Provides classification metrics for the input ML-based models 
def test_models(models, X_train, X_test, y_train, y_test):
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Print Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Print Classification Report
        cr = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(cr)

def minmax(latent_train,latent_test,y_train,y_test):

    print("\nMin-max normalization")

    scaler = MinMaxScaler()
    train_minmax = scaler.fit_transform(latent_train)
    test_minmax = scaler.transform(latent_test)

    test_models(models, train_minmax, test_minmax, y_train, y_test)

def standardization(latent_train,latent_test,y_train,y_test):

    print("\nStandardization")
    
    scaler = StandardScaler()
    train_std = scaler.fit_transform(latent_train)
    test_std = scaler.transform(latent_test)

    test_models(models, train_std, test_std, y_train, y_test)

if __name__ == '__main__':

    task = Task.init(project_name='clearml-init', task_name='Autoencoder prediction')

    # Parse input arguments
    parser = argparse.ArgumentParser(description="Script to do predictions using a model ID")
    parser.add_argument('--model_id', type=str, help='model ID input')

    args = parser.parse_args()
    model_id = str(args.model_id)
    task.execute_remotely(queue_name="default")

    # load dataset from clearml
    dataset = Dataset.get('e3c4b1f5295f4210a18fc9fd7a49b089',alias='dataset7030v2')
    file_path = dataset.get_local_copy()

    df_train = pd.read_csv(file_path + "/" + dataset.list_files()[1])
    y_train = np.array(df_train['category'])
    del df_train['category']
    train = np.array(df_train.to_numpy())
    train = train.reshape(train.shape[0],3,2,1)

    df_test = pd.read_csv(file_path + "/" + dataset.list_files()[0])
    y_test = np.array(df_test['category'])
    del df_test['category']
    test = np.array(df_test.to_numpy())
    test = test.reshape(test.shape[0],3,2,1)

    # Retrieve the model by using the model ID (no task_id)
    model = Model(model_id=model_id)

    # Download the model to a local path
    model_path = model.get_local_copy()
    print(f'Model is downloaded to: {model_path}')

    ae = load_model(model_path)

    encoder = ae.get_layer('enc_1')

    encoder.summary()
    
    # Features derived by latent space

    latent_train = encoder.predict(train)
    latent_test = encoder.predict(test)

    print("\nResults without scaling")
    test_models(models, latent_train, latent_test, y_train, y_test)

    minmax(latent_train,latent_test,y_train,y_test)
    standardization(latent_train,latent_test,y_train,y_test)
import time
import pickle
from sklearn.svm import LinearSVC


def train_svc(X_train, y_train, X_scaler):
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    t=time.time()
    svc_pickle = {}
    svc_pickle["svc"] = svc
    svc_pickle["X_scaler"] = X_scaler
    pickle.dump( svc_pickle, open( "svc.p", "wb" ) )
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to save SVC to pickle...')

    return svc

def test_svc(X_test, y_test):
    svc, X_scaler = load_svc()

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

def load_svc():
    svc_pickle = pickle.load( open( "svc.p", "rb" ) )
    svc = svc_pickle["svc"]
    X_scaler = svc_pickle["X_scaler"]
    return svc, X_scaler
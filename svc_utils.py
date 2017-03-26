import time
import pickle
from sklearn.svm import LinearSVC

# Function to train SVC and save to file
def train_svc(X_train, y_train, X_scaler, fname="svc.p"):

    # Use a linear SVC 
    svc = LinearSVC()

    # Fit training data to SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Save to pickle file
    svc_pickle = {}
    svc_pickle["svc"] = svc
    svc_pickle["X_scaler"] = X_scaler
    pickle.dump( svc_pickle, open( fname, "wb" ) )

    return svc

# Function to test SVC
def test_svc(X_test, y_test, fname="svc.p"):

    # Load svc from Pickle file
    svc, X_scaler = load_svc(fname)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Function to load svc from pickle
def load_svc(fname="svc.p"):
    svc_pickle = pickle.load( open( fname, "rb" ) )
    svc = svc_pickle["svc"]
    X_scaler = svc_pickle["X_scaler"]
    return svc, X_scaler
import pandas as pd
import numpy as np

print("loading datasets...")
# reading train and test files and converting datetime column to actual datetime format
train = pd.read_csv("train.csv", parse_dates=["datetime"])
test  = pd.read_csv("test.csv",  parse_dates=["datetime"])


# function for preprocessing
# extracting time related features and keeping only useful columns
def preprocessing(data):
    d = data.copy()
    d['hour']  = d['datetime'].dt.hour          # extracting hour
    d['day']   = d['datetime'].dt.dayofweek     # day of week
    d['month'] = d['datetime'].dt.month         # month
    d['year']  = d['datetime'].dt.year          # year

    # keeping only the features we need for training
    features=['hour','day','month','year','season','holiday','workingday',
              'weather','temp','atemp','humidity','windspeed']
    return d[features]


# custom lasso function using coordinate descent
# iterates over each feature and updates weight
# alpha controls how strongly features get pushed to zero
def fast_lasso(X, y, alpha, iters=50, w_init=None):
    if w_init is None:
        w = np.zeros(X.shape[1])        # starting weights as zero initially
    else:
        w = w_init.copy()               # or using previous weights if passed

    col_norm = (X**2).sum(axis=0)       # precomputing column norm for speed

    for _ in range(iters):
        for j in range(len(w)):
            r = y - (X @ w) + (w[j] * X[:, j])   # residual excluding current weight
            rho = np.dot(X[:, j], r)

            if j == 0:                 # intercept should not be regularized
                w[j] = rho / col_norm[j]
            else:
                # soft thresholding used in lasso
                if rho > alpha:
                    w[j] = (rho - alpha) / col_norm[j]
                elif rho < -alpha:
                    w[j] = (rho + alpha) / col_norm[j]
                else:
                    w[j] = 0
    return w


# main hybrid model
# first tries ridge to get a stable initial solution
# then refines it using lasso to reduce irrelevant coefficients
def choose_best_model(Xtr, ytr, Xts, yts):
    I = np.eye(Xtr.shape[1])
    sst = np.sum((yts - np.mean(yts))**2)   # total variance for computing r2

    ridge_alphas=[1e-6,1e-4,1e-3,1e-2,0.1]
    best=("ridge",None,1e18,-1e9)           # storing best model results

    # first checking multiple ridge regularization values
    for a in ridge_alphas:
        A = Xtr.T @ Xtr + a*I
        b = Xtr.T @ ytr
        w = np.linalg.solve(A, b)

        pred = Xts @ w
        mse = np.mean((yts - pred)**2)
        r2  = 1 - (np.sum((yts - pred)**2) / sst)

        if r2 > best[3]:
            best=("ridge", w, mse, r2)

    # then applying lasso using ridge weights as warm start
    lasso_alphas=[0.001,0.01,0.05,0.1]
    w0 = best[1]

    for a in lasso_alphas:
        w = fast_lasso(Xtr, ytr, a, iters=50, w_init=w0)
        pred = Xts @ w
        mse = np.mean((yts - pred)**2)
        r2  = 1 - (np.sum((yts - pred)**2) / sst)

        if r2 > best[3]:
            best=("lasso", w, mse, r2)
            w0 = w

    return best[1], best[2], best[3]


# preprocessing train data
X = preprocessing(train)
y = train['count']

# splitting into train and test manually (80/20)
X_test = X.sample(frac=0.2,random_state=67)
y_test = y.loc[X_test.index]
X_train = X.drop(X_test.index)
y_train = y.loc[X_train.index]

# converting to numpy for calculations
Xtr = X_train.values; Xts = X_test.values
ytr = y_train.values; yts = y_test.values

# normalizing feature values for stability
mean = Xtr.mean(axis=0)
std  = Xtr.std(axis=0); std[std==0] = 1
Xtr = (Xtr - mean) / std
Xts = (Xts - mean) / std

# generating polynomial features (degree 2) without interactions
Xtr_f = np.hstack([np.ones((len(Xtr),1)), Xtr, Xtr**2])
Xts_f = np.hstack([np.ones((len(Xts),1)), Xts, Xts**2])

# training baseline model
w,mse,r2 = choose_best_model(Xtr_f,ytr,Xts_f,yts)
print("initial deg2 mse:",mse)
print("initial deg2 r2:",r2)


# adding interaction terms one by one and checking if it improves accuracy
xtr_int = np.empty((len(Xtr),0))
xts_int = np.empty((len(Xts),0))
pairs=[]     # storing selected feature pairs

all_pairs = [(i,j) for i in range(12) for j in range(i+1,12)]  # total possible pairs

for (i,j) in all_pairs:
    col_tr = (Xtr[:,i]*Xtr[:,j]).reshape(-1,1)
    col_ts = (Xts[:,i]*Xts[:,j]).reshape(-1,1)

    Xtr_try = np.hstack([Xtr_f, xtr_int, col_tr])
    Xts_try = np.hstack([Xts_f, xts_int, col_ts])

    w_try,mse_try,r2_try = choose_best_model(Xtr_try,ytr,Xts_try,yts)

    print(f"trying ({i},{j}) -> mse={mse_try:.5f} r2={r2_try:.5f}")

    # keeping a pair only when model improves
    if mse_try < mse:
        print(f"keeping ({i},{j})")
        xtr_int = np.hstack([xtr_int, col_tr])
        xts_int = np.hstack([xts_int, col_ts])
        pairs.append((i,j))
        w,mse,r2 = w_try,mse_try,r2_try
    else:
        print(f"discarding ({i},{j})")


# training final model including selected interactions
print("\nfinal results:")
Xtr_final = np.hstack([Xtr_f,xtr_int]) if xtr_int.size else Xtr_f
Xts_final = np.hstack([Xts_f,xts_int]) if xts_int.size else Xts_f

w,mse,r2 = choose_best_model(Xtr_final,ytr,Xts_final,yts)

print("final mse:",mse)
print("final r2:",r2)


# printing regression equation properly
print("\nfinal regression equation:\n")

feature_names = ['bias'] + list(X.columns) + [f"{c}^2" for c in X.columns]
for (i,j) in pairs:
    feature_names.append(f"{X.columns[i]}*{X.columns[j]}")

equation = "count = "
for idx,coef in enumerate(w):
    if idx==0:
        equation += f"{coef:.4f}"
    else:
        sign=" + " if coef>=0 else " - "
        equation += f"{sign}{abs(coef):.4f}*{feature_names[idx]}"

print(equation,"\n")


# making predictions for test.csv
Xt=preprocessing(test).values
Xt=(Xt-mean)/std

# polynomial features again
Xb=np.hstack([np.ones((len(Xt),1)),Xt,Xt**2])

# adding same interaction terms selected earlier
final=[]
for i,j in pairs:
    final.append((Xt[:,i]*Xt[:,j]).reshape(-1,1))
final=np.hstack(final) if final else np.empty((len(Xt),0))
Xt_full=np.hstack([Xb,final])

pred = np.maximum(0, Xt_full@w)   # count cannot be negative


# saving submission file
sample=pd.read_csv("sampleSubmission.csv")
sample["count"]=pred
sample.to_csv("submission.csv",index=False)

print("done successfully")

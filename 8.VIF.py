#Check multicollinearity
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(X_train.drop(columns = ['Product Category','Number of Customer Images'], axis=1))
calc_vif(X_test.drop(columns = ['Product Category','Number of Customer Images'], axis=1))

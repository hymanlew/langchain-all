# Python + Scikit-learn + XGBoost 技术栈实现客户流失预测场景

# ====================== 数据准备与特征工程 ======================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 加载数据（企业实际从数据库读取）
df = pd.read_csv('customer_churn.csv')

# 定义特征类型
numeric_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['gender', 'partner', 'internet_service']

# 构建特征工程管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 划分数据集
X = df.drop('churn', axis=1)
y = df['churn'].map({'Yes':1, 'No':0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# =============== 建模与调优（XGBoost + 超参数搜索） ==========
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 构建完整管道
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss'))
])

# 定义超参数网格
param_grid = {
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5],
    'classifier__n_estimators': [100, 200]
}

# 网格搜索（企业实际用贝叶斯优化更高效）
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# 评估最佳模型
best_model = grid_search.best_estimator_
print(f"Best AUC: {grid_search.best_score_:.3f}")


# ====================== 模型评估与解释 ======================
from sklearn.metrics import classification_report, roc_auc_score
import shap

# 预测测试集
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

# 打印评估报告
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

# SHAP解释（企业必备）
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
shap_values = explainer.shap_values(preprocessor.transform(X_train))
shap.summary_plot(shap_values, X_train)

# ====================== 生产部署（FastAPI + Joblib） ======================
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 保存模型
joblib.dump(best_model, 'churn_model.pkl')

# 定义API
app = FastAPI()

class CustomerData(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    gender: str
    partner: str
    internet_service: str

@app.post("/predict")
async def predict(data: CustomerData):
    model = joblib.load('churn_model.pkl')
    df = pd.DataFrame([data.dict()])
    proba = model.predict_proba(df)[0,1]
    return {"churn_probability": float(proba), "risk_level": "high" if proba > 0.7 else "low"}
		
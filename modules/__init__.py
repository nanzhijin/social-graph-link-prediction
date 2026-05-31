"""
模块化链接预测框架

三类模块，标准接口，可独立替换和交叉搭配:

    FeatureSelector  —— 从119维中选/降出最优子集
        ├─ IdentitySelector   (全保留, 119维)
        ├─ SHAPSelector       (SHAP top-K)
        ├─ PCAReducer          (PCA 降维)
        └─ SHAPPCASelector    (SHAP → PCA 串行)

    Model            —— 输入特征矩阵, 输出预测分数
        ├─ LGBModel           (LightGBM, 支持 cat_cols)
        ├─ GNNModel           (GraphSAGE/GAT, 待实现)

    Experiment       —— 交叉组合 + AB 对比
        └─ ExperimentRunner   (config 驱动, 自动记录 metrics)

用法:
    from modules import SHAPSelector, LGBModel, ExperimentRunner

    selector = SHAPSelector(k=50)
    model_a = LGBModel(feature_cols=cols_without_friend)
    model_b = LGBModel(feature_cols=cols_with_friend)
    runner = ExperimentRunner(selector, model_a, model_b)
    results = runner.run(train, valid)
"""

from .feature_selector import IdentitySelector, SHAPSelector, PCAReducer, SHAPPCASelector
from .models.lgb_model import LGBModel
from .models.gnn_model import GNNModel
from .models.dysat_model import DySATModel
from .experiment import ExperimentRunner

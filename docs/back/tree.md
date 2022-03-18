# GBDT

## Introduction of GBDT
Gradient boosting decision tree is also the promising solutions for time series issues.

## Introduction of XGBoost
XGBoost is introduced in [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

## Introduction of LightGBM
LightGBM is introduces in [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)

## GBDT in time series
For multi-steps time series prediction, there could be 3 different methods.

- Recursive: if the data has several categorical feature and one quantitive feature, we can use recursive method. Only one model for multi-steps. When predict, the predicted target is transformed as feature.
- Fixed: No special restriction for fixed method. All multi-steps will different target, but it will generate n models, each model use the same feature, but predict different target.
- Extended: Similar requirement for data to recursive version. But it will generate n models for n-multi-steps prediction. So while train new model, the old data will be extended to add more data, and use recently data as feature.
- Hybrid: This is what I invented after thinking about the above three (maybe it has been invented many times), it's used when the data also has other quantitive features, not only the target. So the categorical and target related quantitive feature are the same as extended version, the other quantize feature are treated the same as fixed version.


## Performance
GBDT could also be tuned into SOTA model in time series. I read some implementations of the competition to let me so sure that I'm not so good at tuning the parameters.
So I believe the performance here could be further optimized.

## Further reading
- https://www.kaggle.com/shixw125/1st-place-lgb-model-public-0-506-private-0-511
- https://www.kaggle.com/pureheart/1st-place-lgb-model-public-0-470-private-0-502
- https://www.kaggle.com/plantsgo/solution-public-0-471-private-0-505
# AlphaEarth soil regression

工作流分两步：

```bash
../.venv/bin/python download_alphaearth_2018.py --project YOUR_PROJECT_ID
../.venv/bin/python train_alphaearth_soil_models.py
```

也可以使用环境变量：

```bash
EE_PROJECT=YOUR_PROJECT_ID ../.venv/bin/python download_alphaearth_2018.py
```

输出位置：

- `data/alphaearth_2018_sample_embeddings.csv`: 采样点处的 2018 AlphaEarth embedding 与土壤属性。
- `data/alphaearth_2018_preview_A00_A02.tif`: 研究区低分辨率预览 GeoTIFF。
- `outputs/model_metrics.csv`: 5 折交叉验证指标。
- `outputs/cross_validated_predictions.csv`: 交叉验证预测结果。
- `outputs/observed_vs_predicted.png`: 观测值与预测值对比图。
- `outputs/model_metrics.png`: 模型指标图。
- `outputs/band_importance.png`: embedding band 置换重要性图。

说明：松嫩平原整区 10 m、64 bands raster 体量过大；当前脚本下载采样点级 embedding，并另存 3 个 band 的低分辨率预览用于确认空间覆盖。

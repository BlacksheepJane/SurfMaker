surfdock/bash_scripts/test_scripts/eval_sample.sh是surfdock的运行脚本

eval_samples_new.sh是使用surfgen的surfmaker生成表面的运行脚本，即surfgen/data/surface_maker/surface_maker.py。

surfdock/model_weights/posepredict/best_model.pt和surfdock/model_weights/docking/best_ema_inference_epoch_model.pt过大，没有上传

## 路径问题

surfgen/data/surface_maker/surface_maker.py第20行和36-40行硬编码了路径。

surfdock/bash_scripts/test_scripts/eval_sample_new.sh第38行硬编码了timesplit363的路径。

'Pipeline stages in execution order. Each module has  run(cfg, paths)  and  is_done(cfg, paths)'
STAGES = ["prep", "blocking_train", "eval_blocking", "features_train", "train",
          "blocking_test", "features_test", "predict", "validate"]


def get_stage(name: str):
    import importlib
    mod = {"blocking_train": "blocking", "blocking_test": "blocking",
           "features_train": "features", "features_test": "features"}.get(name, name)
    return importlib.import_module(f"er.stages.{mod}")

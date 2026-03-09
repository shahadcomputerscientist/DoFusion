from solution import DocFusionSolution

solution = DocFusionSolution()

model_dir = solution.train(
    train_dir="data/findit2",
    work_dir="model"
)

solution.predict(
    model_dir=model_dir,
    data_dir="data/findit2",
    out_path="predictions.jsonl"
)
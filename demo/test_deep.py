from deepsparse import Pipeline

yolo_pipeline = Pipeline.create(
    task="yolo",
    model_path="zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none"
)

outputs = yolo_pipeline(images=["input.jpg"])
print(outputs)

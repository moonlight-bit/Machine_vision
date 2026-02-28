from ultralytics import YOLO

# Load a model

model = YOLO('yolov5nu.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='gear.yaml',workers=0, epochs=200, batch=16)
from ultralytics import YOLO
model = YOLO("C:/Users/23517/Desktop/fsdownload/ultralytics-8.4.13/best.pt")
results = model(source=0, show=True,save=False)

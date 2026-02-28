import cv2
import threading
import time
from ultralytics import YOLO
# 全局变量用于线程间共享帧
frame = None
frame_lock = threading.Lock()  # 添加锁来保护共享数据
running = True
model = YOLO("C:/Users/23517/Desktop/fsdownload/ultralytics-8.4.13/best.pt")

def show_frame(cap):
    """专门的显示线程函数"""
    global frame, running
    print("显示线程已启动")
    window_name = "Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while running:
        ret, new_frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
            
        # 使用锁安全地更新全局帧
        with frame_lock:
            frame = new_frame.copy()  # 使用copy()避免引用问题
        
        # 显示画面
        cv2.imshow(window_name, new_frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("检测到退出命令")
            running = False
            break
    
    cv2.destroyAllWindows()
    print("显示线程结束")

def process_frame_thread():
    """另一个线程处理帧的示例"""
    
    
    print("处理线程已启动")
    while running:
        current_frame = None
        # 安全地获取当前帧
        with frame_lock:
            if frame is not None:
                current_frame = frame.copy()
        
        if current_frame is not None:
            # 在这里处理帧，例如进行图像处理
           # height, width = current_frame.shape[:2]
            results = model(current_frame, show=False)
            result = results[0]
            boxes = result.boxes
            cls = boxes.cls.cpu().numpy().astype(int)
            print(cls)


            #print(f"处理帧: {width}x{height}, 时间: {time.time():.2f}")
        
        time.sleep(0.5)  # 每0.5秒处理一次
    
    print("处理线程结束")

def main():
    global running
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开")
    
    # 创建并启动显示线程
    display_thread = threading.Thread(target=show_frame, args=(cap,))
    display_thread.daemon = True
    display_thread.start()
    
    # 创建并启动处理线程
    process_thread = threading.Thread(target=process_frame_thread)
    process_thread.daemon = True
    process_thread.start()
    
    try:
        # 主线程等待
        while running:
            time.sleep(1)
            print("主线程运行中...", end="\r")
            
    except KeyboardInterrupt:
        print("\n用户中断")
        running = False
    
    # 清理
    cap.release()
    print("摄像头已释放")
    print("程序退出")

if __name__ == "__main__":
    main()
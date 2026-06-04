from ultralytics import YOLO
import os

# 1. 载入你刚刚训练好的微型模型
# ⚠️ 注意这里：使用了你终端里显示的绝对路径
model_path = r"C:\Users\ohxua\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

# 2. 设置文件夹路径
unlabeled_dir = "dataset/unlabeled_images"
images_dir = "dataset/images"
labels_dir = "dataset/labels"

print("🤖 AI 标注员已上线，开始处理未标注图片...")

# 3. 开始自动推理并生成标签
processed_count = 0
for filename in os.listdir(unlabeled_dir):
    if not filename.endswith(".jpg"):
        continue
        
    img_path = os.path.join(unlabeled_dir, filename)
    
    # 运行推理 (conf=0.25 设低一点，宁可多框几个，也不要漏框)
    results = model.predict(img_path, conf=0.25, verbose=False)
    
    # 准备写入 txt
    txt_filename = filename.replace(".jpg", ".txt")
    txt_path = os.path.join(labels_dir, txt_filename)
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        with open(txt_path, "w") as f:
            for box in boxes:
                # 获取类别 ID 和 归一化后的 xywh 坐标
                cls_id = int(box.cls[0].item())
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                
    # 标注完成后，将图片移回正式的 images 文件夹，以便统一管理
    os.rename(img_path, os.path.join(images_dir, filename))
    processed_count += 1

print(f"🎉 自动打标完成！共处理了 {processed_count} 张图片。")
print("请重新打开 labelImg 进行最后的质检 (按 D 键快速检查)！")
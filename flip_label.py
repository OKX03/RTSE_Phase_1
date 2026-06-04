import os

labels_dir = "dataset/labels"
file_count = 0
swap_count = 0

print("🔄 开始基因重组：对调 0 (danger) 和 1 (green)...")

for filename in os.listdir(labels_dir):
    # 忽略非 txt 文件和 classes.txt 字典文件
    if not filename.endswith(".txt") or filename == "classes.txt":
        continue
        
    filepath = os.path.join(labels_dir, filename)
    
    # 读取原始标签
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    modified = False
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        # 核心逻辑：0变1，1变0
        if parts[0] == '0':
            parts[0] = '1'
            modified = True
        elif parts[0] == '1':
            parts[0] = '0'
            modified = True
            
        new_lines.append(" ".join(parts) + "\n")
        
    # 如果有修改，就覆盖写入原文件
    if modified:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        swap_count += 1
    file_count += 1

print(f"✅ 任务完成！共扫描 {file_count} 个文件，成功翻转了 {swap_count} 个文件里的标签！")
print("👉 现在请重新打开 labelImg，按 D 键检查，世界应该恢复正常了！")
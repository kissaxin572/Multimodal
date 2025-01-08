import os
import pandas as pd
import numpy as np
import re
import datetime
import hashlib
from PIL import Image
import subprocess
import sys
from tqdm import tqdm

log_file_path = "Logs/log_30s.txt"  # 定义日志文件路径

def log_message(message):
    """记录日志信息到文件并打印到屏幕"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间并格式化
    log_entry = f"{timestamp} - {message}"  # 构造日志条目
    with open(log_file_path, 'a', encoding='utf-8') as log_file:  # 打开日志文件进行追加写入
        log_file.write(log_entry + "\n")  # 写入日志条目
    print(log_entry)  # 打印日志条目

class ImageCombiner:
    """二进制.img文件合并成一个文件"""
    
    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file
        
    def combine_images(self):
        """合并指定文件夹中的所有.img文件"""
        with open(self.output_file, 'wb') as outfile:  # 打开输出文件进行二进制写入
            for filename in os.listdir(self.input_folder):  # 遍历输入文件夹中的所有文件
                if filename.endswith('.img'):  # 如果文件以.img结尾
                    file_path = os.path.join(self.input_folder, filename)  # 获取文件的完整路径
                    with open(file_path, 'rb') as infile:  # 打开文件进行二进制读取
                        outfile.write(infile.read())  # 将文件内容写入输出文件
        
        # 显示合并后文件的大小
        file_size = os.path.getsize(self.output_file)  # 获取合并后文件的大小
        log_message(f'所有 .img 文件已合并到 {self.output_file} 中，文件大小为 {file_size} 字节。')  # 记录日志
        return self.output_file  # 返回合并后的文件路径

class MarkovVisualizer:
    """二进制文件的马尔可夫链可视化类"""
    
    def __init__(self):
        self.transition_matrix = np.zeros((256, 256), dtype=np.int32)
        self.byte_counts = np.zeros(256, dtype=np.int32)
        self.probability_matrix = np.zeros((256, 256), dtype=np.float32)
    
    def process_file(self, input_file):
        """处理输入文件并计算马尔可夫转移概率"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        byte_data = np.frombuffer(data, dtype=np.uint8)
        
        for i in range(len(byte_data) - 1):
            x, y = byte_data[i], byte_data[i+1]
            self.transition_matrix[x, y] += 1
            self.byte_counts[x] += 1
        
        p_max = 0.0
        for i in range(256):
            for j in range(256):
                if self.byte_counts[i] != 0:
                    self.probability_matrix[i, j] = self.transition_matrix[i, j] / self.byte_counts[i]
                    p_max = max(self.probability_matrix[i, j], p_max)
        
        self.image_matrix = (255 * (self.probability_matrix / p_max)).astype(np.uint8)
    
    def save_image(self, output_file):
        img = Image.fromarray(self.image_matrix, mode='L')
        img.save(output_file)
        log_message(f"Markov图像已保存为: {output_file}")

class ByteplotVisualizer:
    """二进制文件的Byteplot可视化类"""
    
    def __init__(self, width=256, final_size=(256, 256)):
        self.width = width
        self.final_size = final_size
        self.image_data = None
    
    def process_file(self, input_file):
        with open(input_file, 'rb') as f:
            data = f.read()
        
        byte_array = np.frombuffer(data, dtype=np.uint8)
        height = int(np.ceil(len(byte_array) / self.width))
        padded_length = self.width * height
        padded_data = np.pad(byte_array, (0, padded_length - len(byte_array)), mode='constant')
        self.image_data = padded_data.reshape((height, self.width))
    
    def save_image(self, output_file):
        if self.image_data is None:
            raise ValueError("请先调用process_file处理输入文件")
        
        img = Image.fromarray(self.image_data, mode='L')
        img_resized = img.resize(self.final_size, resample=Image.BILINEAR)
        img_resized.save(output_file)
        log_message(f"Byteplot图像已保存为: {output_file}")

class SimhashVisualizer:
    """二进制文件的Simhash可视化类"""
    
    def __init__(self, image_size=(256, 256), final_size=(256, 256)):
        self.image_size = image_size
        self.final_size = final_size
        self.Pixel = np.zeros(self.image_size, dtype=np.uint8)  # 初始化像素矩阵

    def _calculate_simhash(self, data):
        """使用新的Simhash算法计算坐标(x, y)"""
        # 对输入数据按128字节分块，不足时补0
        chunks = [data[i:i+128] for i in range(0, len(data), 128)]
        for chunk in chunks:
            if len(chunk) < 128:
                chunk = chunk.ljust(128, b'\0')  # 使用0填充至128字节
            
            # 分成16组，每组8字节
            groups = [chunk[i:i+8] for i in range(0, 128, 8)]
            T = np.zeros((16, 64), dtype=int)  # 初始化16个长度为64的向量
            
            for i, group in enumerate(groups):
                md5_hash = hashlib.md5(group).hexdigest()  # 计算MD5哈希
                hash_bits = bin(int(md5_hash[:16], 16))[2:].zfill(64)  # 取前64位
                for j, bit in enumerate(hash_bits):
                    T[i, j] = 1 if bit == '1' else -1
            
            # 计算全局Simhash向量
            V = np.sum(T, axis=0)
            V = np.where(V >= 0, 1, 0)  # 转换为0和1的向量
            
            # 计算x和y坐标
            x, y = self._vector_to_coordinates(V)
            self.Pixel[x, y] = min(255, self.Pixel[x, y] + 4)  # 更新像素值
    
    def _vector_to_coordinates(self, V):
        """根据Simhash向量计算坐标(x, y)"""
        x_bits, y_bits = V[:32], V[32:]  # 分成两部分
        
        def bits_to_decimal(bits):
            # 每4位生成一个16进制数，并根据值是否大于7生成8位二进制向量
            hex_values = [int("".join(map(str, bits[i*4:(i+1)*4])), 2) for i in range(8)]
            binary_vector = [1 if value > 7 else 0 for value in hex_values]
            return int("".join(map(str, binary_vector)), 2)
        
        x = bits_to_decimal(x_bits)
        y = bits_to_decimal(y_bits)
        return x, y
    
    def process_file(self, input_file):
        """处理输入文件并生成像素矩阵"""
        with open(input_file, 'rb') as f:
            data = f.read()
        self._calculate_simhash(data)  # 调用Simhash计算

    def save_image(self, output_file):
        """保存生成的Simhash图像"""
        if self.Pixel is None:
            raise ValueError("请先调用process_file处理输入文件")
        
        img = Image.fromarray(self.Pixel, mode='L')
        img_resized = img.resize(self.final_size, resample=Image.BILINEAR)
        img_resized.save(output_file)
        log_message(f"Simhash图像已保存为: {output_file}")

class SFCVisualizer:
    """使用空间填充曲线(Space-Filling Curve)进行二进制文件可视化"""
    
    def __init__(self, size=256, curve_type="hilbert", image_type="square", color_type="class"):
        self.size = size
        self.curve_type = curve_type
        self.image_type = image_type
        self.color_type = color_type
    
    def process_file(self, input_file, output_file):
        cmd = [
            sys.executable,  # Python解释器路径
            "SFC/binvis",  # binvis程序路径
            "-m", self.curve_type,  # 曲线类型
            "-t", self.image_type,  # 图像类型
            "-c", self.color_type,  # 颜色类型
            "-s", str(self.size),  # 图像大小
            input_file,  # 输入文件
            output_file  # 输出文件
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # 执行外部命令
            log_message(f"SFC图像已保存为: {output_file}")  # 记录日志
            return True  # 返回执行结果
        except subprocess.CalledProcessError as e:
            log_message(f"生成SFC图像时出错: {e.stderr}")  # 记录错误日志
            return False  # 返回执行结果
        except FileNotFoundError:
            log_message("错误: 未找到binvis程序，请确保SFC目录在Python路径中")  # 记录错误日志
            return False  # 返回执行结果

def parse_hpc_file(file_path):
    # 提取文件名中的信息
    filename = os.path.basename(file_path)  # 获取文件名
    x, y = filename.split('_')[:2]  # 从文件名中提取x和y的值
    
    # 读取文件内容
    data = []  # 存储解析后的数据
    current_time = None  # 当前时间初始化为None
    current_values = {}  # 当前值初始化为空字典
    
    with open(file_path, 'r') as f:  # 打开文件进行读取
        for line in f:  # 遍历文件中的每一行
            if line.startswith('#') or not line.strip():  # 跳过注释行和空行
                continue
                
            parts = line.strip().split()  # 将行内容按空格分割
            if len(parts) < 3:  # 如果分割后的部分少于3，跳过该行
                continue
                
            # 处理<not counted>的情况
            if '<not counted>' in line:  # 检查是否包含<not counted>
                time = float(parts[0])  # 获取时间
                value = 0  # 设置值为0
                event = parts[3]  # 获取事件名称
            else:
                time = float(parts[0])  # 获取时间
                value = parts[1]  # 获取值
                event = parts[2]  # 获取事件名称
            
            # 转换数值
            if value != 0:  # 如果值不为0
                try:
                    value = int(value.replace(',', ''))  # 尝试将值转换为整数
                except ValueError:  # 如果转换失败
                    value = 0  # 设置值为0
                
            if current_time != time:  # 如果当前时间与读取的时间不同
                if current_time is not None:  # 如果当前时间不为None
                    data.append([current_time, current_values.copy()])  # 将当前时间和对应值添加到数据中
                current_time = time  # 更新当前时间
                current_values = {}  # 重置当前值
            
            current_values[event] = value  # 更新当前事件的值
            
        if current_time is not None:  # 如果当前时间不为None
            data.append([current_time, current_values.copy()])  # 添加最后一组数据
            
    return x, y, data  # 返回提取的x, y值和数据

def process_hpc(input_dirs, output_file):
    """处理文件夹中的所有HPC数据文件并生成CSV"""
    # 8个HPC特征
    hpc_events = [
        "branch-instructions", "branch-misses", "cache-misses", "cpu-cycles",
        "instructions", "L1-dcache-loads", "LLC-stores", "iTLB-load-misses"
    ]
    
    all_data = {}  # 存储所有数据的字典
    
    # 处理每个文件
    for file in os.listdir(input_dirs):  # 遍历文件夹中的每个文件
        if file.endswith('.txt'):  # 只处理以.txt结尾的文件
            file_path = os.path.join(input_dirs, file)  # 获取文件的完整路径
            x, y, data = parse_hpc_file(file_path)  # 解析HPC文件
            id_key = f"{x}_{y}"  # 创建唯一的ID键
            
            if id_key not in all_data:  # 如果ID键不在all_data中
                all_data[id_key] = [{} for _ in range(100)]  # 初始化为100个时间点的空字典
                
            for time_idx, (time, values) in enumerate(data):  # 遍历解析后的数据
                if time_idx >= 100:  # 限制最大时间点
                    break
                all_data[id_key][time_idx].update(values)  # 更新对应时间点的值
    
    # 转换为DataFrame格式
    rows = []  # 存储行数据
    for id_key, time_series in all_data.items():  # 遍历所有数据
        x = id_key.split('_')[0]  # 提取x值
        for time_idx, values in enumerate(time_series):  # 遍历时间序列
            row = {
                'sample_id': id_key,  # ID键
                'timestamp_id': time_idx + 1,  # 保持从1开始
                'label': 1 if x == 'M' else 0  # 分类标记
            }
            
            for event in hpc_events:  # 遍历所有HPC事件
                row[event] = values.get(event, 0)  # 获取事件的值，如果不存在则填0
                
            rows.append(row)  # 将行数据添加到rows中
    
    # 创建DataFrame并排序
    df = pd.DataFrame(rows)  # 创建DataFrame
    columns = ['sample_id', 'timestamp_id', 'label'] + hpc_events  # 定义列顺序
    df = df[columns]  # 重新排列DataFrame的列顺序
    
    def extract_number(id_str):  # 提取ID中的数字
        match = re.search(r'(\d+)', id_str)  # 使用正则表达式提取数字
        return int(match.group(1)) if match else 0  # 返回提取的数字或0
    
    df['sort_key'] = df['sample_id'].apply(lambda x: (x.startswith('M'), extract_number(x)))  # 创建排序键
    df_sorted = df.sort_values(['sort_key', 'timestamp_id'])  # 按照排序键和时间戳排序
    df_sorted = df_sorted.drop('sort_key', axis=1)  # 删除排序键列
    
    # 保存CSV文件
    df_sorted.to_csv(output_file, index=False)  # 将DataFrame保存为CSV文件
    return df_sorted  # 返回排序后的DataFrame

def process_img(base_folder, output_base_folder):
    """处理良性和恶意软件数据集"""
    benign_base_folder = os.path.join(base_folder, 'B')
    malicious_base_folder = os.path.join(base_folder, 'M')

    # 创建输出文件夹
    output_folders = {
        "Byteplot": os.path.join(output_base_folder, "Byteplot"),
        "Markov": os.path.join(output_base_folder, "Markov"),
        "Simhash": os.path.join(output_base_folder, "Simhash"),
        "SFC_Gray": os.path.join(output_base_folder, "SFC", "Gray"),
        "SFC_Hilbert": os.path.join(output_base_folder, "SFC", "Hilbert"),
        "SFC_Zorder": os.path.join(output_base_folder, "SFC", "Zorder")
    }

    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    # 处理良性软件
    benign_files = []
    benign_folders = [b_folder for b_folder in os.listdir(benign_base_folder) if os.path.isdir(os.path.join(benign_base_folder, b_folder))]
    for b_folder in tqdm(benign_folders, desc="处理良性文件夹", unit="文件夹"):
        b_folder_path = os.path.join(benign_base_folder, b_folder)
        for ck_folder in os.listdir(b_folder_path):
            ck_folder_path = os.path.join(b_folder_path, ck_folder)
            if os.path.isdir(ck_folder_path):
                # 合并文件并记录
                combiner = ImageCombiner(input_folder=ck_folder_path, output_file=os.path.join(ck_folder_path, 'combined.img'))
                combined_file = combiner.combine_images()
                benign_files.append(combined_file)
                # benign_files.append(os.path.join(ck_folder_path, 'combined.img'))

    # 处理恶意软件
    malicious_files = []
    malicious_folders = [m_folder for m_folder in os.listdir(malicious_base_folder) if os.path.isdir(os.path.join(malicious_base_folder, m_folder))]
    for m_folder in tqdm(malicious_folders, desc="处理恶意文件夹", unit="文件夹"):
        m_folder_path = os.path.join(malicious_base_folder, m_folder)
        for ck_folder in os.listdir(m_folder_path):
            ck_folder_path = os.path.join(m_folder_path, ck_folder)
            if os.path.isdir(ck_folder_path):
                # 合并文件并记录
                combiner = ImageCombiner(input_folder=ck_folder_path, output_file=os.path.join(ck_folder_path, 'combined.img'))
                combined_file = combiner.combine_images()
                malicious_files.append(combined_file)
                # malicious_files.append(os.path.join(ck_folder_path, 'combined.img'))

    # 对每种方法进行处理，确保每种方法都处理完所有文件
    methods = [
        ("Byteplot", ByteplotVisualizer()),
        ("Markov", MarkovVisualizer()),
        ("Simhash", SimhashVisualizer()),
        ("SFC_Gray", SFCVisualizer(curve_type="gray")),
        ("SFC_Hilbert", SFCVisualizer(curve_type="hilbert")),
        ("SFC_Zorder", SFCVisualizer(curve_type="zorder"))
    ]

    # 处理每个方法
    for method_name, method_instance in methods:
        # 确保结果文件夹存在
        os.makedirs(os.path.join(output_folders[method_name], "B"), exist_ok=True)
        os.makedirs(os.path.join(output_folders[method_name], "M"), exist_ok=True)

        log_message(f"开始处理 {method_name} 方法...")

        # 先处理良性文件，使用 tqdm 显示进度条
        for benign_count, combined_file in enumerate(tqdm(benign_files, desc=f"处理良性文件 ({method_name})", unit="文件")):
            log_message(f"正在处理良性文件: {combined_file}")  # 增加提示
            # 从combined_file路径中提取编号   
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(combined_file)))  # 获取B_1这样的文件夹名
            number = folder_name.split('_')[1]  # 提取编号1，B-1->1.png，对齐数据
            # input_file="Datasets/Original/30s/Snapshots/B/B_1/ck1/combined.img",output_file="Datasets/Processed/30s/SFC/Gray/B/1.png"
            if method_name in ["SFC_Gray", "SFC_Hilbert", "SFC_Zorder"]:
                method_instance.process_file(input_file=combined_file, output_file=os.path.join(output_folders[method_name], "B", f"{number}.png"))
            else:
                method_instance.process_file(combined_file)
                method_instance.save_image(os.path.join(output_folders[method_name], "B", f"{number}.png"))

        # 再处理恶意文件，使用 tqdm 显示进度条
        for malicious_count, combined_file in enumerate(tqdm(malicious_files, desc=f"处理恶意文件 ({method_name})", unit="文件")):
            log_message(f"正在处理恶意文件: {combined_file}")  # 增加提示
            # 从combined_file路径中提取编号   
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(combined_file)))  # 获取M_1这样的文件夹名
            number = folder_name.split('_')[1]  # 提取编号1
            if method_name in ["SFC_Gray", "SFC_Hilbert", "SFC_Zorder"]:
                method_instance.process_file(input_file=combined_file, output_file=os.path.join(output_folders[method_name], "M", f"{number}.png"))
            else:
                method_instance.process_file(combined_file)
                method_instance.save_image(os.path.join(output_folders[method_name], "M", f"{number}.png"))

        log_message(f"{method_name} 方法处理完成！")

    # 删除所有combined.img文件
    for combined_file in tqdm(benign_files + malicious_files, desc="删除 combined.img 文件", unit="文件"):
        if os.path.exists(combined_file):
            os.remove(combined_file)

    log_message("所有combined.img文件已删除。")

def hpc():
    input_dirs = "Datasets/Original/30s/8Events"
    output_file = "Datasets/Processed/30s/30s_300ms.csv"
    
    # 创建输出目录(如果不存在)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("正在处理HPC数据...")  # 打印处理信息
    df_sorted = process_hpc(input_dirs, output_file)  # 处理文件夹中的数据
    print(f"已生成CSV文件: {output_file}")  # 打印生成的CSV文件路径

def b2image():
    input_base_folder = "Datasets/Original/30s/Snapshots"
    output_base_folder = "Datasets/Processed/30s"
    process_img(input_base_folder, output_base_folder)

if __name__ == "__main__":
    hpc()  # .txt hpc时序数据转换为csv
    b2image() # .img 二进制数据转换为sfc图像

import os
import sys
import subprocess
import re
import shutil
from tqdm import tqdm
import time
from collections import defaultdict

# 配置路径
GEOSTAB_DIR = "/home/corp/xingqiao.lin/code/GeoStab"
DATA_DIR = "/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train"
CSV_FILE = "/home/corp/xingqiao.lin/code/GeoStab/data/ddG/S8754.csv"

def check_file_exists(file_path, min_size=1):
    """检查文件是否存在且大小大于min_size字节"""
    return os.path.exists(file_path) and os.path.getsize(file_path) > min_size

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    try:
        print(f"🔄 {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} 完成")
            return True
        else:
            print(f"❌ {description} 失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ {description} 出错: {e}")
        return False

def load_names_from_csv(csv_file):
    """从CSV文件加载蛋白质名称"""
    names = []
    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # 跳过标题行，提取第一列（name列）
        for line in lines[1:]:
            if line.strip():
                name = line.split(',')[0].strip()
                names.append(name)
        
        return names
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return []

def extract_pdb_id(name):
    """从蛋白质名称中提取PDB ID"""
    # 格式: rcsb_1ABC_A_... -> 1ABC
    match = re.match(r'rcsb_([A-Z0-9]+)_', name)
    return match.group(1) if match else None

def group_by_pdb_id(names):
    """按PDB ID分组蛋白质名称"""
    pdb_groups = defaultdict(list)
    for name in names:
        pdb_id = extract_pdb_id(name)
        if pdb_id:
            pdb_groups[pdb_id].append(name)
    return pdb_groups

def find_existing_esm2_file(pdb_id, pdb_groups):
    """查找已存在的esm2.pt文件"""
    for name in pdb_groups[pdb_id]:
        clean_name = name.replace(' ', '_')
        wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
        esm_file = f'{wt_folder}/esm2.pt'
        if check_file_exists(esm_file):
            return esm_file
    return None

def copy_esm2_file(source_file, target_file):
    """复制esm2.pt文件"""
    try:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        shutil.copy2(source_file, target_file)
        return True
    except Exception as e:
        print(f"❌ 复制esm2.pt失败: {e}")
        return False

def process_wt_pdb_group(pdb_id, names):
    """处理同一PDB ID的所有WT蛋白质（可以复制）"""
    print(f"\n🔍 处理WT PDB ID: {pdb_id} ({len(names)} 个蛋白质)")
    
    # 查找已存在的esm2.pt文件
    existing_file = find_existing_esm2_file(pdb_id, {pdb_id: names})
    
    if existing_file:
        print(f"✅ 找到已存在的esm2.pt: {existing_file}")
        # 复制到所有需要的目录
        copy_count = 0
        for name in names:
            clean_name = name.replace(' ', '_')
            wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
            target_file = f'{wt_folder}/esm2.pt'
            
            if not check_file_exists(target_file):
                if copy_esm2_file(existing_file, target_file):
                    copy_count += 1
                    print(f"📋 复制到: {name}")
        
        print(f"📊 PDB {pdb_id}: 复制了 {copy_count} 个文件")
        return True, copy_count, 0
    else:
        # 需要生成新的esm2.pt文件
        print(f"🔄 需要生成新的esm2.pt文件")
        
        # 选择第一个蛋白质作为代表来生成
        first_name = names[0]
        clean_name = first_name.replace(' ', '_')
        wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
        fasta_file = f'{wt_folder}/result.fasta'
        
        if not check_file_exists(fasta_file):
            print(f"❌ 找不到FASTA文件: {fasta_file}")
            return False, 0, 0
        
        # 生成esm2.pt
        cmd = f"python {GEOSTAB_DIR}/generate_features/esm2_embedding.py --fasta_file {fasta_file} --saved_folder {wt_folder}"
        if not run_command(cmd, f"生成 {pdb_id} 的esm2.pt"):
            return False, 0, 0
        
        # 复制到同PDB ID的其他目录
        source_file = f'{wt_folder}/esm2.pt'
        copy_count = 0
        for name in names[1:]:  # 跳过第一个，已经生成了
            clean_name = name.replace(' ', '_')
            wt_folder_target = f'{DATA_DIR}/{clean_name}/wt_data'
            target_file = f'{wt_folder_target}/esm2.pt'
            
            if copy_esm2_file(source_file, target_file):
                copy_count += 1
                print(f"📋 复制到: {name}")
        
        print(f"📊 PDB {pdb_id}: 生成了1个文件，复制了 {copy_count} 个文件")
        return True, copy_count + 1, 0

def process_mut_individual(name):
    """处理单个mut蛋白质（必须单独运行）"""
    clean_name = name.replace(' ', '_')
    mut_folder = f'{DATA_DIR}/{clean_name}/mut_data'
    fasta_file = f'{mut_folder}/result.fasta'
    esm_file = f'{mut_folder}/esm2.pt'
    
    # 检查是否已存在
    if check_file_exists(esm_file):
        print(f"⏭️ {name}: esm2.pt 已存在，跳过")
        return True, 0
    
    # 检查FASTA文件是否存在
    if not check_file_exists(fasta_file):
        print(f"❌ 找不到FASTA文件: {fasta_file}")
        return False, 0
    
    # 生成esm2.pt
    cmd = f"python {GEOSTAB_DIR}/generate_features/esm2_embedding.py --fasta_file {fasta_file} --saved_folder {mut_folder}"
    if not run_command(cmd, f"生成 {name} 的mut esm2.pt"):
        return False, 0
    
    print(f"✅ 生成 {name}: {esm_file}")
    return True, 1

def main():
    """主函数"""
    print("🚀 启动GeoStab ESM2特征生成脚本...")
    print("📊 处理逻辑：WT按PDB分组复制，mut每个单独运行")
    print("=" * 80)
    
    # 加载数据
    print("📊 加载数据...")
    names = load_names_from_csv(CSV_FILE)
    print(f"✅ 加载了 {len(names)} 个蛋白质")
    
    # 按PDB ID分组
    print("🔍 按PDB ID分组...")
    pdb_groups = group_by_pdb_id(names)
    print(f"✅ 分为 {len(pdb_groups)} 个PDB组")
    
    # 统计变量
    wt_success_count = 0
    wt_error_count = 0
    wt_generated = 0
    wt_copied = 0
    
    mut_success_count = 0
    mut_error_count = 0
    mut_generated = 0
    
    # 处理WT特征（按PDB分组）
    print("\n🔍 开始处理WT特征...")
    print("=" * 80)
    
    for pdb_id, pdb_names in tqdm(pdb_groups.items(), desc="处理WT PDB组"):
        try:
            success, generated, copied = process_wt_pdb_group(pdb_id, pdb_names)
            if success:
                wt_success_count += len(pdb_names)
                wt_generated += generated
                wt_copied += copied
            else:
                wt_error_count += len(pdb_names)
        except Exception as e:
            wt_error_count += len(pdb_names)
            print(f"❌ PDB {pdb_id} 处理出错: {e}")
    
    # 处理mut特征（每个单独运行）
    print("\n🔍 开始处理mut特征...")
    print("=" * 80)
    
    for name in tqdm(names, desc="处理mut特征"):
        try:
            success, generated = process_mut_individual(name)
            if success:
                mut_success_count += 1
                mut_generated += generated
            else:
                mut_error_count += 1
        except Exception as e:
            mut_error_count += 1
            print(f"❌ {name} 处理出错: {e}")
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📊 处理完成统计:")
    print("=" * 80)
    print("WT特征处理:")
    print(f"  ✅ 成功处理: {wt_success_count}")
    print(f"  ❌ 处理失败: {wt_error_count}")
    print(f"  🔄 新生成esm2.pt: {wt_generated}")
    print(f"  📋 复制esm2.pt: {wt_copied}")
    print()
    print("mut特征处理:")
    print(f"  ✅ 成功处理: {mut_success_count}")
    print(f"  ❌ 处理失败: {mut_error_count}")
    print(f"  🔄 新生成esm2.pt: {mut_generated}")
    print()
    print(f"📊 总计: {len(names)} 个蛋白质")
    
    # 检查生成的文件
    print("\n🔍 检查生成的文件...")
    wt_esm_count = 0
    mut_esm_count = 0
    
    for name in names:
        clean_name = name.replace(' ', '_')
        wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
        mut_folder = f'{DATA_DIR}/{clean_name}/mut_data'
        
        if check_file_exists(f'{wt_folder}/esm2.pt'):
            wt_esm_count += 1
        
        if check_file_exists(f'{mut_folder}/esm2.pt'):
            mut_esm_count += 1
    
    print(f"📊 WT esm2.pt 文件数: {wt_esm_count}")
    print(f"📊 mut esm2.pt 文件数: {mut_esm_count}")
    
    print("\n🎉 ESM2特征生成完成！")

if __name__ == "__main__":
    main()

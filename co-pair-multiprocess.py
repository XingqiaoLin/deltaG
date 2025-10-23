#!/usr/bin/env python3
"""
多进程版本的GeoStab 3D特征生成脚本
支持并行处理WT和mut特征生成，使用进程池提高性能
"""

import os
import sys
import subprocess
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from typing import Tuple, Dict, Any
import signal

# 配置路径
GEOSTAB_DIR = "/home/corp/xingqiao.lin/code/GeoStab"
DATA_DIR = "/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train"
CSV_FILE = "/home/corp/xingqiao.lin/code/GeoStab/data/ddG/S8754.csv"

def check_file_exists(file_path, min_size=1):
    """检查文件是否存在且大小大于min_size字节"""
    return os.path.exists(file_path) and os.path.getsize(file_path) > min_size

def validate_pdb_file(pdb_file):
    """验证PDB文件是否可以被coordinate.py正确处理"""
    try:
        # 尝试运行coordinate.py来测试PDB文件
        test_cmd = f"python {GEOSTAB_DIR}/generate_features/coordinate.py --pdb_file {pdb_file} --saved_folder /tmp"
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return True, "PDB文件格式正确"
        else:
            return False, f"PDB文件处理失败: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "PDB文件验证超时"
    except Exception as e:
        return False, f"PDB文件验证出错: {e}"

def run_command(cmd, description="", process_id=None):
    """运行命令并处理错误"""
    try:
        process_prefix = f"[P{process_id}] " if process_id is not None else ""
        print(f"🔄 {process_prefix}{description}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {process_prefix}{description} 完成")
            # 如果命令成功但stderr有内容，也打印出来用于调试
            if result.stderr.strip():
                print(f"⚠️ {process_prefix}警告信息: {result.stderr.strip()}")
            return True
        else:
            print(f"❌ {process_prefix}{description} 失败: {result.stderr}")
            # 也打印stdout用于调试
            if result.stdout.strip():
                print(f"📄 {process_prefix}输出信息: {result.stdout.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {process_prefix}{description} 超时")
        return False
    except Exception as e:
        print(f"❌ {process_prefix}{description} 出错: {e}")
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

def process_wt_features(args):
    """处理单个蛋白质的WT特征（用于多进程）"""
    name, clean_name, process_id = args
    wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
    process_prefix = f"[P{process_id}] " if process_id is not None else ""
    
    # 检查必要文件
    fasta_file = f'{wt_folder}/result.fasta'
    pdb_file = f'{wt_folder}/relaxed.pdb'
    
    if not check_file_exists(fasta_file):
        print(f"⚠️ {process_prefix}跳过 {name}: FASTA文件不存在")
        return name, 'wt', False, "FASTA文件不存在"
    
    if not check_file_exists(pdb_file):
        print(f"⚠️ {process_prefix}跳过 {name}: PDB文件不存在")
        return name, 'wt', False, "PDB文件不存在"
    
    # 验证PDB文件是否可以被正确处理
    is_valid, error_msg = validate_pdb_file(pdb_file)
    if not is_valid:
        print(f"⚠️ {process_prefix}跳过 {name}: PDB文件验证失败 - {error_msg}")
        return name, 'wt', False, f"PDB文件验证失败: {error_msg}"
    
    success = True
    error_msg = ""
    
    # 1. 生成coordinate.pt
    coord_file = f'{wt_folder}/coordinate.pt'
    coord_generated = False
    if not check_file_exists(coord_file):
        cmd = f"python {GEOSTAB_DIR}/generate_features/coordinate.py --pdb_file {pdb_file} --saved_folder {wt_folder}"
        if run_command(cmd, f"生成 {name} 的coordinate.pt", process_id):
            # 验证文件是否真正生成成功
            if check_file_exists(coord_file):
                coord_generated = True
                print(f"✅ {process_prefix}{name} coordinate.pt 生成成功")
            else:
                print(f"❌ {process_prefix}{name} coordinate.pt 生成失败：文件未创建")
                success = False
                error_msg = "coordinate.pt文件未创建"
        else:
            success = False
            error_msg = "coordinate.pt生成命令失败"
    else:
        print(f"⏭️ {process_prefix}{name} coordinate.pt 已存在，跳过")
        coord_generated = True
    
    # 2. 生成pair特征
    pair_file = f'{wt_folder}/pair.pt'
    if not check_file_exists(pair_file):
        # 检查coordinate.pt是否存在，因为pair.py需要它
        if coord_generated or check_file_exists(coord_file):
            cmd = f"python {GEOSTAB_DIR}/generate_features/pair.py --coordinate_file {coord_file} --saved_folder {wt_folder}"
            if not run_command(cmd, f"生成 {name} 的pair特征", process_id):
                success = False
                error_msg = "pair特征生成命令失败"
        else:
            print(f"⚠️ {process_prefix}{name}: coordinate.pt不存在，无法生成pair特征")
            success = False
            error_msg = "coordinate.pt不存在"
    else:
        print(f"⏭️ {process_prefix}{name} pair.pt 已存在，跳过")
    
    if success:
        return name, 'wt', True, "成功"
    else:
        return name, 'wt', False, error_msg

def process_mut_features(args):
    """处理单个蛋白质的mut特征（用于多进程）"""
    name, clean_name, process_id = args
    mut_folder = f'{DATA_DIR}/{clean_name}/mut_data'
    process_prefix = f"[P{process_id}] " if process_id is not None else ""
    
    # 检查必要文件
    fasta_file = f'{mut_folder}/result.fasta'
    pdb_file = f'{mut_folder}/relaxed_repair.pdb'
    
    if not check_file_exists(fasta_file):
        print(f"⚠️ {process_prefix}跳过 {name}: mut FASTA文件不存在")
        return name, 'mut', False, "mut FASTA文件不存在"
    
    if not check_file_exists(pdb_file):
        print(f"⚠️ {process_prefix}跳过 {name}: mut PDB文件不存在")
        return name, 'mut', False, "mut PDB文件不存在"
    
    # 验证PDB文件是否可以被正确处理
    is_valid, error_msg = validate_pdb_file(pdb_file)
    if not is_valid:
        print(f"⚠️ {process_prefix}跳过 {name}: mut PDB文件验证失败 - {error_msg}")
        return name, 'mut', False, f"mut PDB文件验证失败: {error_msg}"
    
    success = True
    error_msg = ""
    
    # 1. 生成coordinate.pt
    coord_file = f'{mut_folder}/coordinate.pt'
    coord_generated = False
    if not check_file_exists(coord_file):
        cmd = f"python {GEOSTAB_DIR}/generate_features/coordinate.py --pdb_file {pdb_file} --saved_folder {mut_folder}"
        if run_command(cmd, f"生成 {name} 的mut coordinate.pt", process_id):
            # 验证文件是否真正生成成功
            if check_file_exists(coord_file):
                coord_generated = True
                print(f"✅ {process_prefix}{name} mut coordinate.pt 生成成功")
            else:
                print(f"❌ {process_prefix}{name} mut coordinate.pt 生成失败：文件未创建")
                success = False
                error_msg = "mut coordinate.pt文件未创建"
        else:
            success = False
            error_msg = "mut coordinate.pt生成命令失败"
    else:
        print(f"⏭️ {process_prefix}{name} mut coordinate.pt 已存在，跳过")
        coord_generated = True
    
    # 2. 生成pair特征
    pair_file = f'{mut_folder}/pair.pt'
    if not check_file_exists(pair_file):
        # 检查coordinate.pt是否存在，因为pair.py需要它
        if coord_generated or check_file_exists(coord_file):
            cmd = f"python {GEOSTAB_DIR}/generate_features/pair.py --coordinate_file {coord_file} --saved_folder {mut_folder}"
            if not run_command(cmd, f"生成 {name} 的mut pair特征", process_id):
                success = False
                error_msg = "mut pair特征生成命令失败"
        else:
            print(f"⚠️ {process_prefix}{name}: mut coordinate.pt不存在，无法生成pair特征")
            success = False
            error_msg = "mut coordinate.pt不存在"
    else:
        print(f"⏭️ {process_prefix}{name} mut pair.pt 已存在，跳过")
    
    if success:
        return name, 'mut', True, "成功"
    else:
        return name, 'mut', False, error_msg

def process_with_processes(names, process_type, max_workers=4):
    """使用多进程处理蛋白质列表"""
    print(f"\n🔍 开始处理{process_type.upper()}特征...")
    print("=" * 80)
    
    # 准备任务列表
    tasks = []
    for i, name in enumerate(names):
        clean_name = name.replace(' ', '_')
        process_id = i % max_workers + 1
        tasks.append((name, clean_name, process_id))
    
    # 统计变量
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 选择处理函数
    process_func = process_wt_features if process_type == 'wt' else process_mut_features
    
    # 使用ProcessPoolExecutor进行多进程处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_func, task): task for task in tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc=f"生成{process_type.upper()}特征") as pbar:
            for future in as_completed(future_to_task):
                try:
                    name, proc_type, success, message = future.result()
                    
                    if success:
                        success_count += 1
                        print(f"✅ {name} {proc_type}特征生成完成")
                    else:
                        if "出错" in message or "失败" in message:
                            error_count += 1
                            print(f"❌ {name} {proc_type}处理出错: {message}")
                        else:
                            skip_count += 1
                            print(f"⏭️ {name} {proc_type}跳过: {message}")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    error_count += 1
                    print(f"❌ 任务执行出错: {e}")
                    pbar.update(1)
    
    return success_count, skip_count, error_count

def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n⚠️ 收到中断信号 {signum}，正在安全退出...")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='多进程GeoStab 3D特征生成脚本')
    parser.add_argument('--max_workers', type=int, default=None, 
                       help='最大进程数 (默认: CPU核心数)')
    parser.add_argument('--process_wt', action='store_true', help='处理WT特征')
    parser.add_argument('--process_mut', action='store_true', help='处理mut特征')
    parser.add_argument('--process_all', action='store_true', help='处理所有特征 (WT和mut)')
    parser.add_argument('--batch_size', type=int, default=100, 
                       help='批处理大小 (默认: 100)')
    
    args = parser.parse_args()
    
    # 设置默认进程数
    if args.max_workers is None:
        args.max_workers = min(mp.cpu_count(), 8)  # 最多8个进程
    
    # 如果没有指定处理类型，默认处理所有
    if not any([args.process_wt, args.process_mut, args.process_all]):
        args.process_all = True
    
    print("🚀 启动多进程GeoStab 3D特征生成脚本...")
    print(f"🧵 最大进程数: {args.max_workers}")
    print(f"💻 CPU核心数: {mp.cpu_count()}")
    print("📊 生成coordinate.pt和pair特征（WT和mut）")
    print("=" * 80)
    
    # 加载数据
    print("📊 加载数据...")
    names = load_names_from_csv(CSV_FILE)
    print(f"✅ 加载了 {len(names)} 个蛋白质")
    
    # 统计变量
    wt_success_count = 0
    wt_skip_count = 0
    wt_error_count = 0
    
    mut_success_count = 0
    mut_skip_count = 0
    mut_error_count = 0
    
    start_time = time.time()
    
    # 处理WT特征
    if args.process_wt or args.process_all:
        wt_success_count, wt_skip_count, wt_error_count = process_with_processes(
            names, 'wt', args.max_workers
        )
    
    # 处理mut特征
    if args.process_mut or args.process_all:
        mut_success_count, mut_skip_count, mut_error_count = process_with_processes(
            names, 'mut', args.max_workers
        )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📊 处理完成统计:")
    print("=" * 80)
    print("WT特征处理:")
    print(f"  ✅ 成功处理: {wt_success_count}")
    print(f"  ⏭️ 跳过: {wt_skip_count}")
    print(f"  ❌ 处理失败: {wt_error_count}")
    print()
    print("mut特征处理:")
    print(f"  ✅ 成功处理: {mut_success_count}")
    print(f"  ⏭️ 跳过: {mut_skip_count}")
    print(f"  ❌ 处理失败: {mut_error_count}")
    print()
    print(f"📊 总计: {len(names)} 个蛋白质")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒")
    print(f"⚡ 平均速度: {len(names) * 2 / total_time:.2f} 蛋白质/秒")
    
    # 检查生成的文件
    print("\n🔍 检查生成的文件...")
    wt_coord_count = 0
    wt_pair_count = 0
    mut_coord_count = 0
    mut_pair_count = 0
    
    for name in names:
        clean_name = name.replace(' ', '_')
        wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
        mut_folder = f'{DATA_DIR}/{clean_name}/mut_data'
        
        if check_file_exists(f'{wt_folder}/coordinate.pt'):
            wt_coord_count += 1
        if check_file_exists(f'{wt_folder}/pair.pt'):
            wt_pair_count += 1
        if check_file_exists(f'{mut_folder}/coordinate.pt'):
            mut_coord_count += 1
        if check_file_exists(f'{mut_folder}/pair.pt'):
            mut_pair_count += 1
    
    print(f"📊 WT coordinate.pt 文件数: {wt_coord_count}")
    print(f"📊 WT pair.pt 文件数: {wt_pair_count}")
    print(f"📊 mut coordinate.pt 文件数: {mut_coord_count}")
    print(f"📊 mut pair.pt 文件数: {mut_pair_count}")
    
    print("\n🎉 多进程3D特征生成完成！")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()





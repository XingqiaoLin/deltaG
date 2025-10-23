# %%
import pandas as pd
train=pd.read_csv("/home/corp/xingqiao.lin/code/GeoStab/data/ddG/S8754.csv", sep=',')
train

# %%
import sys
import os
if not os.path.exists('/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train'):
    os.mkdir('/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train')
import pandas as pd
train=pd.read_csv("/home/corp/xingqiao.lin/code/GeoStab/data/ddG/S8754.csv", sep=',')

names=train['name'].tolist()
for name in names:
    name=name.replace(' ', '_')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/wt_data'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/wt_data')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/mut_data'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/mut_data')
        


# %%
for name in names:
        # 获取对应的行数据
        row = train[train['name'] == name].iloc[0]
        wt_seq = row['wt_seq']
        mut_seq = row['mut_seq']
        
        # 创建WT FASTA文件
        wt_fasta_path = f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/wt_data/result.fasta'
        with open(wt_fasta_path, 'w') as f:
            f.write(f">result\n")
            f.write(f"{wt_seq}\n")
        
        # 创建MUT FASTA文件
        mut_fasta_path = f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/mut_data/result.fasta'
        with open(mut_fasta_path, 'w') as f:
            f.write(f">{name}_MUT\n")
            f.write(f"{mut_seq}\n")
        
        print(f"✅ 创建 {name} 的FASTA文件")
        

# %%
import subprocess
from tqdm import tqdm

for name in tqdm(names, desc="生成特征"):
    clean_name = name.replace(' ', '_')
    wt_fasta_path = f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{clean_name}/wt_data'
    
    # 生成fixed_embedding
    if not os.path.exists(f'{wt_fasta_path}/fixed_embedding.pt'):
        cmd = f"python /home/corp/xingqiao.lin/code/GeoStab/generate_features/fixed_embedding.py --fasta_file {wt_fasta_path}/result.fasta --saved_folder {wt_fasta_path}"
        subprocess.run(cmd, shell=True)
    
    print(f"✅ {name} 特征生成完成")




# %%
import os, torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForMaskedLM
from tqdm import tqdm
import re
import shutil

HF_MODEL_DIR = "/home/corp/xingqiao.lin/.cache/huggingface/hub/facebook/esm1v_t33_650M_UR90S_1"

# 使用PyTorch格式加载模型（不使用safetensors）
tok = AutoTokenizer.from_pretrained(HF_MODEL_DIR, use_fast=False, local_files_only=True)
model = EsmForMaskedLM.from_pretrained(
    HF_MODEL_DIR, 
    local_files_only=True,
    use_safetensors=False  # 明确指定不使用safetensors
).eval().to("cpu")

def extract_pdb_id(clean_name):
    """从clean_name中提取PDB ID"""
    # 匹配 rcsb_1ABC_A_... 格式
    match = re.match(r'rcsb_([A-Z0-9]+)_', clean_name)
    if match:
        return match.group(1)
    return None

def find_same_pdb_variants(pdb_id, all_names):
    """找到所有相同PDB ID的变体"""
    variants = []
    for name in all_names:
        clean_name = name.replace(' ', '_')
        if clean_name.startswith(f'rcsb_{pdb_id}_'):
            variants.append(clean_name)
    return variants

# 统计信息
processed_pdb_ids = set()
skipped_count = 0
copied_count = 0
generated_count = 0

for name in tqdm(names, desc="生成特征"):
    clean_name = name.replace(' ', '_')
    wt_fasta_path = f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{clean_name}/wt_data'
    seq_path = f'{wt_fasta_path}/result.fasta'
    
    # 检查文件是否存在
    if not os.path.exists(seq_path):
        print(f"⚠️ 跳过 {name}: FASTA文件不存在")
        continue
    
    # 提取PDB ID
    pdb_id = extract_pdb_id(clean_name)
    if not pdb_id:
        print(f"⚠️ 跳过 {name}: 无法提取PDB ID")
        continue
    
    out_path = os.path.join(wt_fasta_path, "esm1v-1.pt")
    
    # 检查是否已经生成
    if os.path.exists(out_path):
        print(f"⏭️ {name}: esm1v-1.pt 已存在，跳过")
        continue
    
    # 检查是否已经处理过这个PDB ID
    if pdb_id in processed_pdb_ids:
        print(f"⏭️ {name}: PDB ID {pdb_id} 已处理过，跳过")
        continue
    
    try:
        # 生成特征
        seq = str(next(SeqIO.parse(seq_path, "fasta")).seq)
        inputs = tok(seq, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            reps = out.hidden_states[-1][0, 1:-1, :].cpu().clone()
        
        # 保存到当前目录
        torch.save(reps, out_path)
        print(f"✅ 生成 {name}: {out_path}, shape={tuple(reps.shape)}")
        generated_count += 1
        
        # 标记这个PDB ID已处理
        processed_pdb_ids.add(pdb_id)
        
        # 找到所有相同PDB ID的变体
        same_pdb_variants = find_same_pdb_variants(pdb_id, names)
        
        # 复制到所有相同PDB ID的目录
        for variant in same_pdb_variants:
            if variant == clean_name:
                continue  # 跳过自己
            
            variant_wt_path = f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{variant}/wt_data'
            variant_out_path = os.path.join(variant_wt_path, "esm1v-1.pt")
            
            # 检查目标目录是否存在
            if not os.path.exists(variant_wt_path):
                print(f"⚠️ 跳过 {variant}: 目标目录不存在")
                continue
            
            # 检查目标文件是否已存在
            if os.path.exists(variant_out_path):
                print(f"⏭️ {variant}: esm1v-1.pt 已存在，跳过复制")
                continue
            
            try:
                # 复制特征文件
                shutil.copy2(out_path, variant_out_path)
                print(f"📋 复制 {pdb_id} 特征到 {variant}")
                copied_count += 1
            except Exception as e:
                print(f"❌ 复制到 {variant} 失败: {e}")
        
        print(f"🎯 PDB {pdb_id}: 生成了1个，复制了{len(same_pdb_variants)-1}个变体")
        
    except Exception as e:
        print(f"❌ {name}: 处理失败 - {e}")

# 输出统计信息
print(f"\n📊 处理完成统计:")
print(f"✅ 生成特征: {generated_count}")
print(f"📋 复制特征: {copied_count}")
print(f"⏭️ 跳过: {skipped_count}")
print(f"🎯 处理的PDB ID数: {len(processed_pdb_ids)}")


#生成ESM-1V-1特征
import sys
import os
if not os.path.exists('/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train'):
    os.mkdir('/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train')
import pandas as pd
train=pd.read_csv("/home/corp/xingqiao.lin/code/GeoStab/data/ddG/S8754.csv", sep=',')

names=train['name'].tolist()
for name in names:
    name=name.replace(' ', '_')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/wt_data'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/wt_data')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/mut_data'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/mut_data')
import subprocess
for name in names:  # 只处理前5个
    clean_name = name.replace(' ', '_')
    wt_fasta_path = f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{clean_name}/wt_data'
    
    # 生成fixed_embedding
    if not os.path.exists(f'{wt_fasta_path}/esm1v-1.pt'):
        cmd = f"python /home/corp/xingqiao.lin/code/GeoStab/generate_features/esm1v_logits.py --model_index 1 --fasta_file {wt_fasta_path}/result.fasta --saved_folder {wt_fasta_path}"
        subprocess.run(cmd, shell=True)
    
    print(f"✅ {name} 特征生成完成")

# %%
#!/usr/bin/env bash

# 🚀 PDB下载脚本 - Python版本
import os
import re
import requests
from tqdm import tqdm
import time

def find_pdb_ids(data_dir):
    """从目录结构中提取PDB ID"""
    pdb_ids = set()
    
    # 查找所有rcsb_*目录
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name.startswith('rcsb_'):
                # 提取PDB ID (rcsb_1ABC_A_... -> 1ABC)
                match = re.match(r'rcsb_([A-Z0-9]+)_', dir_name)
                if match:
                    pdb_id = match.group(1)
                    pdb_ids.add(pdb_id)
    
    return sorted(list(pdb_ids))
#下载PDB文件
def download_pdb(pdb_id, output_dir, timeout=30):
    """下载单个PDB文件"""
    pdb_id_lower = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"
    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        return True, f"✅ {pdb_id} 下载成功"
    
    except requests.exceptions.RequestException as e:
        return False, f"❌ {pdb_id} 下载失败: {e}"
    except Exception as e:
        return False, f"❌ {pdb_id} 下载出错: {e}"

# 配置
data_dir = "/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train"
output_dir = "/home/corp/xingqiao.lin/code/GeoStab/pdbs"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print("🔍 正在扫描目录结构...")
pdb_ids = find_pdb_ids(data_dir)

print(f"📊 找到 {len(pdb_ids)} 个唯一的PDB ID")
print(f"📁 输出目录: {output_dir}")
print("=" * 60)

# 统计变量
success_count = 0
failed_count = 0
skipped_count = 0

# 下载PDB文件
for pdb_id in tqdm(pdb_ids, desc="下载PDB文件"):
    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    # 检查文件是否已存在
    if os.path.exists(output_file):
        skipped_count += 1
        tqdm.write(f"⏭️  {pdb_id} 已存在，跳过")
        continue
    
    # 下载文件
    success, message = download_pdb(pdb_id, output_dir)
    
    if success:
        success_count += 1
        tqdm.write(message)
    else:
        failed_count += 1
        tqdm.write(message)
    
    # 添加小延迟避免请求过于频繁
    time.sleep(0.1)

# 输出统计结果
print("\n" + "=" * 60)
print("📊 下载完成统计:")
print("=" * 60)
print(f"✅ 成功下载: {success_count}")
print(f"⏭️  跳过 (已存在): {skipped_count}")
print(f"❌ 下载失败: {failed_count}")
print(f"📊 总计处理: {len(pdb_ids)}")

# 检查下载的文件
downloaded_files = [f for f in os.listdir(output_dir) if f.endswith('.pdb')]
print(f"📁 输出目录中的PDB文件数: {len(downloaded_files)}")

if failed_count > 0:
    print("\n⚠️  部分文件下载失败，请检查网络连接或PDB ID是否有效")

print("\n🎉 PDB下载任务完成！")


# %%
import os
import re
import shutil
from tqdm import tqdm
from collections import defaultdict

def find_pdb_directories(data_dir):
    """找到所有rcsb目录并提取PDB ID"""
    pdb_to_dirs = defaultdict(list)
    
    # 查找所有rcsb_*目录
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name.startswith('rcsb_'):
                # 提取PDB ID (rcsb_1ABC_A_... -> 1ABC)
                match = re.match(r'rcsb_([A-Z0-9]+)_', dir_name)
                if match:
                    pdb_id = match.group(1)
                    full_path = os.path.join(root, dir_name, 'wt_data')
                    if os.path.exists(full_path):
                        pdb_to_dirs[pdb_id].append(full_path)
    
    return pdb_to_dirs

def distribute_pdb_files(pdb_to_dirs, source_dir):
    """将PDB文件分发到所有相关目录"""
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    print(f"🔍 开始分发PDB文件...")
    print(f"📁 源目录: {source_dir}")
    print("=" * 80)
    
    for pdb_id, target_dirs in tqdm(pdb_to_dirs.items(), desc="分发PDB文件"):
        source_file = os.path.join(source_dir, f"{pdb_id}.pdb")
        
        # 检查源文件是否存在
        if not os.path.exists(source_file):
            tqdm.write(f"❌ {pdb_id}.pdb 源文件不存在，跳过")
            error_count += 1
            continue
        
        # 复制到所有相关目录
        for target_dir in target_dirs:
            target_file = os.path.join(target_dir, f"{pdb_id}.pdb")
            
            try:
                # 检查目标文件是否已存在
                if os.path.exists(target_file):
                    skipped_count += 1
                    tqdm.write(f"⏭️  {pdb_id}.pdb 已存在于 {target_dir}")
                    continue
                
                # 复制文件
                shutil.copy2(source_file, target_file)
                success_count += 1
                tqdm.write(f"✅ {pdb_id}.pdb -> {target_dir}")
                
            except Exception as e:
                error_count += 1
                tqdm.write(f"❌ 复制 {pdb_id}.pdb 到 {target_dir} 失败: {e}")
    
    return success_count, error_count, skipped_count

def main():
    # 配置
    data_dir = "/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train"
    source_dir = "/home/corp/xingqiao.lin/code/GeoStab/pdbs"  # PDB文件源目录
    
    print("🚀 启动PDB文件分发脚本...")
    print("=" * 80)
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        print("请先运行PDB下载脚本")
        return
    
    # 找到所有PDB目录映射
    print("🔍 正在扫描目录结构...")
    pdb_to_dirs = find_pdb_directories(data_dir)
    
    print(f"📊 找到 {len(pdb_to_dirs)} 个唯一的PDB ID")
    print(f"📁 源目录: {source_dir}")
    
    # 统计信息
    total_dirs = sum(len(dirs) for dirs in pdb_to_dirs.values())
    print(f"📊 总共需要分发到 {total_dirs} 个目录")
    print("=" * 80)
    
    # 分发文件
    success_count, error_count, skipped_count = distribute_pdb_files(pdb_to_dirs, source_dir)
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📊 分发完成统计:")
    print("=" * 80)
    print(f"✅ 成功复制: {success_count}")
    print(f"⏭️  跳过 (已存在): {skipped_count}")
    print(f"❌ 复制失败: {error_count}")
    print(f"📊 总计处理: {success_count + skipped_count + error_count}")
    
    # 检查每个PDB ID的分发情况
    print("\n📋 各PDB文件分发详情:")
    print("-" * 50)
    for pdb_id, dirs in pdb_to_dirs.items():
        source_file = os.path.join(source_dir, f"{pdb_id}.pdb")
        if os.path.exists(source_file):
            print(f"✅ {pdb_id}: 分发到 {len(dirs)} 个目录")
        else:
            print(f"❌ {pdb_id}: 源文件不存在")
    
    print("\n🎉 PDB文件分发完成！")

# 运行主函数
main()

ESM-1V 1-5 特征生成
import os, torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForMaskedLM

# ===== 配置 =====
BASE_DIR = "/home/corp/xingqiao.lin/.cache/huggingface/hub/facebook"
MODEL_PREFIX = "esm1v_t33_650M_UR90S_"  # _1 … _5
FASTA = "/path/to/protein.fasta"
SAVE  = "/path/to/save"
DEVICE = "cpu"   # 或 "cuda"
os.makedirs(SAVE, exist_ok=True)

# 读第一条序列
seq = str(next(SeqIO.parse(FASTA, "fasta")).seq)

# tokenizer 各模型相同，只需加载一次
tok_dir = os.path.join(BASE_DIR, f"{MODEL_PREFIX}1")
tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=False, local_files_only=True)

# 分词（自动加 <cls> 与 </s>）
inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

for i in range(1, 6):
    model_dir = os.path.join(BASE_DIR, f"{MODEL_PREFIX}{i}")
    print(f"Loading model {i} from: {model_dir}")

    model = EsmForMaskedLM.from_pretrained(model_dir, local_files_only=True).eval().to(DEVICE)

    with torch.no_grad():
        out  = model(**inputs, output_hidden_states=True)
        last = out.hidden_states[-1]          # [1, L+2, 1280]
        reps = last[0, 1:-1, :].detach()      # 去掉 <cls>, </s> → [L, 1280]
        reps_cpu = reps.to("cpu").clone()

    out_path = os.path.join(SAVE, f"esm1v-{i}.pt")
    torch.save(reps_cpu, out_path)
    print(f"✅ saved: {out_path}  shape={tuple(reps_cpu.shape)}")


# %%
import sys
import os
if not os.path.exists('/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train'):
    os.mkdir('/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train')
import pandas as pd
train=pd.read_csv("/home/corp/xingqiao.lin/code/GeoStab/data/ddG/S8754.csv", sep=',')

names=train['name'].tolist()
for name in names:
    name=name.replace(' ', '_')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/wt_data'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/wt_data')
    if not os.path.exists(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/mut_data'):
        os.mkdir(f'/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train/{name}/mut_data')

# %%
import os
import glob
import shutil
from tqdm import tqdm

def rename_all_pdb_to_relaxed(base_dir):
    """将ddG_train目录下所有子目录中的PDB文件重命名为relaxed.pdb"""
    
    print(f"🔍 开始扫描目录: {base_dir}")
    
    # 查找所有包含wt_data的目录
    wt_data_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "wt_data" in root:
            wt_data_dirs.append(root)
    
    print(f"📊 找到 {len(wt_data_dirs)} 个wt_data目录")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for wt_dir in tqdm(wt_data_dirs, desc="重命名PDB文件"):
        try:
            # 查找目录下的PDB文件
            pdb_files = glob.glob(os.path.join(wt_dir, "*.pdb"))
            
            if not pdb_files:
                print(f"⚠️ {wt_dir} 中没有找到PDB文件")
                skip_count += 1
                continue
            
            # 检查是否已经有relaxed.pdb
            relaxed_pdb = os.path.join(wt_dir, "relaxed.pdb")
            if os.path.exists(relaxed_pdb):
                print(f"⏭️ {os.path.basename(wt_dir)}: 已经有relaxed.pdb，跳过")
                skip_count += 1
                continue
            
            # 如果只有一个PDB文件，直接重命名
            if len(pdb_files) == 1:
                old_pdb = pdb_files[0]
                shutil.move(old_pdb, relaxed_pdb)
                print(f"✅ {os.path.basename(wt_dir)}: {os.path.basename(old_pdb)} -> relaxed.pdb")
                success_count += 1
            
            # 如果有多个PDB文件，选择最大的那个
            elif len(pdb_files) > 1:
                # 按文件大小排序，选择最大的
                pdb_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                old_pdb = pdb_files[0]
                shutil.move(old_pdb, relaxed_pdb)
                print(f"✅ {os.path.basename(wt_dir)}: {os.path.basename(old_pdb)} -> relaxed.pdb (选择最大文件)")
                success_count += 1
                
                # 删除其他PDB文件
                for other_pdb in pdb_files[1:]:
                    os.remove(other_pdb)
                    print(f"🗑️ 删除: {os.path.basename(other_pdb)}")
            
        except Exception as e:
            error_count += 1
            print(f"❌ {os.path.basename(wt_dir)}: 处理出错 - {e}")
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📊 重命名完成统计:")
    print("=" * 80)
    print(f"✅ 成功重命名: {success_count}")
    print(f"⏭️ 跳过: {skip_count}")
    print(f"❌ 处理失败: {error_count}")
    print(f"📊 总计处理: {len(wt_data_dirs)}")
    
    print("\n🎉 PDB文件重命名完成！")

# 运行脚本
base_dir = "/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train"
rename_all_pdb_to_relaxed(base_dir)

# %%
import os
import sys
import subprocess
from tqdm import tqdm
import time

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

def process_wt_features(name, clean_name):
    """处理单个蛋白质的WT特征"""
    wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
    
    # 检查必要文件
    fasta_file = f'{wt_folder}/result.fasta'
    pdb_file = f'{wt_folder}/relaxed.pdb'
    
    if not check_file_exists(fasta_file):
        print(f"⚠️ 跳过 {name}: FASTA文件不存在")
        return False
    
    if not check_file_exists(pdb_file):
        print(f"⚠️ 跳过 {name}: PDB文件不存在")
        return False
    
    success = True
    
    # 1. 生成coordinate.pt
    coord_file = f'{wt_folder}/coordinate.pt'
    if not check_file_exists(coord_file):
        cmd = f"python {GEOSTAB_DIR}/generate_features/coordinate.py --pdb_file {pdb_file} --saved_folder {wt_folder}"
        if not run_command(cmd, f"生成 {name} 的coordinate.pt"):
            success = False
    else:
        print(f"⏭️ {name} coordinate.pt 已存在，跳过")
    
    # 2. 生成pair特征 - 修正参数名
    pair_file = f'{wt_folder}/pair.pt'
    if not check_file_exists(pair_file):
        # 检查coordinate.pt是否存在，因为pair.py需要它
        if check_file_exists(coord_file):
            cmd = f"python {GEOSTAB_DIR}/generate_features/pair.py --coordinate_file {coord_file} --saved_folder {wt_folder}"
            if not run_command(cmd, f"生成 {name} 的pair特征"):
                success = False
        else:
            print(f"⚠️ {name}: coordinate.pt不存在，无法生成pair特征")
            success = False
    else:
        print(f"⏭️ {name} pair.pt 已存在，跳过")
    
    return success

def main():
    """主函数"""
    print("🚀 启动GeoStab WT特征生成脚本...")
    print("📊 只生成coordinate.pt和pair特征")
    print("=" * 80)
    
    # 加载数据
    print("📊 加载数据...")
    names = load_names_from_csv(CSV_FILE)
    print(f"✅ 加载了 {len(names)} 个蛋白质")
    
    # 统计变量
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print("\n🔍 开始处理WT特征...")
    print("=" * 80)
    
    # 处理每个蛋白质
    for name in tqdm(names, desc="生成WT特征"):
        clean_name = name.replace(' ', '_')
        
        try:
            if process_wt_features(name, clean_name):
                success_count += 1
                print(f"✅ {name} WT特征生成完成")
            else:
                skip_count += 1
                print(f"⏭️ {name} 跳过")
        except Exception as e:
            error_count += 1
            print(f"❌ {name} 处理出错: {e}")
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📊 处理完成统计:")
    print("=" * 80)
    print(f"✅ 成功处理: {success_count}")
    print(f"⏭️ 跳过: {skip_count}")
    print(f"❌ 处理失败: {error_count}")
    print(f"📊 总计: {len(names)}")
    
    # 检查生成的文件
    print("\n🔍 检查生成的文件...")
    coord_count = 0
    pair_count = 0
    
    for name in names:
        clean_name = name.replace(' ', '_')
        wt_folder = f'{DATA_DIR}/{clean_name}/wt_data'
        
        if check_file_exists(f'{wt_folder}/coordinate.pt'):
            coord_count += 1
        if check_file_exists(f'{wt_folder}/pair.pt'):
            pair_count += 1
    
    print(f"📊 coordinate.pt 文件数: {coord_count}")
    print(f"📊 pair.pt 文件数: {pair_count}")
    
    print("\n🎉 WT特征生成完成！")

if __name__ == "__main__":
    main()

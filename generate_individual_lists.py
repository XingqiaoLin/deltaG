import click
import os
from pathlib import Path
import re



def parse_mutation_from_name(name):
    """
    从name中解析突变信息
    例如: rcsb_1A0N_B_I121L_7_25 -> (B, 121, I, L)
    """
    parts = name.split('_')
    if len(parts) >= 4:
        chain_id = parts[2]  # B
        mut_info = parts[3]  # I121L
        
        # 提取位置数字
        match = re.search(r'(\d+)', mut_info)
        if match:
            position = int(match.group(1))
            wt_aa = mut_info[0]  # I
            mut_aa = mut_info[-1]  # L
            return chain_id, position, wt_aa, mut_aa
    return None


@click.command()
@click.option("--base_dir", default="/home/corp/xingqiao.lin/code/GeoStab/data/ddG_train", type=str, help="基础输出目录")
@click.option("--pattern", default="rcsb_*", type=str, help="样本目录匹配模式")
@click.option("--force", is_flag=True, help="强制重新生成已存在的individual_list.txt")
def main(base_dir, pattern, force):
    """
    根据目录路径生成individual_list.txt文件
    """
    
    # 设置基础目录
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"❌ 错误: 目录不存在: {base_dir}")
        return
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print(f"📁 扫描目录: {base_dir}")
    print(f"🔍 匹配模式: {pattern}")
    
    # 查找所有匹配的样本目录
    sample_dirs = list(base_dir.glob(pattern))
    if not sample_dirs:
        print(f"❌ 未找到匹配的样本目录")
        return
    
    print(f"📊 找到 {len(sample_dirs)} 个样本目录")
    
    # 处理每个样本目录
    for idx, sample_dir in enumerate(sample_dirs):
        if not sample_dir.is_dir():
            continue
            
        sample_name = sample_dir.name
        print(f"\n处理 {idx+1}/{len(sample_dirs)}: {sample_name}")
        
        # 从样本名称中解析突变信息
        mutation_info = parse_mutation_from_name(sample_name)
        if mutation_info is None:
            print(f"❌ 跳过: 无法从名称解析突变信息")
            error_count += 1
            continue
        
        chain_id, pos_1based, wt_aa, mut_aa = mutation_info
        
        # 检查mut_data目录
        mut_data_dir = sample_dir / "mut_data"
        if not mut_data_dir.exists():
            print(f"❌ 跳过: mut_data目录不存在")
            error_count += 1
            continue
        
        # 检查individual_list.txt是否已存在
        individual_list_file = mut_data_dir / "individual_list.txt"
        if individual_list_file.exists() and not force:
            print(f"⏭️  跳过: individual_list.txt 已存在")
            skip_count += 1
            continue
        
        try:
            # 写入individual_list.txt - 使用正确的格式
            individual_list_content = f"{wt_aa}{chain_id}{pos_1based}{mut_aa};\n"
            with open(individual_list_file, 'w') as f:
                f.write(individual_list_content)
            
            print(f"✅ 成功: 位置{pos_1based} {wt_aa}->{mut_aa} (链{chain_id})")
            print(f"   individual_list.txt: {chain_id}{pos_1based}{wt_aa}{mut_aa};")
            print(f"   文件路径: {individual_list_file}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            error_count += 1
            continue
    
    # 输出统计信息
    print(f"\n🎉 处理完成！")
    print(f"✅ 成功: {success_count}")
    print(f"⏭️  跳过: {skip_count}")
    print(f"❌ 错误: {error_count}")
    print(f"📁 处理目录: {base_dir}")


if __name__ == "__main__":
    main()

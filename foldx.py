#通过foldx，把wild PDB 变为mut PDB
@click.command()
@click.option("--sample_dir", required=True, type=str, help="样本目录路径（包含wt_data和mut_data）")
@click.option("--geostab_dir", default="/home/corp/xingqiao.lin/code/GeoStab", type=str, help="GeoStab项目根目录")
def main(sample_dir, geostab_dir):
    """
    为现有样本目录生成relaxed_repair.pdb文件
    
    使用方法：
    python generate_relaxed_repair.py --sample_dir /path/to/sample
    """
    
    # 设置路径
    geostab_dir = Path(geostab_dir)
    software_foldx = geostab_dir / "foldx" / "foldx_20251231"
    sample_dir = Path(sample_dir)
    
    # 检查目录结构
    wt_folder = sample_dir / "wt_data"
    mut_folder = sample_dir / "mut_data"
    
    if not wt_folder.exists():
        print(f"❌ 错误: wt_data目录不存在: {wt_folder}")
        return
    
    if not mut_folder.exists():
        print(f"❌ 错误: mut_data目录不存在: {mut_folder}")
        return
    
    # 检查必要文件
    wt_pdb = wt_folder / "relaxed.pdb"
    individual_list = mut_folder / "individual_list.txt"
    
    if not wt_pdb.exists():
        print(f"❌ 错误: 野生型PDB文件不存在: {wt_pdb}")
        return
    
    if not individual_list.exists():
        print(f"❌ 错误: individual_list.txt文件不存在: {individual_list}")
        return
    
    # 检查FoldX可执行文件
    if not software_foldx.exists() or not os.access(str(software_foldx), os.X_OK):
        print(f"❌ 错误: FoldX可执行文件不存在或无执行权限: {software_foldx}")
        return
    
    print(f"📁 处理样本目录: {sample_dir}")
    print(f"🔧 使用FoldX: {software_foldx}")
    
    try:
        # 1. 运行FoldX RepairPDB
        print(f"🔧 运行FoldX RepairPDB...")
        foldx_tmp = mut_folder / "foldx_tmp"
        foldx_tmp.mkdir(exist_ok=True)
        
        repair_cmd = [
            str(software_foldx),
            "--command=RepairPDB",
            "--pdb=relaxed.pdb",
            f"--pdb-dir={wt_folder}",
            f"--output-dir={foldx_tmp}"
        ]
        
        print(f"   命令: {' '.join(repair_cmd)}")
        result = subprocess.run(repair_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ RepairPDB失败:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return
        
        # 找到修复后的PDB
        repaired_pdb = find_repaired_pdb(foldx_tmp)
        if not repaired_pdb:
            print(f"❌ 未找到修复后的PDB文件")
            print(f"   foldx_tmp目录内容: {list(foldx_tmp.glob('*'))}")
            return
        
        print(f"✅ 找到修复后的PDB: {repaired_pdb}")
        
        # 2. 复制修复后的PDB到mut_data目录
        relaxed_repair_pdb = mut_folder / "relaxed_repair.pdb"
        shutil.copy2(repaired_pdb, relaxed_repair_pdb)
        print(f"✅ 复制到: {relaxed_repair_pdb}")
        
        # 3. 运行FoldX BuildModel
        print(f"🧬 运行FoldX BuildModel...")
        build_cmd = [
            str(software_foldx),
            "--command=BuildModel",
            "--pdb=relaxed_repair.pdb",
            f"--pdb-dir={mut_folder}",
            f"--mutant-file={individual_list}",
            "--numberOfRuns=3",
            f"--output-dir={foldx_tmp}"
        ]
        
        print(f"   命令: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ BuildModel失败:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return
        
        # 4. 找到生成的突变体PDB
        mut_pdb = find_mutant_pdb(foldx_tmp)
        if not mut_pdb:
            print(f"❌ 未找到突变体PDB文件")
            print(f"   foldx_tmp目录内容: {list(foldx_tmp.glob('*'))}")
            return
        
        print(f"✅ 找到突变体PDB: {mut_pdb}")
        
        # 5. 复制最终结果
        final_mut_pdb = mut_folder / "relaxed_repair.pdb"
        shutil.copy2(mut_pdb, final_mut_pdb)
        print(f"✅ 更新突变体PDB: {final_mut_pdb}")
        
        # 6. 重新编号残基序号
        print(f"🔢 重新编号残基序号...")
        pdb_utils_path = geostab_dir / "tools" / "pdb_utils.py"
        if pdb_utils_path.exists():
            # 使用项目中的pdb_utils.py
            subprocess.run([
                "python", str(pdb_utils_path), 
                str(wt_pdb), str(wt_pdb), "0"
            ])
            subprocess.run([
                "python", str(pdb_utils_path), 
                str(final_mut_pdb), str(final_mut_pdb), "0"
            ])
            print(f"✅ 残基序号重新编号完成")
        else:
            print(f"⚠️  pdb_utils.py不存在，跳过重新编号")
        
        print(f"\n🎉 处理完成！")
        print(f"   WT PDB: {wt_pdb}")
        print(f"   MUT PDB: {final_mut_pdb}")
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


def find_repaired_pdb(foldx_tmp_dir):
    """查找RepairPDB生成的文件"""
    for file in foldx_tmp_dir.glob("*_Repair.pdb"):
        return file
    return None


def find_mutant_pdb(foldx_tmp_dir):
    """查找BuildModel生成的突变体文件"""
    patterns = [
        "*_1_*.pdb",
        "*_1*.pdb"
    ]
    
    for pattern in patterns:
        for file in foldx_tmp_dir.glob(pattern):
            return file
    return None


if __name__ == "__main__":
    main()

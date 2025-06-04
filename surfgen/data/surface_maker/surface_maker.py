import os
import numpy as np
import subprocess
import pymesh
import tempfile, shutil
#import Bio.PDB
from Bio.PDB import PDBParser, PDBIO, Select 
from Bio.PDB import NeighborSearch, Selection
from rdkit import Chem
from scipy.spatial import distance, KDTree
from IPython.utils import io
from joblib import Parallel, delayed
#from deepdock.utils.mol2graph import *

#os.environ["BABEL_LIBDIR"] = "/home/shenchao/.conda/envs/deepdock/lib/openbabel/3.1.0"
#from openbabel import pybel

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append("/data6/jialin/SurfGen/utils/masif")
from compute_normal import compute_normal
from computeAPBS import computeAPBS
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from computeMSMS import computeMSMS
from fixmesh import fix_mesh
from save_ply import save_ply
import glob

# msms_bin="./install_software/APBS-3.0.0.Linux/bin/msms"
# apbs_bin = './install_software/APBS-3.0.0.Linux/bin/apbs'
# pdb2pqr_bin="./install_software/pdb2pqr-linux-bin64-2.1.1/pdb2pqr"
# multivalue_bin="./install_software/APBS-3.0.0.Linux/share/apbs/tools/bin/multivalue"

#os.environ["LD_LIBRARY_PATH"] = '/home/haotian/Molecule_Generation/APBS-3.0.0.Linux/lib:/home/haotian/software/miniconda3/envs/deepdock/lib'
msms_bin="/data6/jialin/SurfDock/comp_surface/tools/transfer/APBS-3.4.1.Linux/bin/msms"
apbs_bin="/data6/jialin/SurfDock/comp_surface/tools/transfer/APBS-3.4.1.Linux/bin/apbs"
pdb2pqr_bin="/data6/jialin/SurfDock/comp_surface/tools/transfer/pdb2pqr-linux-bin64-2.1.1/pdb2pqr"
multivalue_bin="/data6/jialin/SurfDock/comp_surface/tools/transfer/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue"

def compute_inp_surface(prot_path, 
                        lig_path, 
                        outdir='.',
                        dist_threshold=8.0,
                        use_hbond=True,
                        use_hphob=True,
                        use_apbs=True,
                        compute_iface=True,
                        mesh_res=1.0, # 源代码这里用的1.0而非1.5
                        epsilon=1.0e-6,
                        feature_interpolation=True,						
                        out_name= None):    
    workdir = tempfile.mkdtemp()
    protname = os.path.basename(prot_path).replace(".pdb","")
    # Get atom coordinates
    suffix = lig_path.split('.')[-1]
    if suffix == 'mol':
        mol = Chem.MolFromMolFile(lig_path)
    elif suffix == 'mol2':
        mol = Chem.MolFromMol2File(lig_path)
    elif suffix == 'sdf':
        suppl = Chem.SDMolSupplier(lig_path, sanitize=False)
        mols = [mol for mol in suppl if mol]	
        mol = mols[0]
        # we just use the first mol of .sdf file by default
    else:
        raise Exception("Invalid ligand file type! Just support .mol, .mol2, .sdf")
    #g = mol_to_nx(mol)
    #atomCoords = np.array([g.nodes[i]['pos'].tolist() for i in g.nodes])
    atomCoords = mol.GetConformers()[0].GetPositions()
    # Read protein and select aminino acids in the binding pocket
    parser = PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.
    
    structures = parser.get_structure('target', prot_path)
    structure = structures[0] # 'structures' may contain several proteins in this case only one.
    
    atoms  = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atoms)
    
    close_residues= []
    for a in atomCoords:  
        close_residues.extend(ns.search(a, dist_threshold, level='R'))
    close_residues = Selection.uniqueify(close_residues)
    
    class SelectNeighbors(Select):
        def accept_residue(self, residue):
            if residue in close_residues:
                if all(a in [i.get_name() for i in residue.get_unpacked_list()] for a in ['N', 'CA', 'C', 'O']) or residue.resname=='HOH':
                    return True
                else:
                    return False
            else:
                return False
        
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save("%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold), SelectNeighbors())
    
    # Identify closes atom to the ligand
    structures = parser.get_structure('target', "%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold))
    structure = structures[0] # 'structures' may contain several proteins in this case only one.
    atoms = Selection.unfold_entities(structure, 'A')
    
    try:
        dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
        atom_idx = np.argmin(dist)
        vertices1, faces1, normals1, names1, areas1 = computeMSMS("%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold),  
                                                                    protonate=True, 
                                                                    one_cavity=atom_idx, 
                                                                    msms_bin=msms_bin,
                                                                    workdir=workdir)
                                                                           
        # Find the distance between every vertex in binding site surface and each atom in the ligand.
        kdt = KDTree(atomCoords)
        d, r = kdt.query(vertices1)
        assert(len(d) == len(vertices1))
        iface_v = np.where(d <= dist_threshold - 5)[0]
        faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
        
        # Compute "charged" vertices
        if use_hbond:
            vertex_hbond = computeCharges(prot_path.replace(".pdb",""), vertices1, names1)    
        
        # For each surface residue, assign the hydrophobicity of its amino acid. 
        if use_hphob:
            vertex_hphobicity = computeHydrophobicity(names1) 
        
        vertices2 = vertices1
        faces2 = faces1
        # Fix the mesh.
        mesh = pymesh.form_mesh(vertices2, faces2)
        mesh = pymesh.submesh(mesh, faces_to_keep, 0)
        with io.capture_output() as captured:
            regular_mesh = fix_mesh(mesh, mesh_res)
    
    except:
        try:
            dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
            atom_idx = np.argsort(np.min(dist, axis=1))[0]
            vertices1, faces1, normals1, names1, areas1 = computeMSMS("%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold),  
                                                                        protonate=True, 
                                                                        one_cavity=atom_idx, 
                                                                        msms_bin=msms_bin,
                                                                        workdir=workdir)
                                                                           
            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold - 5)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
            
            # Compute "charged" vertices
            if use_hbond:
                vertex_hbond = computeCharges(prot_path.replace(".pdb",""), vertices1, names1)    
                
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if use_hphob:
                vertex_hphobicity = computeHydrophobicity(names1) 
            
            vertices2 = vertices1
            faces2 = faces1
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, mesh_res)
        except:
            vertices1, faces1, normals1, names1, areas1 = computeMSMS("%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold),  
                                                                        protonate=True, 
                                                                        one_cavity=None, 
                                                                        msms_bin=msms_bin,
                                                                        workdir=workdir)
            
            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold - 5)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
            
            # Compute "charged" vertices
            if use_hbond:
                vertex_hbond = computeCharges(prot_path.replace(".pdb",""), vertices1, names1)    
                
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if use_hphob:
                vertex_hphobicity = computeHydrophobicity(names1) 
            
            vertices2 = vertices1
            faces2 = faces1
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, mesh_res)
    
    ## resolve all degeneracies 源代码没有这个
    regular_mesh, info = pymesh.remove_degenerated_triangles(regular_mesh)
    
    # Compute the normals 源代码没有epsilon
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces, eps=epsilon)
    # vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    
    
    # Assign charges on new vertices based on charges of old vertices (nearest neighbor)
    if use_hbond:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, feature_interpolation)
    
    if use_hphob:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, feature_interpolation)
    
    if use_apbs:
        vertex_charges = computeAPBS(regular_mesh.vertices, "%s/%s_pocket_%s.pdb"%(workdir, protname, dist_threshold), 
                                    apbs_bin, pdb2pqr_bin, multivalue_bin, workdir)
    
    # Compute the principal curvature components for the shape index. 
    regular_mesh.add_attribute("vertex_mean_curvature")
    H = regular_mesh.get_attribute("vertex_mean_curvature")
    regular_mesh.add_attribute("vertex_gaussian_curvature")
    K = regular_mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)
    
    # Step 1: Create subfolder using pdb_id or out_name
    pdb_id = os.path.basename(prot_path)[:4].lower()
    if out_name is None:
        subfolder_name = pdb_id
    else:
        subfolder_name = out_name
    subfolder_path = os.path.join(outdir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Step 2: Move the processed pocket PDB to subfolder
    pocket_pdb_filename = os.path.join(workdir, f"{protname}_pocket_{dist_threshold}.pdb")
    pdbio.save(pocket_pdb_filename, SelectNeighbors())
    renamed_pdb_filename = os.path.join(subfolder_path, f"{pdb_id}_protein_processed_{int(dist_threshold)}A.pdb")
    shutil.copy(pocket_pdb_filename, renamed_pdb_filename)

    # Step 3: Save PLY
    ply_filename = os.path.join(subfolder_path, f"{subfolder_name}_protein_processed_{dist_threshold}A.ply")
    save_ply(ply_filename, regular_mesh.vertices, regular_mesh.faces,
            normals=vertex_normal, charges=vertex_charges,
            normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity, si=si)

    # Step 4: 清理临时文件
    shutil.rmtree(workdir)
    
def collect_protein_ligand_pairs(data_dir):
    """遍历数据目录，收集所有蛋白-配体文件路径"""
    pairs = []
    for item in os.listdir(data_dir):
        item_dir = os.path.join(data_dir, item)
        if not os.path.isdir(item_dir):
            continue
        protein_path = os.path.join(item_dir, f"{item}_protein.pdb")
        ligand_path = os.path.join(item_dir, f"{item}_ligand.sdf")
        if not os.path.exists(ligand_path):
            ligand_path = os.path.join(item_dir, f"{item}_ligand.mol2")
        if os.path.exists(protein_path) and os.path.exists(ligand_path):
            pairs.append((protein_path, ligand_path))
    return pairs

def process_all(data_dir, out_dir, dist_threshold=8, n_jobs=1):
    """主处理流程：收集配对，调用 compute_inp_surface，并行处理"""
    os.makedirs(out_dir, exist_ok=True)
    pairs = collect_protein_ligand_pairs(data_dir)
    print(f"总共需要处理的蛋白-配体对数量: {len(pairs)}")

    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(compute_inp_surface)(protein, ligand, out_dir, dist_threshold)
        for protein, ligand in pairs
    )

    failed = [res for res in results if res != 0]
    print(f"\n成功: {len(pairs) - len(failed)}，失败: {len(failed)}")
    if failed:
        print("失败项如下：")
        for f in failed:
            print(f)

    # 清理临时文件
    clean_temp_files(out_dir)

def clean_temp_files(out_dir):
    """删除 _temp 和 msms 临时文件"""
    files = glob.glob(os.path.join(out_dir, '*_temp*')) + glob.glob(os.path.join(out_dir, '*msms*'))
    for f in files:
        try:
            os.remove(f)
        except:
            pass

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='输入数据根目录')
    parser.add_argument('--out_dir', type=str, required=True, help='输出 .ply 存储目录')
    parser.add_argument('--dist_threshold', type=int, default=8, help='配体到蛋白的距离阈值')
    parser.add_argument('--n_jobs', type=int, default=30, help='并行处理线程数')
    args = parser.parse_args()

    process_all(args.data_dir, args.out_dir, args.dist_threshold, args.n_jobs)
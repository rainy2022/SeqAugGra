from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

ATOM_LIST = list(range(1,120)) # Includes mask token
NUM_ATOM_TYPE = len(ATOM_LIST)

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
NUM_CHIRALITY_TAG = len(CHIRALITY_LIST)

CHARGE_LIST = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
NUM_CHARGE_LIST = len(CHARGE_LIST)

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED
]
NUM_HYBRIDIZATION_LIST = len(HYBRIDIZATION_LIST)

NUM_H_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8]
NUM_NUM_H_LIST = len(NUM_H_LIST)

VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6]
NUM_VALENCE_LIST = len(VALENCE_LIST)

NUM_AROMATIC = [0, 1]
NUM_AROMATIC_LIST = len(NUM_AROMATIC)

NUM_HYDROGEN = [0, 1, 2, 3, 4, 5]
NUM_NUM_HYDROGEN_LIST = len(NUM_HYDROGEN)

DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_DEGREE_LIST=len(DEGREE_LIST)

BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC,
    BT.UNSPECIFIED,
]
NUM_BOND_TYPE = len(BOND_LIST) + 1 # including aromatic and self-loop edge


BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]
NUM_BOND_DIRECTION = len(BONDDIR_LIST)
STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS
]
NUM_STEREO_LIST = len(STEREO_LIST)
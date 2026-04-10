"""
KLIFS Database Loader
=====================

Loads kinase structures from KLIFS (Kinase-Ligand Interaction Fingerprints and Structures)
database with DFG conformation annotations.

KLIFS provides:
- 15,000+ kinase-ligand structures
- Standardized 85-residue binding pocket alignment
- DFG and C-helix conformation annotations
- REST API for programmatic access

Reference: https://klifs.net/
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class DFGConformation(Enum):
    """DFG motif conformational states."""
    IN = "in"
    OUT = "out"
    OUT_LIKE = "out-like"
    IN_LIKE = "in-like"
    NA = "na"
    
    @classmethod
    def from_string(cls, s: Optional[str]) -> 'DFGConformation':
        if s is None:
            return cls.NA
        s_lower = s.lower().strip()
        if s_lower == "in":
            return cls.IN
        elif s_lower == "out":
            return cls.OUT
        elif "out" in s_lower:
            return cls.OUT_LIKE
        elif "in" in s_lower:
            return cls.IN_LIKE
        return cls.NA
    
    def is_dfg_in(self) -> bool:
        return self in [DFGConformation.IN, DFGConformation.IN_LIKE]
    
    def is_dfg_out(self) -> bool:
        return self in [DFGConformation.OUT, DFGConformation.OUT_LIKE]


class CHelixConformation(Enum):
    """αC-helix conformational states."""
    IN = "in"
    OUT = "out"
    OUT_LIKE = "out-like"
    NA = "na"
    
    @classmethod
    def from_string(cls, s: Optional[str]) -> 'CHelixConformation':
        if s is None:
            return cls.NA
        s_lower = s.lower().strip()
        if s_lower == "in":
            return cls.IN
        elif s_lower == "out":
            return cls.OUT
        elif "out" in s_lower:
            return cls.OUT_LIKE
        return cls.NA


@dataclass
class KLIFSKinase:
    """Kinase information from KLIFS."""
    kinase_id: int
    name: str
    full_name: str = ""
    family: str = ""
    group: str = ""
    uniprot: str = ""
    species: str = "Human"
    
    def __hash__(self):
        return hash(self.kinase_id)


@dataclass
class KLIFSStructure:
    """Structure information from KLIFS."""
    structure_id: int
    kinase_id: int
    pdb_id: str
    chain: str
    dfg: DFGConformation
    ac_helix: CHelixConformation
    pocket_sequence: str  # 85 residue KLIFS pocket
    resolution: Optional[float] = None
    quality_score: Optional[float] = None
    ligand: Optional[str] = None
    ligand_pdb: Optional[str] = None
    
    # 3D coordinates (loaded separately)
    pocket_coords: Optional[np.ndarray] = None  # [85, 3] CA coords
    
    @property
    def is_dfg_in(self) -> bool:
        return self.dfg.is_dfg_in()
    
    @property
    def is_dfg_out(self) -> bool:
        return self.dfg.is_dfg_out()


@dataclass
class ConformationalPair:
    """Paired structures of same kinase in different conformations."""
    kinase: KLIFSKinase
    structure_in: KLIFSStructure
    structure_out: KLIFSStructure
    ligand: Optional[str] = None
    pki_in: Optional[float] = None
    pki_out: Optional[float] = None
    
    @property
    def delta_pki(self) -> Optional[float]:
        if self.pki_in is not None and self.pki_out is not None:
            return self.pki_in - self.pki_out
        return None


# =============================================================================
# KLIFS API Client
# =============================================================================

class KLIFSClient:
    """
    Client for KLIFS REST API.
    
    API v2 Documentation: https://klifs.net/swagger/
    """
    
    BASE_URL = "https://klifs.net/api_v2"
    CACHE_DIR = Path("data/klifs_cache")
    
    # Amino acid vocabulary for encoding
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize KLIFS client."""
        self.cache_enabled = cache_enabled
        
        if cache_enabled:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        logger.info("Initialized KLIFS client")
    
    def _cache_path(self, endpoint: str, params: dict = None) -> Path:
        """Generate cache file path."""
        key = f"{endpoint}_{hash(str(sorted((params or {}).items())))}"
        cache_key = hashlib.md5(key.encode()).hexdigest()
        return self.CACHE_DIR / f"{cache_key}.json"
    
    def _request(self, endpoint: str, params: dict = None) -> Any:
        """Make API request with caching."""
        params = params or {}
        
        # Check cache
        if self.cache_enabled:
            cache_path = self._cache_path(endpoint, params)
            if cache_path.exists():
                try:
                    with open(cache_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
        
        # Make request
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache response
            if self.cache_enabled:
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"KLIFS API error: {endpoint} - {e}")
            return []
    
    # =========================================================================
    # API Methods
    # =========================================================================
    
    def get_kinase_names(self, species: str = "Human") -> List[KLIFSKinase]:
        """Get list of all kinases."""
        data = self._request("kinase_names", {"species": species})
        
        kinases = []
        for item in data:
            kinases.append(KLIFSKinase(
                kinase_id=item.get('kinase_ID', 0),
                name=item.get('name', ''),
                full_name=item.get('full_name', ''),
                family=item.get('family', ''),
                group=item.get('group', ''),
                uniprot=item.get('uniprot', ''),
                species=species
            ))
        
        return kinases
    
    def get_kinase_info(self, kinase_id: int) -> Optional[KLIFSKinase]:
        """Get detailed kinase information."""
        data = self._request("kinase_information", {"kinase_ID": kinase_id})
        
        if data and len(data) > 0:
            item = data[0]
            return KLIFSKinase(
                kinase_id=item.get('kinase_ID', kinase_id),
                name=item.get('name', ''),
                full_name=item.get('full_name', ''),
                family=item.get('family', ''),
                group=item.get('group', ''),
                uniprot=item.get('uniprot', ''),
                species=item.get('species', 'Human')
            )
        return None
    
    def get_structures(
        self,
        kinase_id: Optional[int] = None,
        pdb_codes: Optional[List[str]] = None
    ) -> List[KLIFSStructure]:
        """Get structures for a kinase or PDB codes."""
        if kinase_id:
            # Use structures_list endpoint (requires kinase_ID)
            data = self._request("structures_list", {'kinase_ID': [kinase_id]})
        elif pdb_codes:
            # Use structures_pdb_list endpoint
            data = self._request("structures_pdb_list", {'pdb-codes': pdb_codes})
        else:
            return []
        
        if not data or isinstance(data, dict):
            return []
        
        structures = []
        for item in data:
            structures.append(KLIFSStructure(
                structure_id=item.get('structure_ID', 0),
                kinase_id=item.get('kinase_ID', 0),
                pdb_id=item.get('pdb', ''),
                chain=item.get('chain', ''),
                dfg=DFGConformation.from_string(item.get('DFG')),
                ac_helix=CHelixConformation.from_string(item.get('aC_helix')),
                pocket_sequence=item.get('pocket', ''),
                resolution=item.get('resolution'),
                quality_score=item.get('quality_score'),
                ligand=item.get('ligand'),
                ligand_pdb=item.get('ligand_PDB')
            ))
        
        return structures
    
    def get_pocket_coordinates(
        self,
        structure_id: int,
        pocket_seq: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Get Cα coordinates for the 85 KLIFS pocket residues.

        Parses the cached KLIFS complex PDB and returns shape (85, 3).
        Gap positions get [0, 0, 0].
        """
        from pathlib import Path as _Path

        pdb_cache = _Path("data/klifs_pockets") / str(structure_id) / "complex.pdb"
        if not pdb_cache.exists():
            # Try downloading via API
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/structure_get_pdb_complex",
                    params={"structure_ID": structure_id},
                    timeout=30,
                )
                resp.raise_for_status()
                pdb_cache.parent.mkdir(parents=True, exist_ok=True)
                pdb_cache.write_text(resp.text)
            except Exception as e:
                logger.warning("Failed to fetch PDB for %d: %s", structure_id, e)
                return None

        pdb_text = pdb_cache.read_text()

        # Parse Cα coordinates
        ca_coords = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                if line[16:17] not in (" ", "A"):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
                except ValueError:
                    continue

        if not ca_coords:
            return None

        all_ca = np.array(ca_coords, dtype=np.float32)

        # Get pocket sequence if not provided
        if pocket_seq is None:
            structures = self.get_structures(kinase_id=None, pdb_codes=[])
            # Fallback: use all Cα, padded/truncated to 85
            pocket_seq = "A" * min(len(all_ca), 85)

        # Map resolved pocket positions to Cα coords
        result = np.zeros((85, 3), dtype=np.float32)
        n_resolved = sum(1 for ch in pocket_seq if ch not in ('-', '_', ' ', ''))
        available = min(n_resolved, len(all_ca))

        resolved_count = 0
        for pos, ch in enumerate(pocket_seq[:85]):
            if ch not in ('-', '_', ' ', ''):
                if resolved_count < available:
                    result[pos] = all_ca[resolved_count]
                resolved_count += 1

        return result
    
    # =========================================================================
    # High-Level Methods
    # =========================================================================
    
    def get_kinases_with_both_conformations(
        self,
        min_structures: int = 2
    ) -> List[Dict]:
        """
        Find kinases that have structures in both DFG-in and DFG-out.
        
        This is the key data for conformational selectivity prediction.
        """
        kinases = self.get_kinase_names()
        
        results = []
        for kinase in kinases:
            structures = self.get_structures(kinase_id=kinase.kinase_id)
            
            dfg_in = [s for s in structures if s.is_dfg_in]
            dfg_out = [s for s in structures if s.is_dfg_out]
            
            if len(dfg_in) >= min_structures and len(dfg_out) >= min_structures:
                results.append({
                    'kinase': kinase,
                    'n_dfg_in': len(dfg_in),
                    'n_dfg_out': len(dfg_out),
                    'structures_in': dfg_in,
                    'structures_out': dfg_out,
                    'total': len(dfg_in) + len(dfg_out)
                })
        
        # Sort by total structures
        results.sort(key=lambda x: x['total'], reverse=True)
        
        logger.info(f"Found {len(results)} kinases with both conformations")
        return results
    
    def create_conformational_pairs(
        self,
        kinase_data: Dict,
        max_pairs: int = 10
    ) -> List[ConformationalPair]:
        """Create paired structures for a kinase."""
        pairs = []
        
        kinase = kinase_data['kinase']
        structures_in = kinase_data['structures_in']
        structures_out = kinase_data['structures_out']
        
        # Pair structures (prefer same ligand if available)
        for s_in in structures_in[:max_pairs]:
            for s_out in structures_out[:max_pairs]:
                pairs.append(ConformationalPair(
                    kinase=kinase,
                    structure_in=s_in,
                    structure_out=s_out,
                    ligand=s_in.ligand_pdb if s_in.ligand_pdb == s_out.ligand_pdb else None
                ))
        
        return pairs
    
    # =========================================================================
    # Feature Extraction
    # =========================================================================
    
    def encode_pocket_sequence(self, sequence: str) -> np.ndarray:
        """Encode 85-residue pocket as integer array."""
        # Ensure 85 residues
        seq = sequence[:85].ljust(85, '-')
        return np.array([self.AA_TO_IDX.get(aa.upper(), 21) for aa in seq], dtype=np.int64)
    
    @staticmethod
    def extract_structural_features(
        coords_in: np.ndarray,
        coords_out: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract structural features from conformational pair.
        
        These are the features that make structure necessary for prediction.
        """
        # Coordinate difference
        diff = coords_out - coords_in
        
        # DFG region (KLIFS residues 79-83)
        dfg_diff = diff[79:84]
        dfg_magnitude = np.sqrt((dfg_diff ** 2).sum(axis=-1)).mean()
        
        # C-helix region (KLIFS residues 20-30)
        chelix_diff = diff[20:31]
        chelix_magnitude = np.sqrt((chelix_diff ** 2).sum(axis=-1)).mean()
        
        # Hinge region (KLIFS residues 46-52)
        hinge_diff = diff[46:53]
        hinge_magnitude = np.sqrt((hinge_diff ** 2).sum(axis=-1)).mean()
        
        # Overall RMSD
        rmsd = np.sqrt((diff ** 2).mean())
        
        # Gatekeeper (KLIFS residue 45)
        gk_diff = np.linalg.norm(diff[45])
        
        return {
            'dfg_flip_magnitude': float(dfg_magnitude),
            'chelix_shift': float(chelix_magnitude),
            'hinge_shift': float(hinge_magnitude),
            'gatekeeper_shift': float(gk_diff),
            'overall_rmsd': float(rmsd),
        }


# =============================================================================
# Known Drug Data
# =============================================================================

# Drug type classification with PubChem CIDs
KNOWN_KINASE_INHIBITORS = {
    # Type I (DFG-in binders)
    'Gefitinib': {
        'cid': 123631,
        'type': 'I',
        'targets': ['EGFR'],
        'dfg_preference': 'in',
        'expected_delta_pki': 'positive',  # prefers DFG-in
    },
    'Erlotinib': {
        'cid': 176870,
        'type': 'I',
        'targets': ['EGFR'],
        'dfg_preference': 'in',
        'expected_delta_pki': 'positive',
    },
    'Dasatinib': {
        'cid': 3062316,
        'type': 'I',
        'targets': ['ABL1', 'SRC'],
        'dfg_preference': 'in',
        'expected_delta_pki': 'positive',
    },
    
    # Type II (DFG-out binders)
    'Imatinib': {
        'cid': 5291,
        'type': 'II',
        'targets': ['ABL1', 'KIT', 'PDGFR'],
        'dfg_preference': 'out',
        'expected_delta_pki': 'negative',  # prefers DFG-out
    },
    'Sorafenib': {
        'cid': 216239,
        'type': 'II',
        'targets': ['RAF1', 'BRAF', 'VEGFR2'],
        'dfg_preference': 'out',
        'expected_delta_pki': 'negative',
    },
    'Nilotinib': {
        'cid': 644241,
        'type': 'II',
        'targets': ['ABL1'],
        'dfg_preference': 'out',
        'expected_delta_pki': 'negative',
    },
    'Lapatinib': {
        'cid': 208908,
        'type': 'I',  # Actually Type I but binds inactive-like
        'targets': ['EGFR', 'ERBB2'],
        'dfg_preference': 'in',  # but with C-helix out
        'expected_delta_pki': 'positive',
    },
}


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing KLIFS Client...")
    
    client = KLIFSClient()
    
    # Test kinase listing
    print("\n[1] Fetching kinase list...")
    kinases = client.get_kinase_names()
    print(f"  Found {len(kinases)} human kinases")
    
    if kinases:
        # Test structure fetching for ABL1
        print("\n[2] Fetching structures for ABL1...")
        abl1 = next((k for k in kinases if 'ABL1' in k.name.upper()), None)
        if abl1:
            structures = client.get_structures(kinase_id=abl1.kinase_id)
            dfg_in = [s for s in structures if s.is_dfg_in]
            dfg_out = [s for s in structures if s.is_dfg_out]
            print(f"  ABL1: {len(structures)} total, {len(dfg_in)} DFG-in, {len(dfg_out)} DFG-out")
    
    # Test finding kinases with both conformations
    print("\n[3] Finding kinases with both DFG-in and DFG-out...")
    dual_conf = client.get_kinases_with_both_conformations(min_structures=3)
    print(f"  Found {len(dual_conf)} kinases with ≥3 structures in each conformation")
    
    if dual_conf:
        print("\n  Top 5 kinases:")
        for d in dual_conf[:5]:
            print(f"    {d['kinase'].name}: {d['n_dfg_in']} DFG-in, {d['n_dfg_out']} DFG-out")
    
    print("\n✓ KLIFS client working")

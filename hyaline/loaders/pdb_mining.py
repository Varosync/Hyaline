"""
PDB Mining for TF-DNA Complex Structures
=========================================

Query and download transcription factor-DNA complex structures from RCSB PDB.
Uses the RCSB Search API v2 for programmatic access.

Based on Nectar research findings:
- API endpoint: https://search.rcsb.org/rcsbsearch/v2/query
- Format: JSON
- Available filters: entity_type, experimental_method, resolution, etc.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import urllib.request
import urllib.error


# RCSB API endpoint (from Nectar extraction)
RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

# Known TF-DNA complexes from Nectar research
SEED_TF_DNA_STRUCTURES = [
    {"pdb_id": "1aln", "tf_name": "CAP", "domain": "helix-turn-helix", "resolution": 1.9},
    {"pdb_id": "1byb", "tf_name": "AP-1 (c-Fos/c-Jun)", "domain": "basic leucine zipper", "resolution": 2.4},
    {"pdb_id": "1cgp", "tf_name": "GCN4", "domain": "basic leucine zipper", "resolution": 2.0},
    {"pdb_id": "1ig7", "tf_name": "Mnt", "domain": "helix-loop-helix", "resolution": 2.1},
    {"pdb_id": "1jfi", "tf_name": "Zif268", "domain": "C2H2 zinc finger", "resolution": 1.9},
    {"pdb_id": "1j59", "tf_name": "p53", "domain": "DNA binding domain", "resolution": 2.15},
]

# Validation pairs from system spec
VALIDATION_PAIRS = [
    {"target": "BRD4", "drug": "JQ1", "pdb_id": "3MXF"},
    {"target": "MDM2", "drug": "Nutlin-3a", "pdb_id": "4HG7"},
    {"target": "BCL6", "drug": "BI-3812", "pdb_id": "6F4P"},
    {"target": "STAT3", "drug": "OPB-31121", "pdb_id": "6NJS"},
    {"target": "ETV6", "drug": "BRD32048", "pdb_id": None},  # No known structure
]


@dataclass
class PDBEntry:
    """Metadata for a PDB structure."""
    pdb_id: str
    resolution: float
    title: str = ""
    organism: str = ""
    has_dna: bool = False
    has_protein: bool = False
    tf_name: Optional[str] = None
    domain_type: Optional[str] = None
    chains: List[str] = field(default_factory=list)


@dataclass
class MiningConfig:
    """Configuration for PDB mining."""
    max_resolution: float = 2.5  # Angstroms
    min_length: int = 50  # Minimum protein length
    max_structures: int = 500
    include_organisms: List[str] = field(default_factory=lambda: [
        "Homo sapiens",
        "Mus musculus",
        "Saccharomyces cerevisiae",
    ])
    exclude_keywords: List[str] = field(default_factory=lambda: [
        "chimeric",
        "synthetic",
        "designed",
    ])


class PDBMiner:
    """
    Mine TF-DNA complex structures from RCSB PDB.
    
    Uses the RCSB Search API to find structures containing both
    protein and DNA with high resolution.
    """
    
    def __init__(
        self,
        data_dir: str = "data/pdb",
        config: Optional[MiningConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or MiningConfig()
        
        self._cache_file = self.data_dir / "pdb_cache.json"
        self._cache: Dict[str, PDBEntry] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cached PDB metadata."""
        if self._cache_file.exists():
            with open(self._cache_file) as f:
                data = json.load(f)
                for pdb_id, entry_data in data.items():
                    self._cache[pdb_id] = PDBEntry(**entry_data)
    
    def _save_cache(self):
        """Save PDB metadata cache."""
        data = {
            pdb_id: {
                "pdb_id": entry.pdb_id,
                "resolution": entry.resolution,
                "title": entry.title,
                "organism": entry.organism,
                "has_dna": entry.has_dna,
                "has_protein": entry.has_protein,
                "tf_name": entry.tf_name,
                "domain_type": entry.domain_type,
                "chains": entry.chains,
            }
            for pdb_id, entry in self._cache.items()
        }
        with open(self._cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def build_search_query(self) -> Dict:
        """
        Build RCSB Search API query for TF-DNA complexes.
        
        Returns JSON query following RCSB Search API v2 format.
        """
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    # Must have DNA
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_DNA",
                            "operator": "greater",
                            "value": 0
                        }
                    },
                    # Must have protein
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                            "operator": "greater",
                            "value": 0
                        }
                    },
                    # X-ray structures only
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "X-RAY DIFFRACTION"
                        }
                    },
                    # Resolution filter
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": self.config.max_resolution
                        }
                    },
                    # Transcription-related text
                    {
                        "type": "terminal",
                        "service": "full_text",
                        "parameters": {
                            "value": "transcription factor DNA binding"
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": self.config.max_structures
                },
                "sort": [
                    {
                        "sort_by": "rcsb_entry_info.resolution_combined",
                        "direction": "asc"
                    }
                ]
            }
        }
        return query
    
    def search_rcsb(self) -> List[str]:
        """
        Execute search query against RCSB API.
        
        Returns:
            List of PDB IDs matching the query
        """
        query = self.build_search_query()
        
        try:
            req = urllib.request.Request(
                RCSB_SEARCH_API,
                data=json.dumps(query).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                
            pdb_ids = [hit["identifier"] for hit in result.get("result_set", [])]
            print(f"Found {len(pdb_ids)} TF-DNA complex structures")
            return pdb_ids
            
        except urllib.error.URLError as e:
            print(f"RCSB API error: {e}")
            # Fall back to seed structures
            return [s["pdb_id"] for s in SEED_TF_DNA_STRUCTURES]
    
    def download_pdb(self, pdb_id: str, format: str = "pdb") -> Optional[Path]:
        """
        Download a PDB file.
        
        Args:
            pdb_id: 4-letter PDB code
            format: 'pdb' or 'cif'
            
        Returns:
            Path to downloaded file, or None if failed
        """
        pdb_id = pdb_id.lower()
        ext = "pdb" if format == "pdb" else "cif"
        filename = f"{pdb_id}.{ext}"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            return filepath
        
        url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.{ext}"
        
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {pdb_id}")
            return filepath
        except urllib.error.URLError as e:
            print(f"Failed to download {pdb_id}: {e}")
            return None
    
    def download_all(
        self,
        pdb_ids: Optional[List[str]] = None,
        delay: float = 0.5,
    ) -> List[Path]:
        """
        Download multiple PDB files.
        
        Args:
            pdb_ids: List of PDB IDs (or None to use search results)
            delay: Delay between downloads to avoid rate limiting
            
        Returns:
            List of successfully downloaded file paths
        """
        if pdb_ids is None:
            pdb_ids = self.search_rcsb()
        
        downloaded = []
        for pdb_id in pdb_ids:
            filepath = self.download_pdb(pdb_id)
            if filepath:
                downloaded.append(filepath)
            time.sleep(delay)
        
        print(f"Successfully downloaded {len(downloaded)}/{len(pdb_ids)} structures")
        return downloaded
    
    def get_validation_structures(self) -> List[Path]:
        """Download structures for retrospective validation."""
        validation_ids = [
            pair["pdb_id"] for pair in VALIDATION_PAIRS 
            if pair["pdb_id"] is not None
        ]
        return self.download_all(validation_ids)
    
    def get_seed_structures(self) -> List[Path]:
        """Download seed TF-DNA structures from Nectar research."""
        seed_ids = [s["pdb_id"] for s in SEED_TF_DNA_STRUCTURES]
        return self.download_all(seed_ids)


def quick_mine(
    output_dir: str = "data/tf_dna",
    max_structures: int = 100,
) -> List[str]:
    """
    Quick mining function for TF-DNA structures.
    
    Args:
        output_dir: Directory to save PDB files
        max_structures: Maximum number of structures to download
        
    Returns:
        List of downloaded PDB IDs
    """
    config = MiningConfig(max_structures=max_structures)
    miner = PDBMiner(data_dir=output_dir, config=config)
    
    paths = miner.download_all()
    return [p.stem for p in paths]


if __name__ == "__main__":
    print("TF-DNA PDB Mining")
    print("=" * 50)
    
    # Initialize miner
    miner = PDBMiner(data_dir="data/tf_dna")
    
    print("\nSeed structures from Nectar research:")
    for s in SEED_TF_DNA_STRUCTURES[:3]:
        print(f"  {s['pdb_id']}: {s['tf_name']} ({s['domain']}) - {s['resolution']}Å")
    
    print("\nValidation pairs from spec:")
    for v in VALIDATION_PAIRS[:3]:
        print(f"  {v['target']} + {v['drug']} → {v['pdb_id'] or 'no structure'}")
    
    # Test API query build
    query = miner.build_search_query()
    print(f"\nSearch query nodes: {len(query['query']['nodes'])}")
    
    # Download seed structures
    print("\nDownloading seed structures...")
    paths = miner.get_seed_structures()
    print(f"Downloaded: {len(paths)} files")
    
    print("\n✓ PDB miner ready!")
